#include "turbo_ocr/detection/paddle_det.h"
#include "turbo_ocr/kernels/kernels.h"
#include "turbo_ocr/detection/det_postprocess.h"

#include "turbo_ocr/common/errors.h"

#include <algorithm>
#include <cmath>
#include <cstdlib>
#include <ranges>

#include <opencv2/imgproc.hpp>

using turbo_ocr::engine::TrtEngine;

namespace turbo_ocr::detection {

bool PaddleDet::load_model(const std::string &model_path) {
  engine_ = std::make_unique<TrtEngine>(model_path);
  if (!engine_->load())
    return false;
  return init_buffers();
}

bool PaddleDet::init_buffers() {
  // Read configurable max side length (clamped to [32, 4096] to prevent
  // integer overflow in max_pixels = kMaxSideLen_^2 and downstream allocs)
  if (const char *env = std::getenv("DET_MAX_SIDE"))
    kMaxSideLen_ = std::atoi(env);
  kMaxSideLen_ = std::clamp(kMaxSideLen_, 32, 4096);

  size_t max_pixels = static_cast<size_t>(kMaxSideLen_) * kMaxSideLen_;
  input_size_ = max_pixels * 3 * sizeof(float);
  output_size_ = max_pixels * sizeof(float);

  d_input_ = CudaPtr<float>(max_pixels * 3);
  d_output_ = CudaPtr<float>(max_pixels);

  // Pre-allocate bitmap buffer (960x960 uint8)
  d_bitmap_buf_ = CudaPtr<uint8_t>(max_pixels);

  // Pre-allocate batch buffers (kMaxBatchSize x max_pixels)
  batch_input_size_ = static_cast<size_t>(kMaxBatchSize) * max_pixels * 3 * sizeof(float);
  batch_output_size_ = static_cast<size_t>(kMaxBatchSize) * max_pixels * sizeof(float);
  d_batch_input_ = CudaPtr<float>(static_cast<size_t>(kMaxBatchSize) * max_pixels * 3);
  d_batch_output_ = CudaPtr<float>(static_cast<size_t>(kMaxBatchSize) * max_pixels);
  d_batch_bitmap_ = CudaPtr<uint8_t>(static_cast<size_t>(kMaxBatchSize) * max_pixels);

  // Device arrays for batched kernel launch parameters
  d_batch_src_ptrs_ = CudaPtr<void *>(kMaxBatchSize);
  d_batch_src_steps_ = CudaPtr<int>(kMaxBatchSize);
  d_batch_src_heights_ = CudaPtr<int>(kMaxBatchSize);
  d_batch_src_widths_ = CudaPtr<int>(kMaxBatchSize);
  // Pinned host staging for async copy (avoids pageable fallback)
  h_batch_src_ptrs_ = CudaHostPtr<void *>(kMaxBatchSize);
  h_batch_src_steps_ = CudaHostPtr<int>(kMaxBatchSize);
  h_batch_src_heights_ = CudaHostPtr<int>(kMaxBatchSize);
  h_batch_src_widths_ = CudaHostPtr<int>(kMaxBatchSize);

  // Set working pointers to the single-image buffers
  cur_output_ = d_output_.get();
  cur_bitmap_ = d_bitmap_buf_.get();

  // Bind I/O pointers once for single-image path (never change)
  engine_->bind_io(d_input_.get(), d_output_.get());

  // GPU CCL mode: 0=CPU contours, 1=GPU CCL+per-ROI findContours (default),
  // 2=GPU CCL fast (experimental, speed-only — poor accuracy F1~53%)
  if (const char *env = std::getenv("GPU_CCL"))
    gpu_ccl_mode_ = std::atoi(env);

  if (gpu_ccl_mode_ > 0) {
    // Pre-allocate ALL GPU CCL buffers (no per-request alloc)
    d_ccl_labels_ = CudaPtr<int>(max_pixels);
    d_ccl_compact_ids_ = CudaPtr<int>(max_pixels);
    d_ccl_id_counter_ = CudaPtr<int>(1);
    // 2x kMaxGpuComponents: first half for per-component bboxes, second half for filtered output
    d_ccl_bboxes_ = CudaPtr<turbo_ocr::kernels::GpuDetBox>(
        turbo_ocr::kernels::kMaxGpuComponents * 2);
    d_ccl_num_boxes_ = CudaPtr<int>(1);

    // Pinned host memory for result transfer
    h_ccl_boxes_ = CudaHostPtr<turbo_ocr::kernels::GpuDetBox>(
        turbo_ocr::kernels::kMaxGpuComponents);
  }

  return true;
}

// GPU CCL path: connected component labeling on GPU, then extract real contours
// from bitmap within each component's bbox for accurate unclip polygons.
std::vector<Box>
PaddleDet::run_gpu_ccl(int resize_h, int resize_w,
                        int orig_h, int orig_w,
                        cudaStream_t stream) {
  float ratio_h = static_cast<float>(resize_h) / orig_h;
  float ratio_w = static_cast<float>(resize_w) / orig_w;

  int h_num_boxes = 0;
  turbo_ocr::kernels::cuda_gpu_ccl_detect(
      cur_bitmap_, cur_output_, resize_w, resize_h,
      kDetDbBoxThresh,
      d_ccl_labels_.get(), d_ccl_compact_ids_.get(), d_ccl_id_counter_.get(),
      d_ccl_bboxes_.get(), d_ccl_num_boxes_.get(),
      h_ccl_boxes_.get(), &h_num_boxes, stream);

  std::vector<Box> boxes;
  if (h_num_boxes == 0)
    return boxes;

  // Download ONLY the bitmap (not pred_map -- GPU CCL already computed scores).
  // We need the bitmap for per-ROI findContours to get accurate polygon contours.
  cv::Mat bitmap(resize_h, resize_w, CV_8UC1);
  CUDA_CHECK(cudaMemcpyAsync(bitmap.data, cur_bitmap_, resize_w * resize_h,
                              cudaMemcpyDeviceToHost, stream));
  CUDA_CHECK(cudaStreamSynchronize(stream));
  // No pred_map download -- use gb.score from GPU CCL instead of box_score_fast

  boxes.reserve(h_num_boxes);

  // For each GPU-detected component, extract the real contour from the bitmap
  // within the bbox region, then feed to the same unclip pipeline as CPU path.
  for (int i = 0; i < h_num_boxes; i++) {
    const auto &gb = h_ccl_boxes_.get()[i];

    int bw = gb.xmax - gb.xmin + 1;
    int bh = gb.ymax - gb.ymin + 1;
    if (bw < 3 || bh < 3)
      continue;

    // Extract the small bitmap ROI for this component's bbox
    // Pad by 1 pixel to ensure findContours can find closed contours at edges
    int roi_x = std::max(0, gb.xmin - 1);
    int roi_y = std::max(0, gb.ymin - 1);
    int roi_x2 = std::min(resize_w - 1, gb.xmax + 1);
    int roi_y2 = std::min(resize_h - 1, gb.ymax + 1);
    int roi_w = roi_x2 - roi_x + 1;
    int roi_h = roi_y2 - roi_y + 1;

    // Clone the ROI since findContours may modify the source image
    cv::Mat roi = bitmap(cv::Rect(roi_x, roi_y, roi_w, roi_h)).clone();

    // Find contours within this small ROI (~50x20 pixels, negligible cost)
    ccl_roi_contours_buf_.clear();
    cv::findContours(roi, ccl_roi_contours_buf_, cv::RETR_LIST,
                     cv::CHAIN_APPROX_SIMPLE);

    if (ccl_roi_contours_buf_.empty())
      continue;

    // Pick the largest contour in the ROI (should be the component itself)
    const auto &best_contour = (ccl_roi_contours_buf_.size() == 1)
      ? ccl_roi_contours_buf_[0]
      : *std::ranges::max_element(ccl_roi_contours_buf_, {}, [](const std::vector<cv::Point> &c) {
          return cv::contourArea(c);
        });

    if (best_contour.size() <= 2)
      continue;

    // Shift contour from ROI-local coords to global bitmap coords
    ccl_contour_buf_.clear();
    ccl_contour_buf_.reserve(best_contour.size());
    for (const auto &pt : best_contour)
      ccl_contour_buf_.push_back(cv::Point(pt.x + roi_x, pt.y + roi_y));

    // Use GPU CCL score (already filtered by kDetDbBoxThresh in the GPU kernel)
    // Skip box_score_fast — saves downloading pred_map (2.4MB) entirely

    float ssid = 0;
    (void)get_mini_boxes(ccl_contour_buf_, ssid);
    if (ssid < kMinBoxSide)
      continue;

    auto unclipped = unclip(ccl_contour_buf_, kDetDbUnclipRatio);
    if (unclipped.size() < 3)
      continue;

    float ssid2 = 0;
    auto box = get_mini_boxes(unclipped, ssid2);
    if (ssid2 < kMinUnclippedSide)
      continue;

    // Scale back to original image
    for (int k = 0; k < 4; ++k) {
      box[k][0] = std::clamp(static_cast<int>(std::round(box[k][0] / ratio_w)), 0, orig_w - 1);
      box[k][1] = std::clamp(static_cast<int>(std::round(box[k][1] / ratio_h)), 0, orig_h - 1);
    }

    // Filter tiny boxes
    int rw = static_cast<int>(std::sqrt(((box[0][0] - box[1][0]) * (box[0][0] - box[1][0])) +
                                        ((box[0][1] - box[1][1]) * (box[0][1] - box[1][1]))));
    int rh = static_cast<int>(std::sqrt(((box[0][0] - box[3][0]) * (box[0][0] - box[3][0])) +
                                        ((box[0][1] - box[3][1]) * (box[0][1] - box[3][1]))));
    if (rw <= 3 || rh <= 3)
      continue;

    boxes.push_back(box);
  }

  return boxes;
}

// GPU CCL fast path (EXPERIMENTAL, GPU_CCL=2): direct bbox expansion, NO CPU contour processing.
// Computes unclip expansion analytically from bbox dimensions.
// No bitmap download, no findContours, no Clipper — pure arithmetic.
// WARNING: poor accuracy (F1~53%). Use only for speed benchmarks, not production.
std::vector<Box>
PaddleDet::run_gpu_ccl_fast(int resize_h, int resize_w,
                              int orig_h, int orig_w,
                              cudaStream_t stream) {
  float ratio_h = static_cast<float>(resize_h) / orig_h;
  float ratio_w = static_cast<float>(resize_w) / orig_w;

  int h_num_boxes = 0;
  turbo_ocr::kernels::cuda_gpu_ccl_detect(
      cur_bitmap_, cur_output_, resize_w, resize_h,
      kDetDbBoxThresh,
      d_ccl_labels_.get(), d_ccl_compact_ids_.get(), d_ccl_id_counter_.get(),
      d_ccl_bboxes_.get(), d_ccl_num_boxes_.get(),
      h_ccl_boxes_.get(), &h_num_boxes, stream);

  std::vector<Box> boxes;
  if (h_num_boxes == 0)
    return boxes;

  boxes.reserve(h_num_boxes);

  for (int i = 0; i < h_num_boxes; i++) {
    const auto &gb = h_ccl_boxes_.get()[i];

    int bw = gb.xmax - gb.xmin + 1;
    int bh = gb.ymax - gb.ymin + 1;
    if (bw < 3 || bh < 3)
      continue;

    // Analytical unclip: expand bbox by distance = area * unclip_ratio / perimeter
    // For axis-aligned box: area = bw*bh, perimeter = 2*(bw+bh)
    float area = static_cast<float>(bw * bh);
    float perimeter = 2.0f * (bw + bh);
    float expand = area * kDetDbUnclipRatio / perimeter;

    // Expand and clamp to image bounds
    int x0 = std::max(0, static_cast<int>(std::round(gb.xmin - expand)));
    int y0 = std::max(0, static_cast<int>(std::round(gb.ymin - expand)));
    int x1 = std::min(resize_w - 1, static_cast<int>(std::round(gb.xmax + expand)));
    int y1 = std::min(resize_h - 1, static_cast<int>(std::round(gb.ymax + expand)));

    int ew = x1 - x0 + 1;
    int eh = y1 - y0 + 1;
    if (ew < kMinUnclippedSide || eh < kMinUnclippedSide)
      continue;

    // Scale to original image coordinates
    Box box;
    box[0] = {std::clamp(static_cast<int>(std::round(x0 / ratio_w)), 0, orig_w - 1),
              std::clamp(static_cast<int>(std::round(y0 / ratio_h)), 0, orig_h - 1)};
    box[1] = {std::clamp(static_cast<int>(std::round(x1 / ratio_w)), 0, orig_w - 1),
              std::clamp(static_cast<int>(std::round(y0 / ratio_h)), 0, orig_h - 1)};
    box[2] = {std::clamp(static_cast<int>(std::round(x1 / ratio_w)), 0, orig_w - 1),
              std::clamp(static_cast<int>(std::round(y1 / ratio_h)), 0, orig_h - 1)};
    box[3] = {std::clamp(static_cast<int>(std::round(x0 / ratio_w)), 0, orig_w - 1),
              std::clamp(static_cast<int>(std::round(y1 / ratio_h)), 0, orig_h - 1)};

    // Filter tiny boxes
    int rw = box[1][0] - box[0][0];
    int rh = box[3][1] - box[0][1];
    if (rw <= 3 || rh <= 3)
      continue;

    boxes.push_back(box);
  }

  return boxes;
}

// CPU fallback path (original findContours)
std::vector<Box>
PaddleDet::run_cpu_contours(int resize_h, int resize_w,
                             int orig_h, int orig_w,
                             cudaStream_t stream) {
  // Download raw probability map for score filtering
  cv::Mat pred_map(resize_h, resize_w, CV_32F);
  CUDA_CHECK(cudaMemcpyAsync(pred_map.data, cur_output_,
                              resize_h * resize_w * sizeof(float),
                              cudaMemcpyDeviceToHost, stream));

  cv::Mat bitmap(resize_h, resize_w, CV_8UC1);
  CUDA_CHECK(cudaMemcpyAsync(bitmap.data, cur_bitmap_, resize_w * resize_h,
                              cudaMemcpyDeviceToHost, stream));

  CUDA_CHECK(cudaStreamSynchronize(stream));

  return extract_boxes_from_bitmap(
      pred_map, bitmap, orig_h, orig_w, resize_h, resize_w,
      kDetDbBoxThresh, kDetDbUnclipRatio, kMinBoxSide, kMinUnclippedSide,
      shifted_buf_, mask_buf_, contours_buf_, hierarchy_buf_);
}

std::vector<Box>
PaddleDet::run(const GpuImage &gpu_img, int orig_h, int orig_w,
               cudaStream_t stream) {
  int h = orig_h;
  int w = orig_w;
  float ratio = 1.0f;
  if (std::max(h, w) > kMaxSideLen_) {
    ratio = (h > w) ? static_cast<float>(kMaxSideLen_) / h
                    : static_cast<float>(kMaxSideLen_) / w;
  }
  int resize_h = std::max(static_cast<int>(round(h * ratio / 32.0) * 32), 32);
  int resize_w = std::max(static_cast<int>(round(w * ratio / 32.0) * 32), 32);

  // Reset working pointers to single-image buffers (in case batch mode changed them)
  cur_output_ = d_output_.get();
  cur_bitmap_ = d_bitmap_buf_.get();

  // 1+2. Fused resize + normalize + CHW (single kernel, no intermediate buffer)
  turbo_ocr::kernels::cuda_fused_resize_normalize_det(gpu_img, d_input_.get(), resize_w,
                                                resize_h, stream);

  // 3. Inference (dynamic H,W) -- I/O already bound in load_model
  nvinfer1::Dims4 input_dims{1, 3, resize_h, resize_w};
  if (!engine_->infer_dynamic(input_dims, stream)) {
    throw turbo_ocr::InferenceError("Detection TRT inference failed");
  }

  // 4. Threshold on GPU for bitmap
  turbo_ocr::kernels::cuda_threshold_to_u8(cur_output_, cur_bitmap_, resize_w,
                                    resize_h, kDetDbThresh, stream);

  // 5. Choose contour extraction path
  if (gpu_ccl_mode_ == 2) {
    return run_gpu_ccl_fast(resize_h, resize_w, orig_h, orig_w, stream);
  } else if (gpu_ccl_mode_ == 1) {
    return run_gpu_ccl(resize_h, resize_w, orig_h, orig_w, stream);
  } else {
    return run_cpu_contours(resize_h, resize_w, orig_h, orig_w, stream);
  }
}

// ============================================================================
// Batched detection: process N images in a single TRT inference call.
// All images are resized to the same target dimensions (max of the batch).
// ============================================================================
std::vector<std::vector<Box>>
PaddleDet::run_batch(const std::vector<GpuImage> &gpu_imgs,
                     const std::vector<std::pair<int,int>> &orig_dims,
                     cudaStream_t stream) {
  const int n = static_cast<int>(gpu_imgs.size());
  if (n == 0)
    return {};

  // Fallback: single image → use optimized single path
  if (n == 1) {
    auto boxes = run(gpu_imgs[0], orig_dims[0].first, orig_dims[0].second, stream);
    return {std::move(boxes)};
  }

  // Clamp to max batch size
  const int batch_size = std::min(n, kMaxBatchSize);

  // --- Compute unified resize dimensions (max across batch, rounded to 32) ---
  int max_resize_h = 0, max_resize_w = 0;
  struct PerImgInfo {
    int orig_h, orig_w;
    float ratio;
    int resize_h, resize_w;
  };
  std::vector<PerImgInfo> infos(batch_size);

  for (int i = 0; i < batch_size; i++) {
    int h = orig_dims[i].first;
    int w = orig_dims[i].second;
    float ratio = 1.0f;
    if (std::max(h, w) > kMaxSideLen_) {
      ratio = (h > w) ? static_cast<float>(kMaxSideLen_) / h
                      : static_cast<float>(kMaxSideLen_) / w;
    }
    int rh = std::max(static_cast<int>(round(h * ratio / 32.0) * 32), 32);
    int rw = std::max(static_cast<int>(round(w * ratio / 32.0) * 32), 32);
    infos[i] = {h, w, ratio, rh, rw};
    max_resize_h = std::max(max_resize_h, rh);
    max_resize_w = std::max(max_resize_w, rw);
  }

  // Use the unified (max) dimensions for all images in the batch
  const int resize_h = max_resize_h;
  const int resize_w = max_resize_w;
  const int pixels_per_image = resize_h * resize_w;

  // --- 1. Upload per-image metadata to device ---
  // Use pre-allocated pinned buffers for truly async transfers
  for (int i = 0; i < batch_size; i++) {
    h_batch_src_ptrs_.get()[i] = gpu_imgs[i].data;
    h_batch_src_steps_.get()[i] = static_cast<int>(gpu_imgs[i].step);
    h_batch_src_heights_.get()[i] = gpu_imgs[i].rows;
    h_batch_src_widths_.get()[i] = gpu_imgs[i].cols;
  }

  CUDA_CHECK(cudaMemcpyAsync(d_batch_src_ptrs_.get(), h_batch_src_ptrs_.get(),
                              batch_size * sizeof(void *),
                              cudaMemcpyHostToDevice, stream));
  CUDA_CHECK(cudaMemcpyAsync(d_batch_src_steps_.get(), h_batch_src_steps_.get(),
                              batch_size * sizeof(int),
                              cudaMemcpyHostToDevice, stream));
  CUDA_CHECK(cudaMemcpyAsync(d_batch_src_heights_.get(), h_batch_src_heights_.get(),
                              batch_size * sizeof(int),
                              cudaMemcpyHostToDevice, stream));
  CUDA_CHECK(cudaMemcpyAsync(d_batch_src_widths_.get(), h_batch_src_widths_.get(),
                              batch_size * sizeof(int),
                              cudaMemcpyHostToDevice, stream));

  // --- 2. Batched fused resize + normalize + CHW ---
  turbo_ocr::kernels::cuda_batch_fused_resize_normalize_det(
      (const void *const *)d_batch_src_ptrs_.get(), d_batch_src_steps_.get(),
      d_batch_src_heights_.get(), d_batch_src_widths_.get(),
      d_batch_input_.get(), resize_w, resize_h, batch_size, stream);

  // --- 3. Single TRT inference call with batch=N ---
  // Temporarily rebind I/O to batch buffers
  engine_->bind_io(d_batch_input_.get(), d_batch_output_.get());

  nvinfer1::Dims4 input_dims{batch_size, 3, resize_h, resize_w};
  if (!engine_->infer_dynamic(input_dims, stream)) {
    // Restore single-image binding before throwing
    engine_->bind_io(d_input_.get(), d_output_.get());
    throw turbo_ocr::InferenceError("Batched detection TRT inference failed");
  }

  // Restore single-image I/O binding for future single-image calls
  engine_->bind_io(d_input_.get(), d_output_.get());

  // --- 4. Batched threshold (all images at once) ---
  turbo_ocr::kernels::cuda_batch_threshold_to_u8(
      d_batch_output_.get(), d_batch_bitmap_.get(), resize_w, resize_h,
      batch_size, kDetDbThresh, stream);

  // --- 5. Per-image post-processing (GPU CCL fast / CPU contours) ---
  // For each image slice in the batch output, run the appropriate path.
  // We temporarily alias cur_output_ / cur_bitmap_ to point at each image's
  // slice. Scope guard ensures they are restored even if an exception is thrown.
  std::vector<std::vector<Box>> all_boxes(batch_size);

  for (int i = 0; i < batch_size; i++) {
    const int orig_h = infos[i].orig_h;
    const int orig_w = infos[i].orig_w;

    // Alias working pointers to this image's slice.
    auto saved_out = cur_output_;
    auto saved_bmp = cur_bitmap_;
    auto restore = [&, saved_out, saved_bmp]() {
      cur_output_ = saved_out;
      cur_bitmap_ = saved_bmp;
    };
    struct ScopeGuard { decltype(restore) fn; ~ScopeGuard() noexcept { fn(); } } guard{restore};

    cur_output_ = d_batch_output_.get() + static_cast<size_t>(i) * pixels_per_image;
    cur_bitmap_ = d_batch_bitmap_.get() + static_cast<size_t>(i) * pixels_per_image;

    if (gpu_ccl_mode_ == 2) {
      all_boxes[i] = run_gpu_ccl_fast(resize_h, resize_w, orig_h, orig_w, stream);
    } else if (gpu_ccl_mode_ == 1) {
      all_boxes[i] = run_gpu_ccl(resize_h, resize_w, orig_h, orig_w, stream);
    } else {
      all_boxes[i] = run_cpu_contours(resize_h, resize_w, orig_h, orig_w, stream);
    }
  }

  // --- 6. Handle overflow: process remaining images via single-image path ---
  if (n > kMaxBatchSize) {
    all_boxes.resize(n);
    for (int i = kMaxBatchSize; i < n; i++) {
      all_boxes[i] = run(gpu_imgs[i], orig_dims[i].first, orig_dims[i].second, stream);
    }
  }

  return all_boxes;
}

} // namespace turbo_ocr::detection
