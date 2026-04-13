#include "turbo_ocr/layout/paddle_layout.h"

#include <algorithm>
#include <cstring>
#include <iostream>

#include <cuda_runtime.h>

#include "turbo_ocr/common/cuda_check.h"
#include "turbo_ocr/kernels/kernels.h"

namespace turbo_ocr::layout {

PaddleLayout::~PaddleLayout() noexcept {
  if (d2h_event_)
    cudaEventDestroy(d2h_event_);
}

bool PaddleLayout::discover_tensor_names() {
  // PP-DocLayoutV3 has:
  //   inputs : image (4-D), im_shape (2-D), scale_factor (2-D)
  //   outputs: fetch_name_0 (N, 7), fetch_name_1 (B,), fetch_name_2 (N, 200, 200)
  // paddle2onnx does not guarantee input order, so dispatch by name and rank.
  for (const auto &n : engine_->input_names()) {
    if (n == "image" || n.find("image") != std::string::npos) {
      if (name_image_.empty()) name_image_ = n;
    } else if (n == "im_shape") {
      name_im_shape_ = n;
    } else if (n == "scale_factor") {
      name_scale_factor_ = n;
    }
  }
  if (name_image_.empty() || name_im_shape_.empty() || name_scale_factor_.empty()) {
    std::cerr << "[layout] expected inputs image/im_shape/scale_factor, got: ";
    for (const auto &n : engine_->input_names()) std::cerr << n << " ";
    std::cerr << '\n';
    return false;
  }

  const auto &outs = engine_->output_names();
  if (outs.size() != 3) {
    std::cerr << "[layout] expected 3 outputs, got " << outs.size() << '\n';
    return false;
  }
  // Identify by rank: the (N, 7) detection tensor has nbDims == 2 and last
  // extent 7; the count tensor has nbDims == 1; the mask tensor has nbDims
  // == 3. We query shapes after setting a concrete input shape, so do this
  // once from the engine binding metadata (TRT stores the profile's max
  // dims with the -1 dynamic placeholders stripped to the kMAX profile).
  for (const auto &name : outs) {
    auto dims = engine_->tensor_shape(name);
    if (dims.nbDims == 3 && dims.d[1] == 200 && dims.d[2] == 200) {
      name_out2_ = name;
    } else if (dims.nbDims == 1) {
      name_out1_ = name;
    } else {
      // (-1, 7) — the detection tensor
      name_out0_ = name;
    }
  }
  if (name_out0_.empty() || name_out1_.empty() || name_out2_.empty()) {
    std::cerr << "[layout] could not classify outputs by shape; outs=";
    for (const auto &n : outs) std::cerr << n << " ";
    std::cerr << '\n';
    return false;
  }
  return true;
}

bool PaddleLayout::init_buffers() {
  // Image: [1, 3, 800, 800] float
  d_image_.reset(static_cast<size_t>(1) * 3 * kInputSize * kInputSize);
  d_im_shape_.reset(2);
  d_scale_factor_.reset(2);

  // Detection output (N, 7). We reserve kMaxDetections rows.
  d_out0_.reset(static_cast<size_t>(kMaxDetections) * 7);
  d_out1_.reset(1);
  // Mask tensor we don't read: (kMaxDetections, 200, 200) int32 ≈ 48 MB. We
  // still have to own the buffer so TRT has a valid address to write to.
  d_out2_.reset(static_cast<size_t>(kMaxDetections) * 200 * 200);

  h_out0_.reset(static_cast<size_t>(kMaxDetections) * 7);
  h_out1_.reset(1);
  h_im_shape_.reset(2);
  h_scale_factor_.reset(2);

  // Bind tensor addresses once. We re-bind after select_profile, but
  // PaddleLayout only uses profile 0 so once is enough.
  engine_->set_tensor_address(name_image_,        d_image_.get());
  engine_->set_tensor_address(name_im_shape_,     d_im_shape_.get());
  engine_->set_tensor_address(name_scale_factor_, d_scale_factor_.get());
  engine_->set_tensor_address(name_out0_,         d_out0_.get());
  engine_->set_tensor_address(name_out1_,         d_out1_.get());
  engine_->set_tensor_address(name_out2_,         d_out2_.get());

  // Event used to notify collect() that the D2H readback is complete.
  // Timing is disabled — we only care about the ordering guarantee.
  CUDA_CHECK(cudaEventCreateWithFlags(&d2h_event_, cudaEventDisableTiming));
  return true;
}

bool PaddleLayout::load_model(const std::string &trt_path) {
  engine_ = std::make_unique<engine::TrtEngine>(trt_path);
  if (!engine_->load()) {
    std::cerr << "[layout] failed to load TRT engine: " << trt_path << '\n';
    return false;
  }
  if (!discover_tensor_names()) return false;
  if (!init_buffers()) return false;
  return true;
}

bool PaddleLayout::enqueue(const GpuImage &gpu_img, int orig_h, int orig_w,
                           cudaStream_t stream) {
  pending_orig_h_ = orig_h;
  pending_orig_w_ = orig_w;
  if (!engine_) return false;

  // 1. Preprocess: fused resize-to-800x800 + pixel/255 on GPU. Writes
  //    directly into d_image_ (CHW float, batch=1).
  kernels::cuda_fused_resize_normalize_layout(
      gpu_img, d_image_.get(), kInputSize, kInputSize, stream);

  // 2. Fill im_shape and scale_factor, then async H2D.
  //    PaddleX convention: im_shape = [resized_h, resized_w] (always 800x800),
  //    scale_factor = [h_scale, w_scale] = [800/orig_h, 800/orig_w].
  h_im_shape_.get()[0] = static_cast<float>(kInputSize);
  h_im_shape_.get()[1] = static_cast<float>(kInputSize);
  h_scale_factor_.get()[0] =
      static_cast<float>(kInputSize) / static_cast<float>(orig_h);
  h_scale_factor_.get()[1] =
      static_cast<float>(kInputSize) / static_cast<float>(orig_w);
  CUDA_CHECK(cudaMemcpyAsync(d_im_shape_.get(), h_im_shape_.get(),
                              sizeof(float) * 2, cudaMemcpyHostToDevice, stream));
  CUDA_CHECK(cudaMemcpyAsync(d_scale_factor_.get(), h_scale_factor_.get(),
                              sizeof(float) * 2, cudaMemcpyHostToDevice, stream));

  // 3. Set per-input shapes (batch=1). Idempotent in TRT, so cheap.
  nvinfer1::Dims4 img_dims{1, 3, kInputSize, kInputSize};
  nvinfer1::Dims2 vec_dims{1, 2};
  if (!engine_->set_input_shape(name_image_, img_dims))        return false;
  if (!engine_->set_input_shape(name_im_shape_, vec_dims))     return false;
  if (!engine_->set_input_shape(name_scale_factor_, vec_dims)) return false;

  // 4. Execute (async). The stream continues without blocking.
  if (!engine_->execute(stream)) {
    std::cerr << "[layout] TRT execute failed\n";
    return false;
  }

  // 5. Record event so collect() knows when the execute has finished.
  //    Shape query + D2H are deferred to collect() to keep enqueue() fully
  //    async — avoids the implicit GPU sync that getTensorShape() causes
  //    on DETR models with data-dependent output shapes.
  pending_stream_ = stream;
  CUDA_CHECK(cudaEventRecord(d2h_event_, stream));
  return true;
}

std::vector<LayoutBox> PaddleLayout::collect(float score_threshold) {
  std::vector<LayoutBox> out;
  if (!engine_ || !d2h_event_) return out;

  // Wait for the TRT execute to finish on layout_stream_.
  CUDA_CHECK(cudaEventSynchronize(d2h_event_));

  // Query the output shape now that execution is complete. For DETR models
  // this is data-dependent — calling it inside enqueue() would have
  // implicitly synced the GPU, stalling the worker thread.
  auto out_dims = engine_->tensor_shape(name_out0_);
  int n_rows = (out_dims.nbDims >= 2 ? out_dims.d[0] : 0);
  if (n_rows <= 0) return out;
  n_rows = std::min(n_rows, kMaxDetections);

  // D2H the detection tensor (8 KB for 300 rows × 7 × 4 B). The GPU is
  // idle on this stream so the copy completes immediately.
  CUDA_CHECK(cudaMemcpyAsync(
      h_out0_.get(), d_out0_.get(),
      sizeof(float) * static_cast<size_t>(n_rows) * 7,
      cudaMemcpyDeviceToHost, pending_stream_));
  CUDA_CHECK(cudaStreamSynchronize(pending_stream_));

  const int orig_h  = pending_orig_h_;
  const int orig_w  = pending_orig_w_;

  if (n_rows <= 0) return out;

  // Decode rows: [class_id, score, xmin, ymin, xmax, ymax, read_order].
  // With correct im_shape/scale_factor (PaddleX convention), the model's
  // postprocessor outputs coordinates directly in the original image space.
  out.reserve(static_cast<size_t>(n_rows));
  const float *rows = h_out0_.get();
  for (int i = 0; i < n_rows; ++i) {
    const float *row = rows + i * 7;
    float score = row[1];
    if (score < score_threshold) continue;

    int cls = static_cast<int>(row[0]);
    if (cls < 0 || cls >= static_cast<int>(kLayoutLabels.size())) continue;

    int x0 = std::clamp(static_cast<int>(row[2]), 0, orig_w - 1);
    int y0 = std::clamp(static_cast<int>(row[3]), 0, orig_h - 1);
    int x1 = std::clamp(static_cast<int>(row[4]), 0, orig_w - 1);
    int y1 = std::clamp(static_cast<int>(row[5]), 0, orig_h - 1);
    if (x1 <= x0 || y1 <= y0) continue;

    LayoutBox lb;
    lb.class_id = cls;
    lb.score = score;
    lb.box[0] = {x0, y0};
    lb.box[1] = {x1, y0};
    lb.box[2] = {x1, y1};
    lb.box[3] = {x0, y1};
    out.push_back(lb);
  }

  // Layout NMS (matches PaddleX layout_nms=True):
  //   same-class IoU > 0.6  → suppress the lower-scoring box
  //   cross-class IoU > 0.98 → suppress the lower-scoring box
  // Boxes are already sorted by descending score (model output order).
  std::sort(out.begin(), out.end(),
            [](const LayoutBox &a, const LayoutBox &b) {
              return a.score > b.score;
            });

  auto box_coords = [](const LayoutBox &lb) {
    return std::tuple{lb.box[0][0], lb.box[0][1], lb.box[2][0], lb.box[2][1]};
  };

  auto compute_iou = [&](const LayoutBox &a, const LayoutBox &b) -> float {
    auto [ax0, ay0, ax1, ay1] = box_coords(a);
    auto [bx0, by0, bx1, by1] = box_coords(b);
    int ix0 = std::max(ax0, bx0), iy0 = std::max(ay0, by0);
    int ix1 = std::min(ax1, bx1), iy1 = std::min(ay1, by1);
    float inter = std::max(0, ix1 - ix0) * std::max(0, iy1 - iy0);
    float area_a = (ax1 - ax0) * (ay1 - ay0);
    float area_b = (bx1 - bx0) * (by1 - by0);
    float union_area = area_a + area_b - inter;
    return union_area > 0 ? inter / union_area : 0.0f;
  };

  // Containment: if >90% of box b's area is inside box a, b is contained.
  auto is_contained = [&](const LayoutBox &a, const LayoutBox &b) -> bool {
    auto [ax0, ay0, ax1, ay1] = box_coords(a);
    auto [bx0, by0, bx1, by1] = box_coords(b);
    int ix0 = std::max(ax0, bx0), iy0 = std::max(ay0, by0);
    int ix1 = std::min(ax1, bx1), iy1 = std::min(ay1, by1);
    float inter = std::max(0, ix1 - ix0) * std::max(0, iy1 - iy0);
    float area_b = (bx1 - bx0) * (by1 - by0);
    return area_b > 0 && (inter / area_b) >= 0.8f;
  };

  constexpr float kIoUSame = 0.6f;
  constexpr float kIoUDiff = 0.98f;

  std::vector<LayoutBox> nms_out;
  nms_out.reserve(out.size());
  std::vector<bool> suppressed(out.size(), false);
  for (size_t i = 0; i < out.size(); ++i) {
    if (suppressed[i]) continue;
    nms_out.push_back(out[i]);
    for (size_t j = i + 1; j < out.size(); ++j) {
      if (suppressed[j]) continue;
      float iou = compute_iou(out[i], out[j]);
      float thresh = (out[i].class_id == out[j].class_id) ? kIoUSame : kIoUDiff;
      if (iou >= thresh) { suppressed[j] = true; continue; }
      // Same-class containment: if j is ≥90% inside i, suppress j.
      if (out[i].class_id == out[j].class_id && is_contained(out[i], out[j]))
        suppressed[j] = true;
    }
  }

  // Post-NMS containment cleanup: if box A is ≥90% inside box B (any class),
  // drop A — it's a duplicate subset. Handles both same-class (e.g. nested
  // tables) and cross-class (e.g. text lines inside a table region).
  {
    std::vector<bool> drop(nms_out.size(), false);
    for (size_t i = 0; i < nms_out.size(); ++i) {
      if (drop[i]) continue;
      for (size_t j = i + 1; j < nms_out.size(); ++j) {
        if (drop[j]) continue;
        if (is_contained(nms_out[j], nms_out[i])) {
          drop[i] = true; break;
        }
        if (is_contained(nms_out[i], nms_out[j])) {
          drop[j] = true;
        }
      }
    }
    std::vector<LayoutBox> cleaned;
    cleaned.reserve(nms_out.size());
    for (size_t i = 0; i < nms_out.size(); ++i)
      if (!drop[i]) cleaned.push_back(std::move(nms_out[i]));
    nms_out = std::move(cleaned);
  }

  // Filter "image" detections that cover >82% (portrait) or >93% (landscape) of the page.
  const float img_area = static_cast<float>(orig_w) * orig_h;
  const float area_thresh = (orig_h > orig_w) ? 0.82f : 0.93f;
  constexpr int kImageClassId = 14; // "image" in kLayoutLabels
  std::erase_if(nms_out, [&](const LayoutBox &lb) {
    if (lb.class_id != kImageClassId) return false;
    float box_area = static_cast<float>(lb.box[2][0] - lb.box[0][0]) *
                     (lb.box[2][1] - lb.box[0][1]);
    return box_area > area_thresh * img_area;
  });

  return nms_out;
}

} // namespace turbo_ocr::layout
