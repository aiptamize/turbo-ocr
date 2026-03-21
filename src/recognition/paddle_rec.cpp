#include "turbo_ocr/recognition/paddle_rec.h"
#include "turbo_ocr/kernels/kernels.h"
#include "turbo_ocr/recognition/ctc_decode.h"

#include "turbo_ocr/common/errors.h"
#include "turbo_ocr/common/perspective.h"

#include <algorithm>
#include <cmath>
#include <format>
#include <ranges>

using namespace turbo_ocr::recognition;
using turbo_ocr::engine::TrtEngine;
using turbo_ocr::Box;
using turbo_ocr::GpuImage;

PaddleRec::PaddleRec() { label_list_.push_back("blank"); }

PaddleRec::~PaddleRec() noexcept {
  // CudaPtr/CudaHostPtr members (including output_slots_) are cleaned up by RAII.
}

bool PaddleRec::load_model(const std::string &model_path) {
  engine_ = std::make_unique<TrtEngine>(model_path);
  if (!engine_->load())
    return false;
  return probe_and_init();
}

bool PaddleRec::probe_and_init() {
  nvinfer1::Dims opt_dims;
  opt_dims.nbDims = 4;
  opt_dims.d[0] = rec_batch_num_;
  opt_dims.d[1] = 3;
  opt_dims.d[2] = rec_image_h_;
  opt_dims.d[3] = kMaxRecWidth;

  engine_->probe_output_dims(opt_dims, actual_seq_len_, actual_num_classes_);
  std::cout << std::format("[PaddleRec] Output dims: seq_len={} num_classes={}",
                           actual_seq_len_, actual_num_classes_) << '\n';
  return true;
}

bool PaddleRec::load_dict(const std::string &dict_path) {
  return load_label_dict(dict_path, label_list_);
}

void PaddleRec::allocate_buffers() {
  if (buffers_allocated_)
    return;

  int bs = rec_batch_num_;

  size_t input_elems = static_cast<size_t>(bs) * 3 * rec_image_h_ * kMaxRecWidth;
  d_batch_input_ = CudaPtr<float>(input_elems);

  size_t output_elems = static_cast<size_t>(bs) * actual_seq_len_ * actual_num_classes_;
  d_output_ = CudaPtr<float>(output_elems);

  size_t seq_elems = static_cast<size_t>(bs) * actual_seq_len_;

  // Multi-slot buffers: each slot gets its own indices/scores (GPU + host)
  // AND its own transform pinned buffers to avoid DMA race conditions.
  for (int s = 0; s < kMaxSlots; ++s) {
    output_slots_[s].d_indices = CudaPtr<int>(seq_elems);
    output_slots_[s].d_scores = CudaPtr<float>(seq_elems);
    output_slots_[s].h_indices = CudaHostPtr<int>(seq_elems);
    output_slots_[s].h_scores = CudaHostPtr<float>(seq_elems);
    output_slots_[s].h_M_invs = CudaHostPtr<float>(bs * 9);
    output_slots_[s].h_crop_widths = CudaHostPtr<int>(bs);
  }

  d_M_invs_ = CudaPtr<float>(bs * 9);
  d_crop_widths_ = CudaPtr<int>(bs);

  // Bind I/O once (pointers never change after allocation)
  engine_->bind_io(d_batch_input_.get(), d_output_.get());

  buffers_allocated_ = true;
}

std::vector<std::pair<std::string, float>>
PaddleRec::run(const GpuImage &img, const std::vector<Box> &boxes,
               cudaStream_t stream) {
  std::vector<std::pair<std::string, float>> results;
  if (boxes.empty()) [[unlikely]]
    return results;

  allocate_buffers();

  int total_boxes = static_cast<int>(boxes.size());
  results.resize(total_boxes);

  // Reuse pre-allocated buffer (avoid per-request heap alloc)
  crops_buf_.resize(total_boxes);
  auto &crops = crops_buf_;

  for (int i = 0; i < total_boxes; i++) {
    const auto &box = boxes[i];
    float w = std::sqrt(((box[0][0] - box[1][0]) * (box[0][0] - box[1][0])) +
                        ((box[0][1] - box[1][1]) * (box[0][1] - box[1][1])));
    float h = std::sqrt(((box[0][0] - box[3][0]) * (box[0][0] - box[3][0])) +
                        ((box[0][1] - box[3][1]) * (box[0][1] - box[3][1])));
    float ar = (h > 0) ? (w / h) : 0;

    // Snap the INDIVIDUAL crop width to a bucket
    int crop_imgW = std::min(static_cast<int>(std::ceil(rec_image_h_ * ar)), kMaxRecWidth);
    crop_imgW = std::max(crop_imgW, 32); // minimum 32px, NOT forced to 320
    int bucket = *std::lower_bound(kWidthBuckets.begin(), kWidthBuckets.end(), crop_imgW);

    crops[i] = {i, bucket};
  }

  // Sort by bucket so crops with similar widths batch together
  std::ranges::sort(crops, {}, &CropInfo::bucket_w);

  // ---- Multi-slot deferred-sync recognition --------------------------------
  // Queue ALL batch iterations to the GPU without inter-batch synchronization.
  // Each iteration writes argmax results to its own output slot (d_indices/d_scores).
  // After all iterations are queued, a single cudaStreamSynchronize retrieves
  // all results, then CTC decode runs on CPU for all iterations at once.
  // This eliminates ~N-1 cudaEventSynchronize calls that previously created
  // GPU idle gaps (the CPU couldn't submit batch N+1 until batch N was done).

  struct BatchRecord {
    int beg, end, seq_len, slot;
  };
  std::vector<BatchRecord> batch_records;
  batch_records.reserve(16);

  int beg = 0;
  int slot = 0;

  while (beg < total_boxes) {
    int bucket_w = crops[beg].bucket_w;

    // Find end of this bucket group (or batch limit)
    int end = beg;
    while (end < total_boxes && end - beg < rec_batch_num_ && crops[end].bucket_w == bucket_w)
      end++;

    int cur_batch = end - beg;
    int imgW = bucket_w;

    // If we've exhausted our output slots, sync and decode what we have so far
    if (slot >= kMaxSlots) {
      CUDA_CHECK(cudaStreamSynchronize(stream));
      for (auto &rec : batch_records) {
        int batch_n = rec.end - rec.beg;
        auto &os = output_slots_[rec.slot];
        for (int j = 0; j < batch_n; ++j) {
          int orig_idx = crops[rec.beg + j].orig_idx;
          results[orig_idx] =
              ctc_greedy_decode(os.h_indices.get() + j * rec.seq_len,
                               os.h_scores.get() + j * rec.seq_len, rec.seq_len, label_list_);
        }
      }
      batch_records.clear();
      slot = 0;
    }

    // Build transforms using per-slot pinned host buffers (avoids DMA race)
    auto &os = output_slots_[slot];
    for (int j = 0; j < cur_batch; ++j) {
      int orig_idx = crops[beg + j].orig_idx;
      auto ct = turbo_ocr::compute_crop_transform(boxes[orig_idx], rec_image_h_, imgW);
      os.h_crop_widths.get()[j] = ct.crop_width;
      std::copy_n(ct.M_inv, 9, os.h_M_invs.get() + j * 9);
    }

    // Upload + warp + infer (all async on stream)
    CUDA_CHECK(cudaMemcpyAsync(d_M_invs_.get(), os.h_M_invs.get(), cur_batch * 9 * sizeof(float),
                                cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_crop_widths_.get(), os.h_crop_widths.get(), cur_batch * sizeof(int),
                                cudaMemcpyHostToDevice, stream));

    turbo_ocr::kernels::cuda_batch_roi_warp(img, d_M_invs_.get(), d_crop_widths_.get(),
                                     d_batch_input_.get(), cur_batch, rec_image_h_,
                                     imgW, stream);

    nvinfer1::Dims input_dims;
    input_dims.nbDims = 4;
    input_dims.d[0] = cur_batch;
    input_dims.d[1] = 3;
    input_dims.d[2] = rec_image_h_;
    input_dims.d[3] = imgW;

    if (!engine_->infer_dynamic(input_dims, stream)) {
      throw turbo_ocr::InferenceError("Recognition TRT inference failed");
    }

    nvinfer1::Dims out_dims = engine_->get_output_dims();
    int seq_len = out_dims.d[1];
    int num_classes = out_dims.d[2];

    if (seq_len > actual_seq_len_ || num_classes > actual_num_classes_) {
      std::cerr << std::format("[PaddleRec] WARNING: output dims (seq_len={}, num_classes={}) "
                               "exceed buffer (seq_len={}, num_classes={}), skipping batch\n",
                               seq_len, num_classes, actual_seq_len_, actual_num_classes_);
      beg = end;
      continue;
    }

    turbo_ocr::kernels::cuda_argmax(d_output_.get(), os.d_indices.get(), os.d_scores.get(), cur_batch,
                             seq_len, num_classes, stream);

    // Async D2H copy to this slot's host buffers (no sync needed -- each slot is independent)
    int dl_count = cur_batch * seq_len;
    CUDA_CHECK(cudaMemcpyAsync(os.h_indices.get(), os.d_indices.get(), dl_count * sizeof(int),
                                cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(os.h_scores.get(), os.d_scores.get(), dl_count * sizeof(float),
                                cudaMemcpyDeviceToHost, stream));

    batch_records.push_back({beg, end, seq_len, slot});
    slot++;
    beg = end;
  }

  // Single sync for ALL queued batches
  CUDA_CHECK(cudaStreamSynchronize(stream));

  // CTC decode ALL batches on CPU (all D2H transfers are complete)
  for (auto &rec : batch_records) {
    int batch_n = rec.end - rec.beg;
    auto &os = output_slots_[rec.slot];
    for (int j = 0; j < batch_n; ++j) {
      int orig_idx = crops[rec.beg + j].orig_idx;
      results[orig_idx] =
          ctc_greedy_decode(os.h_indices.get() + j * rec.seq_len,
                           os.h_scores.get() + j * rec.seq_len, rec.seq_len, label_list_);
    }
  }

  return results;
}

std::vector<std::vector<std::pair<std::string, float>>>
PaddleRec::run_multi(const std::vector<ImageCrops> &image_crops,
                     cudaStream_t stream) {
  int num_images = static_cast<int>(image_crops.size());
  std::vector<std::vector<std::pair<std::string, float>>> all_results(num_images);

  // Count total boxes and early-out if none
  int total_boxes = 0;
  for (int i = 0; i < num_images; i++) {
    all_results[i].resize(image_crops[i].boxes.size());
    total_boxes += static_cast<int>(image_crops[i].boxes.size());
  }
  if (total_boxes == 0)
    return all_results;

  allocate_buffers();

  // Flatten all crops with (img_idx, box_idx) tracking
  struct MultiCropInfo {
    int img_idx;
    int box_idx;
    int bucket_w;
  };
  std::vector<MultiCropInfo> crops;
  crops.reserve(total_boxes);

  for (int i = 0; i < num_images; i++) {
    const auto &boxes = image_crops[i].boxes;
    for (int b = 0; b < static_cast<int>(boxes.size()); b++) {
      const auto &box = boxes[b];
      float w = std::sqrt(((box[0][0] - box[1][0]) * (box[0][0] - box[1][0])) +
                          ((box[0][1] - box[1][1]) * (box[0][1] - box[1][1])));
      float h = std::sqrt(((box[0][0] - box[3][0]) * (box[0][0] - box[3][0])) +
                          ((box[0][1] - box[3][1]) * (box[0][1] - box[3][1])));
      float ar = (h > 0) ? (w / h) : 0;

      int crop_imgW = std::min(static_cast<int>(std::ceil(rec_image_h_ * ar)), kMaxRecWidth);
      crop_imgW = std::max(crop_imgW, 32);
      int bucket = *std::lower_bound(kWidthBuckets.begin(),
                                     kWidthBuckets.end(), crop_imgW);
      crops.push_back({i, b, bucket});
    }
  }

  // Sort by (bucket, img_idx) so crops with same width batch together,
  // and within a bucket, crops from the same image are contiguous
  // (minimizes per-image warp kernel calls).
  std::ranges::sort(crops, [](const MultiCropInfo &a, const MultiCropInfo &b) {
    if (a.bucket_w != b.bucket_w) return a.bucket_w < b.bucket_w;
    return a.img_idx < b.img_idx;
  });

  // Multi-slot deferred-sync recognition (same as single-image run())
  struct MultiBatchRecord {
    int beg, end, seq_len, slot;
  };
  std::vector<MultiBatchRecord> batch_records;
  batch_records.reserve(16);

  int beg = 0;
  int slot = 0;

  while (beg < total_boxes) {
    int bucket_w = crops[beg].bucket_w;

    int end = beg;
    while (end < total_boxes && end - beg < rec_batch_num_ &&
           crops[end].bucket_w == bucket_w)
      end++;

    int cur_batch = end - beg;
    int imgW = bucket_w;

    // If we've exhausted output slots, sync and decode what we have so far
    if (slot >= kMaxSlots) {
      CUDA_CHECK(cudaStreamSynchronize(stream));
      for (auto &rec : batch_records) {
        int batch_n = rec.end - rec.beg;
        auto &os = output_slots_[rec.slot];
        for (int j = 0; j < batch_n; ++j) {
          const auto &ci = crops[rec.beg + j];
          all_results[ci.img_idx][ci.box_idx] =
              ctc_greedy_decode(os.h_indices.get() + j * rec.seq_len,
                               os.h_scores.get() + j * rec.seq_len, rec.seq_len, label_list_);
        }
      }
      batch_records.clear();
      slot = 0;
    }

    // Build transforms using per-slot pinned buffers (avoids DMA race)
    auto &os = output_slots_[slot];
    for (int j = 0; j < cur_batch; ++j) {
      const auto &ci = crops[beg + j];
      const auto &box = image_crops[ci.img_idx].boxes[ci.box_idx];
      auto ct = turbo_ocr::compute_crop_transform(box, rec_image_h_, imgW);
      os.h_crop_widths.get()[j] = ct.crop_width;
      std::copy_n(ct.M_inv, 9, os.h_M_invs.get() + j * 9);
    }

    CUDA_CHECK(cudaMemcpyAsync(d_M_invs_.get(), os.h_M_invs.get(),
                                cur_batch * 9 * sizeof(float),
                                cudaMemcpyHostToDevice, stream));
    CUDA_CHECK(cudaMemcpyAsync(d_crop_widths_.get(), os.h_crop_widths.get(),
                                cur_batch * sizeof(int),
                                cudaMemcpyHostToDevice, stream));

    // Warp crops per source image
    {
      size_t slot_stride = static_cast<size_t>(3) * rec_image_h_ * imgW;
      int j = 0;
      while (j < cur_batch) {
        int src_img = crops[beg + j].img_idx;
        int run_start = j;
        while (j < cur_batch && crops[beg + j].img_idx == src_img)
          j++;
        int run_len = j - run_start;

        turbo_ocr::kernels::cuda_batch_roi_warp(
            image_crops[src_img].img,
            d_M_invs_.get() + run_start * 9,
            d_crop_widths_.get() + run_start,
            d_batch_input_.get() + run_start * slot_stride,
            run_len, rec_image_h_, imgW, stream);
      }
    }

    nvinfer1::Dims input_dims;
    input_dims.nbDims = 4;
    input_dims.d[0] = cur_batch;
    input_dims.d[1] = 3;
    input_dims.d[2] = rec_image_h_;
    input_dims.d[3] = imgW;

    if (!engine_->infer_dynamic(input_dims, stream)) {
      throw turbo_ocr::InferenceError("Recognition TRT inference failed");
    }

    nvinfer1::Dims out_dims = engine_->get_output_dims();
    int seq_len = out_dims.d[1];
    int num_classes = out_dims.d[2];

    if (seq_len > actual_seq_len_ || num_classes > actual_num_classes_) {
      std::cerr << std::format("[PaddleRec] WARNING: output dims (seq_len={}, num_classes={}) "
                               "exceed buffer (seq_len={}, num_classes={}), skipping batch\n",
                               seq_len, num_classes, actual_seq_len_, actual_num_classes_);
      beg = end;
      continue;
    }

    turbo_ocr::kernels::cuda_argmax(d_output_.get(), os.d_indices.get(), os.d_scores.get(), cur_batch,
                             seq_len, num_classes, stream);

    int dl_count = cur_batch * seq_len;
    CUDA_CHECK(cudaMemcpyAsync(os.h_indices.get(), os.d_indices.get(), dl_count * sizeof(int),
                                cudaMemcpyDeviceToHost, stream));
    CUDA_CHECK(cudaMemcpyAsync(os.h_scores.get(), os.d_scores.get(), dl_count * sizeof(float),
                                cudaMemcpyDeviceToHost, stream));

    batch_records.push_back({beg, end, seq_len, slot});
    slot++;
    beg = end;
  }

  // Single sync for all queued batches
  CUDA_CHECK(cudaStreamSynchronize(stream));

  // CTC decode all batches
  for (auto &rec : batch_records) {
    int batch_n = rec.end - rec.beg;
    auto &mos = output_slots_[rec.slot];
    for (int j = 0; j < batch_n; ++j) {
      const auto &ci = crops[rec.beg + j];
      all_results[ci.img_idx][ci.box_idx] =
          ctc_greedy_decode(mos.h_indices.get() + j * rec.seq_len,
                           mos.h_scores.get() + j * rec.seq_len, rec.seq_len, label_list_);
    }
  }

  return all_results;
}

