#pragma once

#include <memory>
#include <utility>
#include <vector>

#include "turbo_ocr/classification/paddle_cls.h"
#include "turbo_ocr/common/types.h"
#include "turbo_ocr/decode/gpu_image.h"
#include "turbo_ocr/detection/paddle_det.h"
#include "turbo_ocr/pipeline/i_ocr_pipeline.h"
#include "turbo_ocr/recognition/paddle_rec.h"

namespace turbo_ocr::pipeline {

class OcrPipeline : public IOcrPipeline {
public:
  OcrPipeline();
  ~OcrPipeline() noexcept override;

  [[nodiscard]] bool init(const std::string &det_model, const std::string &rec_model,
                          const std::string &rec_dict,
                          const std::string &cls_model = "") override;

  // IOcrPipeline interface — delegates to stream-aware overloads with stream=0
  void warmup() override { warmup_gpu(0); }
  [[nodiscard]] std::vector<OCRResultItem> run(const cv::Mat &img) override { return run(img, 0); }

  // GPU-specific: Pre-allocate all buffers and run a dummy inference to warm up
  // TensorRT engines and lazy GPU allocations.
  void warmup_gpu(cudaStream_t stream);

  [[nodiscard]] std::vector<OCRResultItem> run(const cv::Mat &img, cudaStream_t stream);

  // Run OCR on an image already in GPU memory (skips CPU→GPU upload).
  // The GpuImage must contain BGR8 data with the given pitch/step.
  // The caller retains ownership of the GPU buffer.
  [[nodiscard]] std::vector<OCRResultItem> run(GpuImage gpu_img, cudaStream_t stream = 0);

  // Ensure the GPU upload buffer can hold an image of the given size.
  // Returns {d_img_buf_, d_img_pitch_} after grow-only reallocation.
  // Useful for callers that want to decode directly into the pipeline's GPU buffer.
  [[nodiscard]] std::pair<void *, size_t> ensure_gpu_buf(int rows, int cols);

  // Batch: process multiple images, return results per image
  [[nodiscard]] std::vector<std::vector<OCRResultItem>>
  run_batch(const std::vector<cv::Mat> &imgs, cudaStream_t stream = 0);

private:
  std::unique_ptr<detection::PaddleDet> det_;
  std::unique_ptr<classification::PaddleCls> cls_;
  std::unique_ptr<recognition::PaddleRec> rec_;

  bool use_cls_ = false;

  // Dedicated stream for recognition — allows det on the caller's stream to
  // overlap with rec on this stream across consecutive requests.
  cudaStream_t rec_stream_ = nullptr;

  // Event recorded on rec_stream_ after recognition GPU work is launched.
  // The next run() waits on this before reusing the image buffer that rec
  // might still be reading.  This enables det/rec overlap across calls:
  //   Call N:    [...det_N on stream...] → launch rec_N on rec_stream_ (async)
  //   Call N+1:  wait rec_event_ → [...det_{N+1} overlaps with tail of rec_N...]
  cudaEvent_t  rec_event_  = nullptr;

  // Event recorded on the caller's stream after det+cls finish.
  // rec_stream_ waits on this before launching recognition, ensuring proper
  // data dependency without a full stream synchronize.
  cudaEvent_t  det_event_  = nullptr;

  // Double-buffered GPU upload buffers (grow-only).
  // Alternating between two buffers lets recognition read the previous image
  // on rec_stream_ while the current image is uploaded + detected on the
  // caller's stream without a data race.
  int cur_img_buf_ = 0; // toggles 0/1 each run() call
  struct ImgBuf {
    void *d_buf = nullptr;
    size_t pitch = 0;
    int cap_rows = 0;
    int cap_cols = 0;
  };
  ImgBuf img_bufs_[2];

  // Pinned host staging buffer for truly async CPU->GPU uploads
  void *h_pinned_buf_ = nullptr;
  size_t h_pinned_size_ = 0;

  // Reusable buffers for selective angle classification
  std::vector<int> vertical_box_indices_;
  std::vector<Box> vertical_boxes_buf_;

  // Pre-allocated batch image GPU buffers (avoid cudaMalloc per batch image)
  static constexpr int kMaxBatchImages = 8;
  struct BatchImgBuf {
    void *d_buf = nullptr;
    size_t pitch = 0;
    int cap_rows = 0, cap_cols = 0;
  };
  BatchImgBuf batch_img_bufs_[kMaxBatchImages];
};

} // namespace turbo_ocr::pipeline
