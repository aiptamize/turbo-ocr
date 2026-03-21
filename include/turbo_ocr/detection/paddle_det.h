#pragma once

#include <memory>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "turbo_ocr/decode/gpu_image.h"
#include "turbo_ocr/engine/trt_engine.h"
#include "turbo_ocr/common/box.h"
#include "turbo_ocr/common/cuda_check.h"
#include "turbo_ocr/common/cuda_ptr.h"
#include "turbo_ocr/kernels/kernels.h"

namespace turbo_ocr::detection {

/// GPU text detector using TensorRT (DB post-processing).
class PaddleDet {
public:
  PaddleDet() = default;
  ~PaddleDet() noexcept = default; // RAII handles all GPU cleanup

  /// Load a TensorRT detection engine and allocate GPU buffers.
  [[nodiscard]] bool load_model(const std::string &model_path);

  // Takes GpuImage directly - no double upload
  [[nodiscard]] std::vector<Box> run(const GpuImage &gpu_img, int orig_h, int orig_w,
                                     cudaStream_t stream = 0);

  // Batched detection: process N images in a single TRT inference call.
  // All images are resized to the same dimensions (max of the batch, rounded to 32).
  // Returns one vector<Box> per image.
  [[nodiscard]] std::vector<std::vector<Box>>
  run_batch(const std::vector<GpuImage> &gpu_imgs,
            const std::vector<std::pair<int,int>> &orig_dims,
            cudaStream_t stream = 0);

private:
  static constexpr float kDetDbThresh = 0.3f;
  static constexpr float kDetDbBoxThresh = 0.6f;
  static constexpr float kDetDbUnclipRatio = 1.5f;
  int kMaxSideLen_ = 960; // Configurable via DET_MAX_SIDE env var
  static constexpr float kMinBoxSide = 3.0f;
  static constexpr float kMinUnclippedSide = 5.0f; // kMinBoxSide + 2

  // GPU CCL mode: 0=CPU contours, 1=GPU CCL+per-ROI findContours (default, same accuracy + faster),
  // 2=GPU CCL fast (experimental, speed-only — poor accuracy F1~53%, not recommended for production)
  int gpu_ccl_mode_ = 1;

  std::unique_ptr<engine::TrtEngine> engine_;

  // Maximum batch size for batched detection
  static constexpr int kMaxBatchSize = 8;

  // Pre-allocated GPU buffers (single-image, RAII)
  CudaPtr<float> d_input_;
  CudaPtr<float> d_output_;
  size_t input_size_ = 0;
  size_t output_size_ = 0;

  // Pre-allocated batch GPU buffers (kMaxBatchSize images, RAII)
  CudaPtr<float> d_batch_input_;
  CudaPtr<float> d_batch_output_;
  CudaPtr<uint8_t> d_batch_bitmap_;
  size_t batch_input_size_ = 0;
  size_t batch_output_size_ = 0;

  // Device-side arrays for batched kernel launch params (RAII)
  CudaPtr<void *> d_batch_src_ptrs_;
  CudaPtr<int> d_batch_src_steps_;
  CudaPtr<int> d_batch_src_heights_;
  CudaPtr<int> d_batch_src_widths_;

  // Pinned host staging for batch metadata (RAII)
  CudaHostPtr<void *> h_batch_src_ptrs_;
  CudaHostPtr<int> h_batch_src_steps_;
  CudaHostPtr<int> h_batch_src_heights_;
  CudaHostPtr<int> h_batch_src_widths_;

  // Pre-allocated bitmap buffer (RAII)
  CudaPtr<uint8_t> d_bitmap_buf_;

  // Non-owning working pointers. Normally point to d_output_.get() and
  // d_bitmap_buf_.get(). In batch mode, temporarily aliased to slices of
  // d_batch_output_ / d_batch_bitmap_ for per-image post-processing.
  float *cur_output_ = nullptr;
  uint8_t *cur_bitmap_ = nullptr;

  // GPU CCL buffers (pre-allocated in load_model — NO per-request alloc, RAII)
  CudaPtr<int> d_ccl_labels_;
  CudaPtr<int> d_ccl_compact_ids_;     // [max_pixels] compact component IDs
  CudaPtr<int> d_ccl_id_counter_;      // [1] atomic counter for compact IDs
  CudaPtr<kernels::GpuDetBox> d_ccl_bboxes_;
  CudaPtr<int> d_ccl_num_boxes_;
  // Host-side result buffer for GPU CCL (pinned memory, RAII)
  CudaHostPtr<kernels::GpuDetBox> h_ccl_boxes_;

  // Reusable contour/mask buffers (avoid per-call heap allocation)
  std::vector<cv::Point> shifted_buf_;
  cv::Mat mask_buf_;
  std::vector<std::vector<cv::Point>> contours_buf_;
  std::vector<cv::Vec4i> hierarchy_buf_;

  // GPU CCL contour extraction buffers (reused per-component)
  std::vector<std::vector<cv::Point>> ccl_roi_contours_buf_;
  std::vector<cv::Point> ccl_contour_buf_;

  // Common buffer allocation (called by both load_model overloads)
  [[nodiscard]] bool init_buffers();

  // GPU CCL path: returns boxes from GPU + per-ROI findContours (accurate)
  [[nodiscard]] std::vector<Box> run_gpu_ccl(int resize_h, int resize_w,
                                              int orig_h, int orig_w,
                                              cudaStream_t stream);

  // GPU CCL fast (EXPERIMENTAL, GPU_CCL=2): direct bbox expansion, NO CPU contour processing.
  // Warning: poor accuracy (F1~53%). Use only for speed benchmarks, not production.
  [[nodiscard]] std::vector<Box> run_gpu_ccl_fast(int resize_h, int resize_w,
                                                    int orig_h, int orig_w,
                                                    cudaStream_t stream);

  // CPU fallback path (original findContours)
  [[nodiscard]] std::vector<Box> run_cpu_contours(int resize_h, int resize_w,
                                                   int orig_h, int orig_w,
                                                   cudaStream_t stream);

};

} // namespace turbo_ocr::detection
