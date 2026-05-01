#pragma once

#include <memory>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "turbo_ocr/engine/cpu_engine.h"
#include "turbo_ocr/common/box.h"

namespace turbo_ocr::detection {

/// CPU text detector using ONNX Runtime (DB post-processing).
class CpuPaddleDet {
public:
  CpuPaddleDet() = default;
  ~CpuPaddleDet() noexcept = default;

  /// Load an ONNX detection model.
  [[nodiscard]] bool load_model(const std::string &model_path);

  // Run detection on a CPU cv::Mat image (BGR, uint8)
  [[nodiscard]] std::vector<Box> run(const cv::Mat &img);

private:
  static constexpr float kDetDbThresh = 0.3f;
  static constexpr float kDetDbBoxThresh = 0.6f;
  static constexpr float kDetDbUnclipRatio = 1.5f;
  // Configurable via DET_MAX_SIDE (default 960, clamp [32, 4096]). Read
  // from detection/det_config.h at load_model time. Was hardcoded to 960
  // before, which silently truncated CPU output when the GPU pipeline was
  // configured larger.
  int kMaxSideLen = 960;
  static constexpr float kMinBoxSide = 3.0f;
  static constexpr float kMinUnclippedSide = 5.0f;

  std::unique_ptr<engine::CpuEngine> engine_;

  // Reusable buffers (avoid per-call heap allocation)
  std::vector<cv::Point> shifted_buf_;
  cv::Mat mask_buf_;
  std::vector<std::vector<cv::Point>> contours_buf_;
  std::vector<cv::Vec4i> hierarchy_buf_;
  std::vector<float> input_data_buf_;
  std::vector<int64_t> input_shape_buf_{1, 3, 0, 0};

};

} // namespace turbo_ocr::detection
