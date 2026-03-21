#pragma once

#include <memory>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "turbo_ocr/engine/cpu_engine.h"
#include "turbo_ocr/common/box.h"

namespace turbo_ocr::classification {

/// CPU angle classifier using ONNX Runtime (flips 180-degree text crops).
class CpuPaddleCls {
public:
  CpuPaddleCls() = default;
  ~CpuPaddleCls() noexcept = default;

  /// Load an ONNX classification model.
  [[nodiscard]] bool load_model(const std::string &model_path);

  // Classify crops and flip 180-degree boxes in-place.
  void run(const cv::Mat &img, std::vector<Box> &boxes);

private:
  static constexpr int kClsImageH = 48;
  static constexpr int kClsImageW = 192;
  static constexpr float kClsThresh = 0.9f;

  std::unique_ptr<engine::CpuEngine> engine_;
};

} // namespace turbo_ocr::classification
