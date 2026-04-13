#pragma once

#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "turbo_ocr/layout/layout_types.h"

namespace turbo_ocr::layout {

/// CPU layout detection using ONNX Runtime (PP-DocLayoutV3).
/// Drop-in replacement for PaddleLayout when no GPU is available.
class CpuPaddleLayout {
public:
  CpuPaddleLayout();
  ~CpuPaddleLayout() noexcept;

  [[nodiscard]] bool load_model(const std::string &onnx_path);

  /// Run layout detection on a CPU image. Returns detected layout boxes.
  [[nodiscard]] std::vector<LayoutBox> run(const cv::Mat &img,
                                           float score_threshold = 0.3f);

  static constexpr int kInputSize = 800;
  static constexpr int kMaxDetections = 300;

private:
  struct Impl;
  std::unique_ptr<Impl> impl_;
};

} // namespace turbo_ocr::layout
