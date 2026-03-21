#pragma once

#include "turbo_ocr/common/box.h"
#include <string>

namespace turbo_ocr {

struct OCRResultItem {
  std::string text;
  float confidence = 0.0f;
  Box box{};
};

/// Minimum confidence to keep a recognition result (matching PaddleOCR Python).
inline constexpr float kDropScore = 0.5f;

} // namespace turbo_ocr
