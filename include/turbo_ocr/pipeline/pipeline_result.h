#pragma once

#include <vector>

#include "turbo_ocr/common/types.h"
#include "turbo_ocr/layout/layout_types.h"

namespace turbo_ocr::pipeline {

/// Bundles text OCR results + optional layout detections from a single
/// pipeline run. `layout` is empty when layout was not requested or
/// the pipeline has no layout model.
struct OcrPipelineResult {
  std::vector<OCRResultItem>      results;
  std::vector<layout::LayoutBox>  layout;
};

} // namespace turbo_ocr::pipeline
