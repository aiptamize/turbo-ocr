#pragma once

#include <vector>

#include "turbo_ocr/common/types.h"
#include "turbo_ocr/layout/layout_types.h"

namespace turbo_ocr::pipeline {

/// Bundles text OCR results + optional layout detections from a single
/// pipeline run. `layout` is empty when layout was not requested or the
/// pipeline has no layout model. `reading_order` is filled only when
/// reading-order assignment was requested. Both optional vectors stay
/// empty in the default path so the back-compat serializer emits a
/// byte-identical response.
struct OcrPipelineResult {
  std::vector<OCRResultItem>            results;
  std::vector<layout::LayoutBox>        layout;
  std::vector<int>                      reading_order;
};

} // namespace turbo_ocr::pipeline
