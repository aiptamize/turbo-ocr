#pragma once

// CUDA-free types for PP-DocLayoutV3 layout detection results. Lives in a
// separate header from paddle_layout.h so that the JSON serializer — and
// unit tests that don't link against CUDA — can include this without
// dragging in <cuda_runtime.h> through cuda_ptr.h.

#include <array>
#include <string_view>

#include "turbo_ocr/common/box.h"

namespace turbo_ocr::layout {

// 25 class labels emitted by PP-DocLayoutV3. Order matches
// ~/.paddlex/official_models/PP-DocLayoutV3/inference.yml label_list.
// Size is deduced via std::to_array so a mismatch between the comment and
// the data can't silently leave a trailing empty slot.
inline constexpr auto kLayoutLabels = std::to_array<std::string_view>({
    "abstract",        "algorithm",       "aside_text",    "chart",
    "content",         "display_formula", "doc_title",     "figure_title",
    "footer",          "footer_image",    "footnote",      "formula_number",
    "header",          "header_image",    "image",         "inline_formula",
    "number",          "paragraph_title", "reference",     "reference_content",
    "seal",            "table",           "text",          "vertical_text",
    "vision_footnote",
});

inline constexpr std::string_view label_name(int class_id) noexcept {
  if (class_id >= 0 && class_id < static_cast<int>(kLayoutLabels.size()))
    return kLayoutLabels[class_id];
  return {};
}

struct LayoutBox {
  int class_id = 0;
  float score = 0.0f;
  // Axis-aligned 4-corner box in the ORIGINAL input image's coordinate
  // system. PP-DocLayoutV3 applies im_shape + scale_factor internally, so
  // the model's output is already in original coordinates.
  Box box{};
  // Cross-reference ID emitted when layout detection is enabled. Default
  // -1 means "not assigned" and the serializer omits the field.
  int id = -1;
};

} // namespace turbo_ocr::layout
