#pragma once

#include <algorithm>
#include <array>
#include <cmath>
#include <compare>
#include <cstddef>
#include <cstdint>
#include <cstdlib>
#include <ranges>
#include <vector>

namespace turbo_ocr {

// Stack-allocated bounding box: 4 corners [tl, tr, br, bl], each [x, y].
// Replaces std::vector<std::vector<int>> -- zero heap allocations.
struct Box {
  std::array<std::array<int, 2>, 4> pts; // [tl, tr, br, bl]

  constexpr auto &operator[](std::size_t i) noexcept { return pts[i]; }
  constexpr const auto &operator[](std::size_t i) const noexcept { return pts[i]; }

  // C++20 three-way comparison -- enables ==, !=, <, >, <=, >= automatically
  constexpr auto operator<=>(const Box &) const noexcept = default;
};

/// Vertical text threshold: crop_h >= crop_w * kVerticalAspectRatio.
/// Used consistently by box detection, classification, and recognition.
inline constexpr float kVerticalAspectRatio = 1.5f;

// Check if a box is vertically oriented (height >= width * 1.5).
// Uses integer arithmetic to avoid floating-point precision issues.
[[nodiscard]] inline bool is_vertical_box(const Box &b) noexcept {
  int w = std::max(std::abs(b[1][0] - b[0][0]), std::abs(b[2][0] - b[3][0]));
  int h = std::max(std::abs(b[3][1] - b[0][1]), std::abs(b[2][1] - b[1][1]));
  return static_cast<int64_t>(h) * h >= static_cast<int64_t>(w) * w * 225 / 100;
}

// Sort boxes top-to-bottom, left-to-right (in-place, deterministic)
// Quantize Y to line bands so boxes on the same line sort by X.
inline void sorted_boxes(std::vector<Box> &dt_boxes) {
  static constexpr int kSameLineThreshold = 10;
  std::ranges::stable_sort(dt_boxes, [](const Box &a, const Box &b) {
    int ya = a[0][1] / kSameLineThreshold;
    int yb = b[0][1] / kSameLineThreshold;
    if (ya != yb) return ya < yb;
    return a[0][0] < b[0][0];
  });
}

} // namespace turbo_ocr
