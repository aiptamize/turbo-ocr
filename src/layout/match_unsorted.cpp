#include "turbo_ocr/layout/match_unsorted.h"

#include <algorithm>
#include <cstdlib>
#include <limits>

namespace turbo_ocr::layout {

OrderLabel order_label_for(int class_id) noexcept {
  switch (class_id) {
    case 6:  // doc_title
      return OrderLabel::kDocTitle;
    case 17: // paragraph_title
      return OrderLabel::kParagraphTitle;
    case 7:  // figure_title
      return OrderLabel::kVisionTitle;
    case 14: // image
    case 21: // table
    case 3:  // chart
      return OrderLabel::kVision;
    case 18: // reference
    case 19: // reference_content
    case 10: // footnote
    case 24: // vision_footnote
      return OrderLabel::kCrossReference;
    case 2:  // aside_text
    case 16: // number (page number)
    case 11: // formula_number
    case 20: // seal
      return OrderLabel::kUnordered;
    default:
      return OrderLabel::kBody;
  }
}

namespace {

// 1D projection overlap ratio along an axis. Mirrors
// paddlex.layout_parsing.utils.calculate_projection_overlap_ratio with
// mode="iou": intersection / union of the projected segments.
inline float projection_overlap(int a0, int a1, int b0, int b1) noexcept {
  const int inter = std::max(0, std::min(a1, b1) - std::max(a0, b0));
  if (inter <= 0) return 0.0f;
  const int uni = std::max(a1, b1) - std::min(a0, b0);
  return uni > 0 ? float(inter) / float(uni) : 0.0f;
}

// get_nearest_edge_distance(bbox1, bbox2, weight)
// weight = [left, right, up, down] — applied to the corresponding gap.
//
// If the boxes overlap on BOTH axes (horizontal_iou>0 AND vertical_iou>0)
// the distance is 0. Otherwise we take the gap on each non-overlapping
// axis (multiplied by the directional weight) and sum.
inline float nearest_edge_distance(const std::array<int, 4> &a,
                                   const std::array<int, 4> &b,
                                   const std::array<float, 4> &w) noexcept {
  const auto [x1, y1, x2, y2] = a;
  const auto [x1p, y1p, x2p, y2p] = b;
  const float h_iou = projection_overlap(x1, x2, x1p, x2p);
  const float v_iou = projection_overlap(y1, y2, y1p, y2p);
  if (h_iou > 0 && v_iou > 0) return 0.0f;

  float dx = 0.0f, dy = 0.0f;
  if (h_iou == 0.0f) {
    const int gap = std::min(std::abs(x1 - x2p), std::abs(x2 - x1p));
    dx = static_cast<float>(gap) * (x2 < x1p ? w[0] : w[1]);
  }
  if (v_iou == 0.0f) {
    const int gap = std::min(std::abs(y1 - y2p), std::abs(y2 - y1p));
    dy = static_cast<float>(gap) * (y2 < y1p ? w[2] : w[3]);
  }
  return dx + dy;
}

// Manhattan distance between bbox top-left corners.
inline float manhattan_dist(const std::array<int, 4> &a,
                            const std::array<int, 4> &b) noexcept {
  return float(std::abs(a[0] - b[0]) + std::abs(a[1] - b[1]));
}

// Squared distance from origin to the centroid of a bbox.
inline double centroid_l2_sq(const std::array<int, 4> &a) noexcept {
  const double cx = 0.5 * (a[0] + a[2]);
  const double cy = 0.5 * (a[1] + a[3]);
  return cx * cx + cy * cy;
}

// _get_weights(label, direction="horizontal") for the four insertion
// gap-direction weights. Mirrors xycut_enhanced/utils.py:_get_weights.
inline std::array<float, 4> weights_for(OrderLabel ol) noexcept {
  switch (ol) {
    case OrderLabel::kDocTitle:
      return {1.0f, 0.1f, 0.1f, 1.0f};
    case OrderLabel::kParagraphTitle:
    case OrderLabel::kVisionTitle:
    case OrderLabel::kVision:
      return {1.0f, 1.0f, 0.1f, 1.0f};
    default:
      return {1.0f, 1.0f, 1.0f, 0.1f};
  }
}

constexpr float kEdgeDistanceTolerance = 2.0f;
constexpr float kEdgeWeight = 1e4f;
constexpr float kUpEdgeWeight = 1.0f;
constexpr float kLeftEdgeWeight = 1e-4f;

// reference_insert: pick the highest sorted block that is fully ABOVE
// `block` (sorted_block.bbox[3] <= block.bbox[1]) and insert AFTER it.
// Used for cross-references / footnotes that should appear at the END
// of whatever section they textually belong to.
void reference_insert(UnsortedBlock block,
                      std::vector<UnsortedBlock> &sorted_blocks) {
  if (sorted_blocks.empty()) {
    sorted_blocks.push_back(block);
    return;
  }
  float min_distance = std::numeric_limits<float>::infinity();
  size_t nearest = 0;
  for (size_t i = 0; i < sorted_blocks.size(); ++i) {
    const auto &sb = sorted_blocks[i];
    if (sb.aabb[3] > block.aabb[1]) continue;
    const float distance = -float(sb.aabb[2]) * 10.0f - float(sb.aabb[3]);
    if (distance < min_distance) {
      min_distance = distance;
      nearest = i;
    }
  }
  sorted_blocks.insert(sorted_blocks.begin() + nearest + 1, block);
}

// manhattan_insert: nearest by Manhattan-distance between top-left
// corners. Used for `unordered` (aside_text, page numbers, seals).
void manhattan_insert(UnsortedBlock block,
                      std::vector<UnsortedBlock> &sorted_blocks) {
  if (sorted_blocks.empty()) {
    sorted_blocks.push_back(block);
    return;
  }
  float min_distance = std::numeric_limits<float>::infinity();
  size_t nearest = 0;
  for (size_t i = 0; i < sorted_blocks.size(); ++i) {
    const float d = manhattan_dist(block.aabb, sorted_blocks[i].aabb);
    if (d < min_distance) {
      min_distance = d;
      nearest = i;
    }
  }
  sorted_blocks.insert(sorted_blocks.begin() + nearest + 1, block);
}

// euclidean_insert: place by ascending centroid distance from origin.
// Used for whole-region blocks where simple radial ordering is enough.
void euclidean_insert(UnsortedBlock block,
                      std::vector<UnsortedBlock> &sorted_blocks) {
  const double block_d = centroid_l2_sq(block.aabb);
  size_t pos = sorted_blocks.size();
  for (size_t i = 0; i < sorted_blocks.size(); ++i) {
    if (centroid_l2_sq(sorted_blocks[i].aabb) > block_d) {
      pos = i;
      break;
    }
  }
  sorted_blocks.insert(sorted_blocks.begin() + pos, block);
}

// weighted_distance_insert: the heavy one. Combines:
//   - edge distance with per-label directional weights
//   - up-edge distance (vertical position)
//   - left-edge distance (horizontal position)
// Each block iteration tracks the minimum aggregate; the chosen
// nearest block determines the insertion point, with a "below-or-above"
// secondary heuristic comparing the candidate's y1/x1 to the sorted
// block's. Used for titles, captions, and vision blocks.
//
// `text_line_width` is the median width of `text`-class blocks on the
// page, used to scale the doc_title disperse tolerance — without it the
// disperse term collapses to 0 and a doc_title's edge_distance ties
// with the wrong neighbour, landing it mid-paragraph.
//
// Vision look-ahead approximation: PaddleX uses get_seg_flag (which
// depends on per-paragraph text-line metadata we don't compute) to
// decide whether to bump insertion ±1 around a vision block. Lacking
// that signal we step back by 1 only when the neighbour is plausibly
// multi-line — a simple "neighbour is taller than 1.5× the block AND
// horizontally encloses it" heuristic that fires when the neighbour
// is a real text paragraph but stays put when it's a single line.
void weighted_distance_insert(UnsortedBlock block,
                              std::vector<UnsortedBlock> &sorted_blocks,
                              int text_line_width) {
  if (sorted_blocks.empty()) {
    sorted_blocks.push_back(block);
    return;
  }
  const auto weights = weights_for(block.order_label);
  const int x1 = block.aabb[0], y1 = block.aabb[1];
  const int x2 = block.aabb[2];

  // Disperse term: doc_title gets a tolerance of max(2, text_line_width)
  // px so up-edge distance ties cleanly across same-row neighbours
  // (mirrors PaddleX disperse = max(1, region.text_line_width)).
  float tolerance_len = kEdgeDistanceTolerance;
  if (block.order_label == OrderLabel::kDocTitle && text_line_width > 0) {
    tolerance_len = std::max(tolerance_len, float(text_line_width));
  }

  float min_weighted = std::numeric_limits<float>::infinity();
  float min_up_edge = std::numeric_limits<float>::infinity();
  size_t nearest = 0;

  for (size_t i = 0; i < sorted_blocks.size(); ++i) {
    const auto &sb = sorted_blocks[i];
    const int y1p = sb.aabb[1];
    const int x1p = sb.aabb[0];
    const int y2p = sb.aabb[3];

    float edge_distance = nearest_edge_distance(block.aabb, sb.aabb, weights);
    float up_edge = float(y1p);
    float left_edge = float(x1p);
    const bool is_below_sorted = y2p < y1;

    const bool flip_for_below =
        (block.order_label == OrderLabel::kDocTitle ||
         block.order_label == OrderLabel::kParagraphTitle ||
         block.order_label == OrderLabel::kVisionTitle ||
         block.order_label == OrderLabel::kVision) &&
        is_below_sorted;
    if (flip_for_below) {
      up_edge = -up_edge;
      left_edge = -left_edge;
    }

    if (std::abs(min_up_edge - up_edge) <= tolerance_len) {
      up_edge = min_up_edge;
    }

    const float weighted = edge_distance * kEdgeWeight +
                           up_edge * kUpEdgeWeight +
                           left_edge * kLeftEdgeWeight;
    if (up_edge < min_up_edge) min_up_edge = up_edge;

    if (weighted < min_weighted) {
      min_weighted = weighted;
      nearest = i;

      int sorted_distance, block_distance;
      if (std::abs(y1 / 2 - y1p / 2) > 0) {
        sorted_distance = y1p;
        block_distance = y1;
      } else if (std::abs(x1 / 2 - x2 / 2) > 0) {
        sorted_distance = x1p;
        block_distance = x1;
      } else {
        const double sb_c = centroid_l2_sq(sb.aabb);
        const double bl_c = centroid_l2_sq(block.aabb);
        sorted_distance = static_cast<int>(sb_c);
        block_distance  = static_cast<int>(bl_c);
      }
      if (block_distance > sorted_distance) {
        nearest = i + 1;
      } else if (i > 0) {
        // PaddleX's get_seg_flag check requires per-paragraph text-line
        // metadata. Approximate: only step back into the previous slot
        // when the neighbour at i looks like a multi-line paragraph
        // that spatially encloses the block (so insertion BEFORE it
        // doesn't break a paragraph).
        const bool is_vision =
            block.order_label == OrderLabel::kVision ||
            block.order_label == OrderLabel::kVisionTitle;
        if (is_vision) {
          const int sb_height = sb.aabb[3] - sb.aabb[1];
          const int sb_width = sb.aabb[2] - sb.aabb[0];
          const int bl_height = std::max(1, block.aabb[3] - block.aabb[1]);
          const bool neighbour_is_multiline =
              sb_height > bl_height * 3 / 2;
          const bool neighbour_encloses_horizontally =
              sb.aabb[0] <= x1 && sb.aabb[2] >= x2 &&
              sb_width > (x2 - x1);
          if (neighbour_is_multiline && neighbour_encloses_horizontally) {
            nearest = i - 1;
          }
        }
      }
    }
  }
  if (nearest > sorted_blocks.size()) nearest = sorted_blocks.size();
  sorted_blocks.insert(sorted_blocks.begin() + nearest, block);
}

} // namespace

void match_unsorted_block(UnsortedBlock block,
                          std::vector<UnsortedBlock> &sorted_blocks,
                          int text_line_width) {
  switch (block.order_label) {
    case OrderLabel::kCrossReference:
      reference_insert(block, sorted_blocks);
      break;
    case OrderLabel::kUnordered:
      manhattan_insert(block, sorted_blocks);
      break;
    case OrderLabel::kDocTitle:
    case OrderLabel::kParagraphTitle:
    case OrderLabel::kVisionTitle:
    case OrderLabel::kVision:
      weighted_distance_insert(block, sorted_blocks, text_line_width);
      break;
    case OrderLabel::kBody:
    default:
      euclidean_insert(block, sorted_blocks);
      break;
  }
}

void match_unsorted_blocks(std::vector<UnsortedBlock> &sorted,
                           std::vector<UnsortedBlock> &unsorted,
                           int text_line_width) {
  // Two-pass to honour the "doc_title pinned to 0 if first" rule: any
  // doc_title in `unsorted` goes through weighted_distance_insert first
  // (the algorithm naturally lands the first one at the top-left).
  std::stable_partition(unsorted.begin(), unsorted.end(),
                        [](const UnsortedBlock &b) {
                          return b.order_label == OrderLabel::kDocTitle;
                        });
  for (auto &b : unsorted) {
    match_unsorted_block(b, sorted, text_line_width);
  }
  unsorted.clear();
}

} // namespace turbo_ocr::layout
