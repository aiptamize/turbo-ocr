#pragma once

// Layer 2 of the PaddleX layout reading-order pipeline: label-aware
// insertion of "unsorted" blocks (titles, captions, references) into a
// list already sorted by XY-cut. Mirrors paddlex/inference/pipelines/
// layout_parsing/xycut_enhanced/utils.py functions:
//
//   - reference_insert
//   - manhattan_insert
//   - euclidean_insert
//   - weighted_distance_insert
//   - match_unsorted_blocks (dispatch)
//
// Simplifications vs the Python reference:
//   - We assume horizontal page direction. PP-DocLayoutV3 itself has
//     no opinion on page direction — PaddleX infers it from bbox
//     aspect ratios via a majority vote across the region's
//     text-class children. Vertical-text documents (Japanese
//     tategaki, traditional Chinese / Korean) will have scrambled
//     cross-column reading order; locally text remains correct.
//     Latin / Cyrillic / Greek / Arabic / Thai / Latin-typeset
//     Chinese / Korean (i.e. >99% of expected production traffic)
//     are unaffected.
//   - No LayoutRegion abstraction — text_line_width / text_line_height
//     are passed through as scalars derived from the median width /
//     height of all text-class blocks on the page.
//   - tolerance_len for non-doc_title blocks is fixed at 2 px
//     (PaddleX's default in XYCUT_SETTINGS["edge_distance_compare_
//     tolerance_len"]); doc_title scales by text_line_width.
//   - No `get_seg_flag` look-ahead — its inputs require per-paragraph
//     text-line metadata we don't compute. Approximate via a
//     "neighbour is multi-line and horizontally encloses" heuristic
//     in the vision step-back branch.

#include <array>
#include <vector>

#include "turbo_ocr/layout/layout_types.h"

namespace turbo_ocr::layout {

// Coarse-grained role used by the label-aware insertion strategies.
// Mirrors PaddleX's BLOCK_LABEL_MAP partitioning.
enum class OrderLabel : int {
  kBody = 0,         // text, formula, algorithm — sorted by XY-cut directly
  kDocTitle,         // doc_title — pinned to position 0
  kParagraphTitle,   // paragraph_title etc. — weighted_distance_insert
  kVisionTitle,      // figure_title, table_title — weighted_distance_insert
  kVision,           // image, table, chart, figure — weighted_distance_insert
  kCrossReference,   // reference, footnote, vision_footnote — reference_insert
  kUnordered,        // aside_text, seal, page number, formula_number — manhattan_insert
};

// Map a PP-DocLayoutV3 class_id to an OrderLabel. Class IDs not in any
// special bucket fall through to kBody (which means "use whatever the
// XY-cut bucketing decides").
[[nodiscard]] OrderLabel order_label_for(int class_id) noexcept;

// Insert `block` into `sorted_blocks` based on the strategy keyed by
// `block`'s order label. Mirrors match_unsorted_blocks.
//
// `sorted_blocks` carries indices into the caller's layout vector.
// Each entry is paired with its AABB so we don't have to refetch on
// every comparison. Mutates sorted_blocks by inserting block.
struct UnsortedBlock {
  int layout_idx;             // index into the outer `layout` vector
  std::array<int, 4> aabb;    // x0, y0, x1, y1
  OrderLabel order_label;
  int class_id;
};

void match_unsorted_block(UnsortedBlock block,
                          std::vector<UnsortedBlock> &sorted_blocks,
                          int text_line_width);

// Convenience: insert every block in `unsorted` into `sorted` using its
// per-block strategy. Iteration order matches PaddleX:
//   - kDocTitle blocks first (so the "pin to 0" rule lands them at top)
//   - everything else after
void match_unsorted_blocks(std::vector<UnsortedBlock> &sorted,
                           std::vector<UnsortedBlock> &unsorted,
                           int text_line_width);

} // namespace turbo_ocr::layout
