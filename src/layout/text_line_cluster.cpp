#include "turbo_ocr/layout/text_line_cluster.h"

#include <algorithm>
#include <climits>

#include "turbo_ocr/common/box.h"

namespace turbo_ocr::layout {

namespace {

// 1D projection overlap ratio along [a0,a1) vs [b0,b1) using "small"
// mode — intersection divided by the smaller of the two extents.
// Mirrors paddlex.layout_parsing.utils.calculate_projection_overlap_ratio
// with mode="small" used by group_boxes_into_lines.
inline float projection_overlap_small(int a0, int a1, int b0, int b1) noexcept {
  const int inter = std::max(0, std::min(a1, b1) - std::max(a0, b0));
  if (inter <= 0) return 0.0f;
  const int len_a = std::max(0, a1 - a0);
  const int len_b = std::max(0, b1 - b0);
  const int small = std::min(len_a, len_b);
  return small > 0 ? float(inter) / float(small) : 0.0f;
}

// Per-result AABB (axis-aligned bbox of the 4-corner detection quad).
struct ResAabb {
  int x0, y0, x1, y1;
  int idx;  // index into results vector
};

// Direction vote for a single bbox using PaddleX's text-line ratio
// (width * 1.5 >= height ⇒ horizontal). Used both per-result and per-
// cell majority.
inline bool box_is_horizontal(const ResAabb &b) noexcept {
  const int w = b.x1 - b.x0;
  const int h = b.y1 - b.y0;
  return float(w) * kTextLineDirectionRatio >= float(h);
}

// A clustered text line: union bbox of the spans assigned to it.
struct TextLineAcc {
  int x0 = INT_MAX, y0 = INT_MAX, x1 = INT_MIN, y1 = INT_MIN;
  int height() const noexcept { return std::max(0, y1 - y0); }
  int width() const noexcept { return std::max(0, x1 - x0); }
  void add(const ResAabb &r) noexcept {
    x0 = std::min(x0, r.x0);
    y0 = std::min(y0, r.y0);
    x1 = std::max(x1, r.x1);
    y1 = std::max(y1, r.y1);
  }
};

} // namespace

void cluster_text_lines(const std::vector<OCRResultItem> &results,
                        std::vector<LayoutBox> &layout) {
  if (layout.empty()) return;

  // Gather per-cell result AABBs.
  std::vector<std::vector<ResAabb>> per_cell(layout.size());
  for (size_t i = 0; i < results.size(); ++i) {
    const int lid = results[i].layout_id;
    if (lid < 0 || static_cast<size_t>(lid) >= layout.size()) continue;
    auto [x0, y0, x1, y1] = turbo_ocr::aabb(results[i].box);
    per_cell[static_cast<size_t>(lid)].push_back(
        {x0, y0, x1, y1, static_cast<int>(i)});
  }

  for (size_t li = 0; li < layout.size(); ++li) {
    auto &cell = layout[li];
    auto &spans = per_cell[li];
    if (spans.empty()) {
      cell.direction = Direction::kHorizontal;
      cell.num_of_lines = 0;
      cell.text_line_height = 0;
      cell.text_line_width = 0;
      cell.seg_start_coordinate = 0;
      cell.seg_end_coordinate = 0;
      continue;
    }

    // 1. Per-cell direction vote.
    int horizontal_votes = 0;
    for (const auto &s : spans) if (box_is_horizontal(s)) ++horizontal_votes;
    cell.direction = (horizontal_votes >= int(spans.size() + 1) / 2)
                         ? Direction::kHorizontal
                         : Direction::kVertical;

    // 2. Sort spans by primary axis (y for horizontal, -x for
    //    vertical). Group_boxes_into_lines uses ascending y for
    //    horizontal and DESCENDING x for vertical (right-to-left CJK).
    if (cell.direction == Direction::kHorizontal) {
      std::sort(spans.begin(), spans.end(),
                [](const ResAabb &a, const ResAabb &b) {
                  return a.y0 < b.y0;
                });
    } else {
      std::sort(spans.begin(), spans.end(),
                [](const ResAabb &a, const ResAabb &b) {
                  return a.x0 > b.x0;
                });
    }

    // 3. Walk spans, grouping into lines. The match-direction is
    //    perpendicular: vertical for horizontal lines (we test if two
    //    spans share a y-band), horizontal for vertical lines.
    std::vector<TextLineAcc> lines;
    lines.reserve(spans.size());
    auto same_line = [&cell](const TextLineAcc &line, const ResAabb &s) {
      if (cell.direction == Direction::kHorizontal) {
        return projection_overlap_small(line.y0, line.y1, s.y0, s.y1) >=
               kLineHeightIouThreshold;
      }
      return projection_overlap_small(line.x0, line.x1, s.x0, s.x1) >=
             kLineHeightIouThreshold;
    };

    {
      TextLineAcc current;
      current.add(spans.front());
      for (size_t i = 1; i < spans.size(); ++i) {
        if (same_line(current, spans[i])) {
          current.add(spans[i]);
        } else {
          lines.push_back(current);
          current = TextLineAcc{};
          current.add(spans[i]);
        }
      }
      lines.push_back(current);
    }

    cell.num_of_lines = static_cast<int>(lines.size());

    // 4. Mean line height/width.
    long long sum_h = 0, sum_w = 0;
    for (const auto &ln : lines) {
      sum_h += ln.height();
      sum_w += ln.width();
    }
    cell.text_line_height = static_cast<int>(sum_h / lines.size());
    cell.text_line_width = static_cast<int>(sum_w / lines.size());

    // 5. Segment start/end coordinates along the cell's primary axis.
    //    PaddleX sets seg_start_coordinate to the FIRST line's left
    //    edge (horizontal) or top (vertical), and seg_end_coordinate
    //    to the LAST line's right edge or bottom.
    const auto &first = lines.front();
    const auto &last = lines.back();
    if (cell.direction == Direction::kHorizontal) {
      cell.seg_start_coordinate = first.x0;
      cell.seg_end_coordinate = last.x1;
    } else {
      cell.seg_start_coordinate = first.y0;
      cell.seg_end_coordinate = last.y1;
    }
  }
}

Direction infer_page_direction(const std::vector<LayoutBox> &layout) {
  int horizontal = 0;
  int total = 0;
  for (const auto &lb : layout) {
    // Only `text`-class cells (class_id 22) participate in page-level
    // direction vote — PaddleX's update_region_label only counts
    // normal_text_block_idxes, which excludes header/footer/title/etc.
    if (lb.class_id != 22) continue;
    ++total;
    if (lb.direction == Direction::kHorizontal) ++horizontal;
  }
  if (total == 0) return Direction::kHorizontal;
  return horizontal >= (total + 1) / 2 ? Direction::kHorizontal
                                       : Direction::kVertical;
}

} // namespace turbo_ocr::layout
