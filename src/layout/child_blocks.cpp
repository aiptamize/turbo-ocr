#include "turbo_ocr/layout/child_blocks.h"

#include <algorithm>
#include <array>
#include <cstdlib>
#include <unordered_set>

#include "turbo_ocr/common/box.h"

namespace turbo_ocr::layout {

namespace {

// Class IDs for the labels we need to test. Mirrors the PaddleX
// BLOCK_LABEL_MAP partitioning used in update_*_child_blocks.
constexpr int kClassDocTitle        = 6;
constexpr int kClassParagraphTitle  = 17;
constexpr int kClassFigureTitle     = 7;
constexpr int kClassImage           = 14;
constexpr int kClassTable           = 21;
constexpr int kClassChart           = 3;
constexpr int kClassText            = 22;

inline bool is_text(int class_id) noexcept {
  return class_id == kClassText;
}
inline bool is_vision(int class_id) noexcept {
  return class_id == kClassImage || class_id == kClassTable ||
         class_id == kClassChart;
}
inline bool is_vision_title(int class_id) noexcept {
  return class_id == kClassFigureTitle;
}
inline bool is_paragraph_title(int class_id) noexcept {
  return class_id == kClassParagraphTitle;
}

struct AABB { int x0, y0, x1, y1; };

inline AABB aabb_of(const LayoutBox &lb) noexcept {
  auto [x0, y0, x1, y1] = turbo_ocr::aabb(lb.box);
  return {x0, y0, x1, y1};
}

inline int area_of(const AABB &b) noexcept {
  return std::max(0, b.x1 - b.x0) * std::max(0, b.y1 - b.y0);
}

inline int short_side(const AABB &b) noexcept {
  return std::min(std::max(0, b.x1 - b.x0), std::max(0, b.y1 - b.y0));
}

inline int long_side(const AABB &b) noexcept {
  return std::max(std::max(0, b.x1 - b.x0), std::max(0, b.y1 - b.y0));
}

inline std::array<int, 2> centroid(const AABB &b) noexcept {
  return {(b.x0 + b.x1) / 2, (b.y0 + b.y1) / 2};
}

// Nearest-edge distance between two AABBs along the union of axes
// they don't overlap on. Mirrors get_nearest_edge_distance with
// uniform weights = 1.
inline int nearest_edge_distance(const AABB &a, const AABB &b) noexcept {
  const bool h_overlap = std::min(a.x1, b.x1) > std::max(a.x0, b.x0);
  const bool v_overlap = std::min(a.y1, b.y1) > std::max(a.y0, b.y0);
  if (h_overlap && v_overlap) return 0;
  int dx = 0, dy = 0;
  if (!h_overlap) dx = std::min(std::abs(a.x0 - b.x1), std::abs(a.x1 - b.x0));
  if (!v_overlap) dy = std::min(std::abs(a.y0 - b.y1), std::abs(a.y1 - b.y0));
  return dx + dy;
}

// IoU over the smaller of the two areas — matches calculate_overlap_ratio
// with mode="small". Returns 0 when either area is zero.
inline float overlap_small(const AABB &a, const AABB &b) noexcept {
  const int ix0 = std::max(a.x0, b.x0);
  const int iy0 = std::max(a.y0, b.y0);
  const int ix1 = std::min(a.x1, b.x1);
  const int iy1 = std::min(a.y1, b.y1);
  if (ix1 <= ix0 || iy1 <= iy0) return 0.0f;
  const int inter = (ix1 - ix0) * (iy1 - iy0);
  const int small = std::min(area_of(a), area_of(b));
  return small > 0 ? float(inter) / float(small) : 0.0f;
}

// Read the cluster pre-pass output for a layout cell. When the cell
// has no clustered text lines (no OCR detection landed in it) we
// fall back to the height-over-text_line_height approximation so
// non-text layout cells (image, table, chart) still produce
// sensible line counts for the proximity gates.
inline int num_lines_for(const LayoutBox &lb, int page_text_line_height) noexcept {
  if (lb.num_of_lines > 0) return lb.num_of_lines;
  const int h = std::max(1, lb.box[3][1] - lb.box[0][1]);
  const int tlh = std::max(1, page_text_line_height);
  return std::max(1, h / tlh);
}

// Per-cell text-line-height. Falls back to the page-level mean when
// the cell has no clustered lines (vision blocks etc.).
inline int line_height_for(const LayoutBox &lb,
                            int page_text_line_height) noexcept {
  return lb.text_line_height > 0 ? lb.text_line_height
                                  : std::max(1, page_text_line_height);
}

// A "vertical neighbour" relation — used to pick the prev/post block
// of a parent. Two blocks neighbour each other vertically when their
// horizontal projections overlap by more than the small-area
// threshold (0.1 in PaddleX's XYCUT_SETTINGS).
inline bool horizontally_overlaps(const AABB &a, const AABB &b,
                                   float thresh = 0.1f) noexcept {
  const int ix0 = std::max(a.x0, b.x0);
  const int ix1 = std::min(a.x1, b.x1);
  if (ix1 <= ix0) return false;
  const int inter = ix1 - ix0;
  const int small = std::min(a.x1 - a.x0, b.x1 - b.x0);
  return small > 0 && float(inter) / float(small) > thresh;
}

// Return indices into `candidates` of blocks that sit ABOVE `parent`
// (sorted by descending y2 — closest first) and BELOW `parent`
// (sorted by ascending y0 — closest first), restricted to those that
// horizontally overlap parent.
struct PrevPost {
  std::vector<int> prev;
  std::vector<int> post;
};

PrevPost get_nearest_neighbours(const AABB &parent,
                                const std::vector<LayoutBox> &layout,
                                const std::vector<int> &candidates) {
  PrevPost out;
  for (int idx : candidates) {
    const AABB cand = aabb_of(layout[idx]);
    if (!horizontally_overlaps(parent, cand)) continue;
    if (cand.y1 <= parent.y0) out.prev.push_back(idx);
    else if (cand.y0 >= parent.y1) out.post.push_back(idx);
  }
  std::sort(out.prev.begin(), out.prev.end(),
            [&](int a, int b) {
              return aabb_of(layout[a]).y1 > aabb_of(layout[b]).y1;
            });
  std::sort(out.post.begin(), out.post.end(),
            [&](int a, int b) {
              return aabb_of(layout[a]).y0 < aabb_of(layout[b]).y0;
            });
  return out;
}

// Mirrors update_doc_title_child_blocks. Children: small adjacent text
// blocks (sub-titles, author lines) and any text fully overlapping the
// title bbox.
void detect_doc_title_children(int parent_idx,
                                const std::vector<LayoutBox> &layout,
                                const std::vector<int> &text_indices,
                                int page_text_line_height,
                                std::vector<int> &children,
                                std::unordered_set<int> &claimed) {
  const AABB parent = aabb_of(layout[parent_idx]);
  const int parent_short = short_side(parent);
  const int parent_long = long_side(parent);
  PrevPost neigh = get_nearest_neighbours(parent, layout, text_indices);

  auto try_attach = [&](int idx) {
    if (claimed.count(idx)) return;
    const AABB cand = aabb_of(layout[idx]);
    const int short_s = short_side(cand);
    const int long_s = long_side(cand);
    if (short_s >= parent_short * 4 / 5) return;
    if (long_s >= parent_long && long_s <= parent_long * 3 / 2) return;
    if (num_lines_for(layout[idx], page_text_line_height) >= 3) return;
    const int tlh = line_height_for(layout[idx], page_text_line_height);
    if (nearest_edge_distance(parent, cand) >= tlh * 2) return;
    children.push_back(idx);
    claimed.insert(idx);
  };
  if (!neigh.prev.empty()) try_attach(neigh.prev.front());
  if (!neigh.post.empty()) try_attach(neigh.post.front());

  for (int idx : text_indices) {
    if (claimed.count(idx)) continue;
    const AABB cand = aabb_of(layout[idx]);
    if (overlap_small(parent, cand) > 0.9f) {
      children.push_back(idx);
      claimed.insert(idx);
    }
  }
}

// Mirrors update_paragraph_title_child_blocks. Children: same-class
// adjacent paragraph_titles with similar left edge (sub-headings).
// Uses the MIN of parent and candidate text_line_heights as PaddleX
// does (`min_text_line_height` in the reference) so a tall caption
// next to a short subtitle still passes the proximity gate.
void detect_paragraph_title_children(int parent_idx,
                                      const std::vector<LayoutBox> &layout,
                                      const std::vector<int> &paragraph_title_indices,
                                      int page_text_line_height,
                                      std::vector<int> &children,
                                      std::unordered_set<int> &claimed) {
  const AABB parent = aabb_of(layout[parent_idx]);
  const int parent_tlh = line_height_for(layout[parent_idx],
                                          page_text_line_height);
  PrevPost neigh = get_nearest_neighbours(parent, layout, paragraph_title_indices);

  auto try_attach_run = [&](const std::vector<int> &run) {
    for (int idx : run) {
      if (idx == parent_idx) continue;
      if (claimed.count(idx)) break;
      const AABB cand = aabb_of(layout[idx]);
      const int min_tlh = std::min(parent_tlh,
                                    line_height_for(layout[idx],
                                                    page_text_line_height));
      if (std::abs(cand.x0 - parent.x0) >= min_tlh * 2) break;
      if (nearest_edge_distance(parent, cand) > min_tlh * 3 / 2) break;
      children.push_back(idx);
      claimed.insert(idx);
    }
  };
  try_attach_run(neigh.prev);
  try_attach_run(neigh.post);
}

// Mirrors update_vision_child_blocks. Children: nearby vision_title
// blocks and small adjacent text caption blocks (vision_footnote).
void detect_vision_children(int parent_idx,
                             const std::vector<LayoutBox> &layout,
                             const std::vector<int> &text_indices,
                             const std::vector<int> &vision_title_indices,
                             int page_text_line_height,
                             std::vector<int> &children,
                             std::unordered_set<int> &claimed) {
  const AABB parent = aabb_of(layout[parent_idx]);
  const int parent_short = short_side(parent);
  const int parent_long = long_side(parent);
  const auto parent_c = centroid(parent);

  bool has_vision_footnote = false;

  std::vector<int> ref_indices;
  ref_indices.reserve(text_indices.size() + vision_title_indices.size());
  ref_indices.insert(ref_indices.end(), text_indices.begin(), text_indices.end());
  ref_indices.insert(ref_indices.end(), vision_title_indices.begin(),
                      vision_title_indices.end());

  PrevPost neigh = get_nearest_neighbours(parent, layout, ref_indices);

  auto consider = [&](int idx) {
    if (claimed.count(idx)) return false;
    const AABB cand = aabb_of(layout[idx]);
    const int dist = nearest_edge_distance(parent, cand);
    const int cls = layout[idx].class_id;
    const int cand_tlh = line_height_for(layout[idx], page_text_line_height);
    if (is_vision_title(cls) && dist <= cand_tlh * 2) {
      children.push_back(idx);
      claimed.insert(idx);
      return true;
    }
    if (is_text(cls) && !has_vision_footnote &&
        long_side(cand) < parent_long &&
        dist <= cand_tlh * 2) {
      const auto cand_c = centroid(cand);
      const int cand_lines = num_lines_for(layout[idx], page_text_line_height);
      const bool tight_caption =
          short_side(cand) < parent_short &&
          long_side(cand) < parent_long / 2 &&
          std::abs(parent_c[0] - cand_c[0]) < 10;
      const bool aligned_left =
          std::abs(parent.x0 - cand.x0) < 10 && cand_lines == 1;
      const bool aligned_right =
          std::abs(parent.x1 - cand.x1) < 10 && cand_lines == 1;
      if (tight_caption || aligned_left || aligned_right) {
        has_vision_footnote = true;
        children.push_back(idx);
        claimed.insert(idx);
        return true;
      }
    }
    return false;
  };

  for (int idx : neigh.prev) { if (!consider(idx)) break; }
  for (int idx : neigh.post) { if (!consider(idx)) break; }

  // Fully-overlapping text ⇒ vision_footnote unconditionally.
  for (int idx : text_indices) {
    if (claimed.count(idx)) continue;
    const AABB cand = aabb_of(layout[idx]);
    if (overlap_small(parent, cand) > 0.9f) {
      children.push_back(idx);
      claimed.insert(idx);
    }
  }
}

} // namespace

std::vector<ChildLinks>
detect_child_blocks(const std::vector<LayoutBox> &layout,
                    int text_line_height) {
  std::vector<ChildLinks> out(layout.size());
  if (layout.empty() || text_line_height <= 0) return out;

  std::vector<int> text_indices, paragraph_title_indices,
                   vision_title_indices, vision_indices, doc_title_indices;
  for (size_t i = 0; i < layout.size(); ++i) {
    const int cls = layout[i].class_id;
    if (is_text(cls)) text_indices.push_back(int(i));
    else if (is_paragraph_title(cls)) paragraph_title_indices.push_back(int(i));
    else if (is_vision_title(cls)) vision_title_indices.push_back(int(i));
    else if (is_vision(cls)) vision_indices.push_back(int(i));
    else if (cls == kClassDocTitle) doc_title_indices.push_back(int(i));
  }

  std::unordered_set<int> claimed;

  // Vision parents claim children first — vision_footnote text/captions
  // are the most consequential gluing case in PaddleX's pipeline.
  for (int idx : vision_indices) {
    if (claimed.count(idx)) continue;
    detect_vision_children(idx, layout, text_indices, vision_title_indices,
                           text_line_height, out[idx].child_indices, claimed);
  }
  // Paragraph titles next — they may pull other paragraph_titles as
  // sub-headings. A paragraph_title that has already been claimed (e.g.
  // as another title's sub-heading) cannot itself become a parent —
  // mirrors PaddleX's order_label == "sub_paragraph_title" early
  // return.
  for (int idx : paragraph_title_indices) {
    if (claimed.count(idx)) continue;
    detect_paragraph_title_children(idx, layout, paragraph_title_indices,
                                    text_line_height, out[idx].child_indices,
                                    claimed);
  }
  // Doc title last — accrues remaining adjacent text (subtitle / author).
  for (int idx : doc_title_indices) {
    if (claimed.count(idx)) continue;
    detect_doc_title_children(idx, layout, text_indices, text_line_height,
                              out[idx].child_indices, claimed);
  }

  return out;
}

std::vector<int>
flatten_descendants(int parent_idx,
                    const std::vector<ChildLinks> &links,
                    const std::vector<LayoutBox> &layout) {
  std::vector<int> out;
  if (parent_idx < 0 || static_cast<size_t>(parent_idx) >= layout.size())
    return out;
  if (links.size() != layout.size()) return out;
  std::unordered_set<int> visited;
  visited.insert(parent_idx);
  // Iterative DFS keyed off a worklist of (idx, child_position) pairs.
  // Each frame remembers its sorted children so we can resume after
  // descending into a child. depth_limit equal to layout.size() is
  // enough for any acyclic tree; anything deeper means a cycle that
  // visited didn't catch (defence in depth).
  struct Frame {
    int idx;
    std::vector<int> kids;
    size_t cursor;
  };
  std::vector<Frame> stack;
  auto sorted_kids = [&](int idx) {
    std::vector<int> kids;
    if (idx < 0 || static_cast<size_t>(idx) >= links.size()) return kids;
    kids = links[static_cast<size_t>(idx)].child_indices;
    std::sort(kids.begin(), kids.end(), [&](int a, int b) {
      auto [ax0, ay0, ax1, ay1] = turbo_ocr::aabb(layout[a].box);
      auto [bx0, by0, bx1, by1] = turbo_ocr::aabb(layout[b].box);
      if (ay0 != by0) return ay0 < by0;
      return ax0 < bx0;
    });
    return kids;
  };
  stack.push_back({parent_idx, sorted_kids(parent_idx), 0});
  const size_t depth_limit = layout.size() + 1;
  while (!stack.empty()) {
    if (stack.size() > depth_limit) break;  // pathological cycle guard
    auto &top = stack.back();
    if (top.cursor >= top.kids.size()) {
      stack.pop_back();
      continue;
    }
    const int ci = top.kids[top.cursor++];
    if (ci < 0 || static_cast<size_t>(ci) >= layout.size()) continue;
    if (!visited.insert(ci).second) continue;  // skip if already seen
    out.push_back(ci);
    stack.push_back({ci, sorted_kids(ci), 0});
  }
  return out;
}

void splice_child_blocks(std::vector<UnsortedBlock> &sorted,
                         const std::vector<ChildLinks> &links,
                         const std::vector<LayoutBox> &layout) {
  if (sorted.empty() || links.empty()) return;

  // 1. Build a set of indices that are children of some parent — these
  //    must NOT emit standalone in `sorted`. Walk every parent's
  //    children to populate.
  std::unordered_set<int> child_layout_idxs;
  for (const auto &cl : links) {
    for (int ci : cl.child_indices) child_layout_idxs.insert(ci);
  }

  // 2. Filter `sorted` removing standalone child entries.
  std::vector<UnsortedBlock> filtered;
  filtered.reserve(sorted.size());
  for (const auto &b : sorted) {
    if (b.layout_idx >= 0 && child_layout_idxs.count(b.layout_idx)) continue;
    filtered.push_back(b);
  }

  // 3. For each parent that has descendants (anywhere in the
  //    sub-tree), splice (parent, descendants…) into `filtered` at
  //    the parent's position. Descendants come right after the parent
  //    in DFS order, each level sorted top-then-left — matching
  //    flatten_descendants' walk.
  std::vector<UnsortedBlock> out;
  out.reserve(filtered.size() + child_layout_idxs.size());
  for (const auto &b : filtered) {
    out.push_back(b);
    if (b.layout_idx < 0 ||
        static_cast<size_t>(b.layout_idx) >= links.size()) continue;
    const auto descendants = flatten_descendants(b.layout_idx, links, layout);
    for (int ci : descendants) {
      auto [x0, y0, x1, y1] = turbo_ocr::aabb(layout[ci].box);
      out.push_back({ci, {x0, y0, x1, y1}, OrderLabel::kBody,
                     layout[ci].class_id});
    }
  }
  sorted = std::move(out);
}

} // namespace turbo_ocr::layout
