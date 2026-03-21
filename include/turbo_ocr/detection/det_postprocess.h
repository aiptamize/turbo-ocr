#pragma once

#include <opencv2/core.hpp>
#include <vector>

#include "turbo_ocr/common/box.h"

namespace turbo_ocr::detection {

// Detection post-processing free functions (shared between GPU and CPU detectors)

// Compute mean probability inside contour polygon.
// shifted_buf and mask_buf are caller-owned scratch buffers for thread-safety.
[[nodiscard]] float box_score_fast(const cv::Mat &pred_map,
                                   const std::vector<cv::Point> &contour,
                                   std::vector<cv::Point> &shifted_buf,
                                   cv::Mat &mask_buf);

// Expand polygon using Clipper library.
[[nodiscard]] std::vector<cv::Point> unclip(const std::vector<cv::Point> &polygon,
                                             float unclip_ratio);

// Extract ordered [tl, tr, br, bl] from min-area rotated rect.
[[nodiscard]] Box get_mini_boxes(const std::vector<cv::Point> &contour, float &min_side);

// Extract boxes from contours -- the shared loop used by both GPU and CPU detectors.
[[nodiscard]] std::vector<Box> extract_boxes_from_bitmap(
    const cv::Mat &pred_map, cv::Mat &bitmap,
    int orig_h, int orig_w, int resize_h, int resize_w,
    float det_db_box_thresh, float det_db_unclip_ratio,
    float min_box_side, float min_unclipped_side,
    std::vector<cv::Point> &shifted_buf, cv::Mat &mask_buf,
    std::vector<std::vector<cv::Point>> &contours_buf,
    std::vector<cv::Vec4i> &hierarchy_buf);

} // namespace turbo_ocr::detection
