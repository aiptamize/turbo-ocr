#include "turbo_ocr/detection/det_postprocess.h"
#include "clipper/clipper.hpp"

#include <algorithm>
#include <cmath>
#include <iostream>
#include <ranges>
#include <opencv2/imgproc.hpp>

using turbo_ocr::Box;

namespace turbo_ocr::detection {

float box_score_fast(const cv::Mat &pred_map,
                     const std::vector<cv::Point> &contour,
                     std::vector<cv::Point> &shifted_buf,
                     cv::Mat &mask_buf) {
  int h = pred_map.rows;
  int w = pred_map.cols;

  int xmin = w, xmax = 0, ymin = h, ymax = 0;
  for (const auto &pt : contour) {
    xmin = std::min(xmin, std::max(0, pt.x));
    xmax = std::max(xmax, std::min(w - 1, pt.x));
    ymin = std::min(ymin, std::max(0, pt.y));
    ymax = std::max(ymax, std::min(h - 1, pt.y));
  }
  if (xmax <= xmin || ymax <= ymin)
    return 0.0f;

  // Shift contour to local ROI coords (reuse caller's buffer)
  shifted_buf.clear();
  shifted_buf.reserve(contour.size());
  for (const auto &pt : contour)
    shifted_buf.push_back(cv::Point(pt.x - xmin, pt.y - ymin));

  int mask_h = ymax - ymin + 1, mask_w = xmax - xmin + 1;
  if (mask_buf.rows != mask_h || mask_buf.cols != mask_w ||
      mask_buf.type() != CV_8UC1)
    mask_buf.create(mask_h, mask_w, CV_8UC1);
  mask_buf.setTo(0);
  const cv::Point *pts_ptr = shifted_buf.data();
  int npts = static_cast<int>(shifted_buf.size());
  cv::fillPoly(mask_buf, &pts_ptr, &npts, 1, cv::Scalar(1));

  cv::Mat roi =
      pred_map(cv::Rect(xmin, ymin, xmax - xmin + 1, ymax - ymin + 1));
  return static_cast<float>(cv::mean(roi, mask_buf)[0]);
}

std::vector<cv::Point> unclip(const std::vector<cv::Point> &polygon,
                              float unclip_ratio) {
  double area = cv::contourArea(polygon);
  double perimeter = cv::arcLength(polygon, true);
  if (perimeter == 0)
    return polygon;

  double distance = area * unclip_ratio / perimeter;

  ClipperLib::Path subj;
  subj.reserve(polygon.size());
  for (const auto &pt : polygon)
    subj.push_back(ClipperLib::IntPoint(pt.x, pt.y));

  ClipperLib::ClipperOffset co;
  co.AddPath(subj, ClipperLib::jtSquare, ClipperLib::etClosedPolygon);

  ClipperLib::Paths solution;
  co.Execute(solution, distance);

  if (solution.empty())
    return polygon;

  // Fast path: usually only one solution polygon
  const auto &best_path = (solution.size() == 1)
    ? solution[0]
    : *std::ranges::max_element(solution, {}, [](const ClipperLib::Path &p) {
        return ClipperLib::Area(p);
      });

  std::vector<cv::Point> result;
  result.reserve(best_path.size());
  for (const auto &p : best_path)
    result.push_back(cv::Point(static_cast<int>(p.X), static_cast<int>(p.Y)));
  return result;
}

Box get_mini_boxes(const std::vector<cv::Point> &contour, float &min_side) {
  cv::RotatedRect rect = cv::minAreaRect(contour);
  min_side = std::min(rect.size.width, rect.size.height);

  cv::Point2f pts[4];
  rect.points(pts);

  // Sort 4 points by x (insertion sort on 4 elements - no alloc)
  for (int i = 1; i < 4; i++) {
    cv::Point2f key = pts[i];
    int j = i - 1;
    while (j >= 0 && pts[j].x > key.x) { pts[j+1] = pts[j]; j--; }
    pts[j+1] = key;
  }

  // Left pair: pts[0], pts[1]. Right pair: pts[2], pts[3].
  int tl = (pts[0].y < pts[1].y) ? 0 : 1;
  int bl = 1 - tl;
  int tr = (pts[2].y < pts[3].y) ? 2 : 3;
  int br = 5 - tr; // 2+3=5

  Box box;
  box[0] = {static_cast<int>(std::round(pts[tl].x)), static_cast<int>(std::round(pts[tl].y))};
  box[1] = {static_cast<int>(std::round(pts[tr].x)), static_cast<int>(std::round(pts[tr].y))};
  box[2] = {static_cast<int>(std::round(pts[br].x)), static_cast<int>(std::round(pts[br].y))};
  box[3] = {static_cast<int>(std::round(pts[bl].x)), static_cast<int>(std::round(pts[bl].y))};
  return box;
}

std::vector<Box> extract_boxes_from_bitmap(
    const cv::Mat &pred_map, cv::Mat &bitmap,
    int orig_h, int orig_w, int resize_h, int resize_w,
    float det_db_box_thresh, float det_db_unclip_ratio,
    float min_box_side, float min_unclipped_side,
    std::vector<cv::Point> &shifted_buf, cv::Mat &mask_buf,
    std::vector<std::vector<cv::Point>> &contours_buf,
    std::vector<cv::Vec4i> &hierarchy_buf) {

  float ratio_h = static_cast<float>(resize_h) / orig_h;
  float ratio_w = static_cast<float>(resize_w) / orig_w;

  std::vector<Box> boxes;
  boxes.reserve(256);
  contours_buf.clear();
  hierarchy_buf.clear();
  cv::findContours(bitmap, contours_buf, hierarchy_buf, cv::RETR_LIST,
                   cv::CHAIN_APPROX_SIMPLE);

  static constexpr int kMaxCandidates = 1000;
  int num_contours = std::min(static_cast<int>(contours_buf.size()), kMaxCandidates);
  if (static_cast<int>(contours_buf.size()) > kMaxCandidates) {
    std::cerr << "[Det] WARNING: truncated to 1000 contours (found "
              << contours_buf.size() << ")\n";
  }

  for (int i = 0; i < num_contours; i++) {
    if (contours_buf[i].size() <= 2)
      continue;

    cv::Rect br = cv::boundingRect(contours_buf[i]);
    if (br.width < 3 || br.height < 3)
      continue;

    float score = box_score_fast(pred_map, contours_buf[i], shifted_buf, mask_buf);
    if (score < det_db_box_thresh)
      continue;

    float ssid = 0;
    (void)get_mini_boxes(contours_buf[i], ssid);
    if (ssid < min_box_side)
      continue;

    auto unclipped = unclip(contours_buf[i], det_db_unclip_ratio);
    if (unclipped.size() < 3)
      continue;

    float ssid2 = 0;
    auto box = get_mini_boxes(unclipped, ssid2);
    if (ssid2 < min_unclipped_side)
      continue;

    for (int k = 0; k < 4; ++k) {
      box[k][0] = std::clamp(static_cast<int>(std::round(box[k][0] / ratio_w)), 0, orig_w - 1);
      box[k][1] = std::clamp(static_cast<int>(std::round(box[k][1] / ratio_h)), 0, orig_h - 1);
    }

    int rw = static_cast<int>(std::sqrt(((box[0][0] - box[1][0]) * (box[0][0] - box[1][0])) +
                                        ((box[0][1] - box[1][1]) * (box[0][1] - box[1][1]))));
    int rh = static_cast<int>(std::sqrt(((box[0][0] - box[3][0]) * (box[0][0] - box[3][0])) +
                                        ((box[0][1] - box[3][1]) * (box[0][1] - box[3][1]))));
    if (rw <= 3 || rh <= 3)
      continue;

    boxes.push_back(box);
  }

  return boxes;
}

} // namespace turbo_ocr::detection
