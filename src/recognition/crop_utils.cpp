#include "turbo_ocr/recognition/crop_utils.h"
#include "turbo_ocr/common/perspective.h"

namespace turbo_ocr::recognition {

cv::Mat get_rotate_crop_image(const cv::Mat &img, const Box &box) {
  auto cg = turbo_ocr::compute_crop_geometry(box);

  auto f = [](int v) { return static_cast<float>(v); };
  cv::Point2f src_pts[4] = {
      {cg.src_pts[0], cg.src_pts[1]},
      {cg.src_pts[2], cg.src_pts[3]},
      {cg.src_pts[4], cg.src_pts[5]},
      {cg.src_pts[6], cg.src_pts[7]},
  };
  cv::Point2f dst_pts[4] = {{0, 0},
                             {f(cg.dst_w), 0},
                             {f(cg.dst_w), f(cg.dst_h)},
                             {0, f(cg.dst_h)}};

  cv::Mat M = cv::getPerspectiveTransform(src_pts, dst_pts);
  cv::Mat cropped;
  cv::warpPerspective(img, cropped, M, cv::Size(cg.dst_w, cg.dst_h),
                       cv::INTER_LINEAR, cv::BORDER_REPLICATE);
  return cropped;
}

} // namespace turbo_ocr::recognition
