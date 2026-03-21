#pragma once

#include <opencv2/core.hpp>
#include <opencv2/imgproc.hpp>

#include "turbo_ocr/common/box.h"

namespace turbo_ocr::recognition {

// Perspective-crop an image region defined by a quadrilateral Box.
// Handles vertical text rotation (crop_h >= crop_w * kVerticalAspectRatio).
[[nodiscard]] cv::Mat get_rotate_crop_image(const cv::Mat &img, const Box &box);

} // namespace turbo_ocr::recognition
