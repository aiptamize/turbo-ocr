#include "turbo_ocr/detection/cpu_paddle_det.h"
#include "turbo_ocr/detection/det_postprocess.h"

#include <algorithm>
#include <cmath>
#include <opencv2/imgproc.hpp>

using namespace turbo_ocr;
using namespace turbo_ocr::detection;
using turbo_ocr::engine::CpuEngine;

bool CpuPaddleDet::load_model(const std::string &model_path) {
  engine_ = std::make_unique<CpuEngine>(model_path);
  return engine_->load();
}

std::vector<Box> CpuPaddleDet::run(const cv::Mat &img) {
  int h = img.rows;
  int w = img.cols;
  float ratio = 1.0f;
  if (std::max(h, w) > kMaxSideLen) {
    ratio = (h > w) ? static_cast<float>(kMaxSideLen) / h
                    : static_cast<float>(kMaxSideLen) / w;
  }
  int resize_h = std::max(static_cast<int>(round(h * ratio / 32.0) * 32), 32);
  int resize_w = std::max(static_cast<int>(round(w * ratio / 32.0) * 32), 32);

  // CPU preprocessing: resize
  cv::Mat resized;
  cv::resize(img, resized, cv::Size(resize_w, resize_h));

  // Convert to float and normalize (ImageNet mean/std)
  cv::Mat float_img;
  resized.convertTo(float_img, CV_32F, 1.0 / 255.0);

  // Normalize per channel: (pixel - mean) / std
  // OpenCV BGR order: B=0.406/0.225, G=0.456/0.224, R=0.485/0.229
  cv::Mat channels[3];
  cv::split(float_img, channels);
  channels[0] = (channels[0] - 0.406f) / 0.225f; // B
  channels[1] = (channels[1] - 0.456f) / 0.224f; // G
  channels[2] = (channels[2] - 0.485f) / 0.229f; // R

  // Convert to NCHW layout: [1, 3, H, W]
  // ORT expects contiguous NCHW float buffer
  int total = 3 * resize_h * resize_w;
  input_data_buf_.resize(total);

  // R channel first (PaddleOCR uses RGB order)
  int plane_size = resize_h * resize_w;
  std::memcpy(input_data_buf_.data(), channels[2].data,
              plane_size * sizeof(float));
  std::memcpy(input_data_buf_.data() + plane_size, channels[1].data,
              plane_size * sizeof(float));
  std::memcpy(input_data_buf_.data() + 2 * plane_size, channels[0].data,
              plane_size * sizeof(float));

  input_shape_buf_ = {1, 3, static_cast<int64_t>(resize_h), static_cast<int64_t>(resize_w)};
  auto result = engine_->infer(input_data_buf_.data(), input_shape_buf_);

  if (result.data.empty())
    return {};

  // Output shape: [1, 1, resize_h, resize_w]
  cv::Mat pred_map(resize_h, resize_w, CV_32F, result.data.data());

  // Threshold to bitmap
  cv::Mat bitmap;
  cv::threshold(pred_map, bitmap, kDetDbThresh, 255, cv::THRESH_BINARY);
  bitmap.convertTo(bitmap, CV_8UC1);

  // Find contours and extract boxes
  return extract_boxes_from_bitmap(
      pred_map, bitmap, h, w, resize_h, resize_w,
      kDetDbBoxThresh, kDetDbUnclipRatio, kMinBoxSide, kMinUnclippedSide,
      shifted_buf_, mask_buf_, contours_buf_, hierarchy_buf_);
}
