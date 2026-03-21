#include "turbo_ocr/classification/cpu_paddle_cls.h"
#include "turbo_ocr/recognition/crop_utils.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <opencv2/imgproc.hpp>

using namespace turbo_ocr::classification;
using turbo_ocr::engine::CpuEngine;
using turbo_ocr::Box;
using turbo_ocr::recognition::get_rotate_crop_image;

bool CpuPaddleCls::load_model(const std::string &model_path) {
  engine_ = std::make_unique<CpuEngine>(model_path);
  return engine_->load();
}

void CpuPaddleCls::run(const cv::Mat &img, std::vector<Box> &boxes) {
  if (boxes.empty())
    return;

  std::vector<float> input_buf;
  const std::vector<int64_t> input_shape = {1, 3, kClsImageH, kClsImageW};

  for (size_t i = 0; i < boxes.size(); ++i) {
    cv::Mat crop = get_rotate_crop_image(img, boxes[i]);

    // Resize to cls input size
    cv::Mat resized;
    cv::resize(crop, resized, cv::Size(kClsImageW, kClsImageH));

    // Normalize: /127.5 - 1.0
    cv::Mat float_img;
    resized.convertTo(float_img, CV_32F, 1.0 / 127.5, -1.0);

    // Convert BGR to RGB and NCHW
    int plane_size = kClsImageH * kClsImageW;
    input_buf.resize(3 * plane_size);

    cv::Mat channels[3];
    cv::split(float_img, channels);

    // RGB order
    std::memcpy(input_buf.data(), channels[2].data,
                plane_size * sizeof(float));
    std::memcpy(input_buf.data() + plane_size, channels[1].data,
                plane_size * sizeof(float));
    std::memcpy(input_buf.data() + 2 * plane_size, channels[0].data,
                plane_size * sizeof(float));

    auto result = engine_->infer(input_buf.data(), input_shape);

    if (result.data.size() >= 2) {
      float score_0 = result.data[0];
      float score_180 = result.data[1];

      if (score_180 > score_0 && score_180 > kClsThresh) {
        std::swap(boxes[i][0], boxes[i][2]);
        std::swap(boxes[i][1], boxes[i][3]);
      }
    }
  }
}
