#include "turbo_ocr/recognition/cpu_paddle_rec.h"
#include "turbo_ocr/recognition/crop_utils.h"
#include "turbo_ocr/recognition/ctc_decode.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <format>
#include <opencv2/imgproc.hpp>

using namespace turbo_ocr::recognition;
using turbo_ocr::engine::CpuEngine;
using turbo_ocr::Box;

CpuPaddleRec::CpuPaddleRec() { label_list_.push_back("blank"); }

bool CpuPaddleRec::load_model(const std::string &model_path) {
  engine_ = std::make_unique<CpuEngine>(model_path);
  if (!engine_->load())
    return false;

  // Probe output dims with a small input
  std::vector<int64_t> probe_shape = {1, 3, static_cast<int64_t>(rec_image_h_),
                                       static_cast<int64_t>(rec_image_w_)};
  engine_->probe_output_dims(probe_shape, actual_seq_len_, actual_num_classes_);
  std::cout << std::format("[CpuPaddleRec] Output dims: seq_len={} num_classes={}",
                          actual_seq_len_, actual_num_classes_) << '\n';
  return true;
}

bool CpuPaddleRec::load_dict(const std::string &dict_path) {
  return load_label_dict(dict_path, label_list_);
}

void CpuPaddleRec::preprocess_crop(const cv::Mat &crop, int target_w,
                                    std::vector<float> &buffer) {
  // Resize to (rec_image_h_, target_w)
  cv::Mat resized;
  cv::resize(crop, resized, cv::Size(target_w, rec_image_h_));

  // Convert to float, normalize to [-1, 1]: pixel/127.5 - 1.0
  cv::Mat float_img;
  resized.convertTo(float_img, CV_32F, 1.0 / 127.5, -1.0);

  // Convert BGR to RGB and to NCHW layout
  int plane_size = rec_image_h_ * target_w;
  buffer.resize(3 * plane_size);

  cv::Mat channels[3];
  cv::split(float_img, channels);

  // RGB order (R=channels[2], G=channels[1], B=channels[0])
  std::memcpy(buffer.data(), channels[2].data, plane_size * sizeof(float));
  std::memcpy(buffer.data() + plane_size, channels[1].data,
              plane_size * sizeof(float));
  std::memcpy(buffer.data() + 2 * plane_size, channels[0].data,
              plane_size * sizeof(float));
}

std::vector<std::pair<std::string, float>>
CpuPaddleRec::run(const cv::Mat &img, const std::vector<Box> &boxes) {
  std::vector<std::pair<std::string, float>> results;
  if (boxes.empty())
    return results;

  results.resize(boxes.size());

  // Process each box one at a time
  std::vector<float> input_buf;
  std::vector<int64_t> input_shape = {1, 3, static_cast<int64_t>(rec_image_h_), 0};

  for (size_t i = 0; i < boxes.size(); i++) {
    cv::Mat cropped = get_rotate_crop_image(img, boxes[i]);

    float ar =
        (cropped.rows > 0) ? static_cast<float>(cropped.cols) / cropped.rows : 0;
    int target_w =
        std::min(static_cast<int>(std::ceil(rec_image_h_ * ar)), kMaxRecWidth);
    target_w = std::max(target_w, rec_image_w_);

    preprocess_crop(cropped, target_w, input_buf);

    input_shape[3] = static_cast<int64_t>(target_w);
    auto result = engine_->infer(input_buf.data(), input_shape);

    if (result.shape.size() >= 3) {
      int seq_len = static_cast<int>(result.shape[1]);
      int num_classes = static_cast<int>(result.shape[2]);
      results[i] = ctc_greedy_decode_raw(result.data.data(), seq_len, num_classes, label_list_);
    }
  }

  return results;
}
