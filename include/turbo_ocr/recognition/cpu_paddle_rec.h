#pragma once

#include <memory>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

#include "turbo_ocr/engine/cpu_engine.h"
#include "turbo_ocr/common/box.h"

namespace turbo_ocr::recognition {

/// CPU text recognizer using ONNX Runtime (CRNN + CTC decoding).
class CpuPaddleRec {
public:
  CpuPaddleRec();
  ~CpuPaddleRec() noexcept = default;

  /// Load an ONNX recognition model and probe output dimensions.
  [[nodiscard]] bool load_model(const std::string &model_path);
  /// Load the character dictionary for CTC decoding.
  [[nodiscard]] bool load_dict(const std::string &dict_path);

  // Run recognition on image crops defined by boxes.
  // img is the original full image (BGR, uint8).
  [[nodiscard]] std::vector<std::pair<std::string, float>>
  run(const cv::Mat &img, const std::vector<Box> &boxes);

private:
  std::vector<std::string> label_list_;
  int rec_image_h_ = 48;
  int rec_image_w_ = 320;

  std::unique_ptr<engine::CpuEngine> engine_;

  static constexpr int kMaxRecWidth = 4000;

  int actual_seq_len_ = 600;
  int actual_num_classes_ = 20000;

  // Preprocess a crop into NCHW float buffer
  void preprocess_crop(const cv::Mat &crop, int target_w,
                       std::vector<float> &buffer);
};

} // namespace turbo_ocr::recognition
