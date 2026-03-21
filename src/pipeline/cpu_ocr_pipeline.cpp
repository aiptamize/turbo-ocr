#include "turbo_ocr/pipeline/cpu_ocr_pipeline.h"

#include <algorithm>
#include <format>
#include <ranges>

using namespace turbo_ocr::pipeline;
using turbo_ocr::Box;
using turbo_ocr::OCRResultItem;
using turbo_ocr::sorted_boxes;
using turbo_ocr::is_vertical_box;
using turbo_ocr::detection::CpuPaddleDet;
using turbo_ocr::classification::CpuPaddleCls;
using turbo_ocr::recognition::CpuPaddleRec;

CpuOcrPipeline::CpuOcrPipeline() {
  det_ = std::make_unique<CpuPaddleDet>();
  rec_ = std::make_unique<CpuPaddleRec>();
}

bool CpuOcrPipeline::init(const std::string &det_model,
                           const std::string &rec_model,
                           const std::string &rec_dict,
                           const std::string &cls_model) {
  if (!det_->load_model(det_model))
    return false;
  if (!rec_->load_model(rec_model))
    return false;
  if (!rec_->load_dict(rec_dict))
    return false;

  if (!cls_model.empty()) {
    cls_ = std::make_unique<CpuPaddleCls>();
    if (!cls_->load_model(cls_model)) {
      std::cerr << std::format("[Pipeline] Failed to load CPU cls model: {}", cls_model) << '\n';
      return false;
    }
    use_cls_ = true;
    std::cout << "[Pipeline] Angle classifier enabled (CPU)" << '\n';
  }

  return true;
}

void CpuOcrPipeline::warmup() {
  cv::Mat dummy(100, 100, CV_8UC3, cv::Scalar(255, 255, 255));
  cv::rectangle(dummy, cv::Point(10, 30), cv::Point(90, 70),
                cv::Scalar(0, 0, 0), 2);
  auto results = run(dummy);
  (void)results;
}

std::vector<OCRResultItem> CpuOcrPipeline::run(const cv::Mat &img) {
  // Detection
  auto boxes = det_->run(img);

  // Sort boxes top-to-bottom, left-to-right
  sorted_boxes(boxes);

  // Optional angle classification -- only classify if any box looks vertical
  if (use_cls_ && cls_ && std::ranges::any_of(boxes, is_vertical_box)) {
    cls_->run(img, boxes);
  }

  // Recognition
  auto rec_results = rec_->run(img, boxes);

  // Combine (filter by drop_score)
  constexpr float kDropScore = turbo_ocr::kDropScore;
  std::vector<OCRResultItem> final_results;
  final_results.reserve(boxes.size());
  for (size_t i = 0; i < boxes.size(); ++i) {
    if (i < rec_results.size()) {
      if (rec_results[i].second < kDropScore)
        continue;
      if (rec_results[i].first.empty())
        continue;
      final_results.push_back({
        .text = std::move(rec_results[i].first),
        .confidence = rec_results[i].second,
        .box = boxes[i],
      });
    }
  }

  return final_results;
}
