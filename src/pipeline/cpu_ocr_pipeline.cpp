#include "turbo_ocr/pipeline/cpu_ocr_pipeline.h"

#include <algorithm>
#include <format>
#include <iostream>
#include <ranges>

#include <opencv2/imgproc.hpp>

#include "turbo_ocr/common/box.h"
#include "turbo_ocr/common/serialization.h"
#include "turbo_ocr/layout/reading_order.h"

namespace turbo_ocr::pipeline {

using ::turbo_ocr::Box;
using ::turbo_ocr::OCRResultItem;
using ::turbo_ocr::sorted_boxes;
using ::turbo_ocr::is_vertical_box;

CpuOcrPipeline::CpuOcrPipeline() {
  det_ = std::make_unique<detection::CpuPaddleDet>();
  rec_ = std::make_unique<recognition::CpuPaddleRec>();
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
    cls_ = std::make_unique<classification::CpuPaddleCls>();
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

bool CpuOcrPipeline::load_layout_model(const std::string &onnx_path) {
  layout_ = std::make_unique<layout::CpuPaddleLayout>();
  if (!layout_->load_model(onnx_path)) {
    layout_.reset();
    return false;
  }
  return true;
}

OcrPipelineResult CpuOcrPipeline::run_with_layout(const cv::Mat &img,
                                                    bool want_layout,
                                                    bool want_reading_order) {
  OcrPipelineResult out;
  out.results = run(img);
  if (want_layout && layout_)
    out.layout = layout_->run(img);

  // Reading-order over layout regions, with synthetic XY-cut entries
  // for orphan results (results whose centroid falls outside every
  // layout box). assign_layout_ids() resolves layout_id by centroid
  // containment so the helper can synthesise correctly. Idempotent —
  // serialization re-runs it but the second call is a no-op.
  if (want_reading_order && !out.layout.empty()) {
    turbo_ocr::assign_layout_ids(out.results, out.layout);
    out.reading_order =
        layout::assign_reading_order_for_results(out.results, out.layout);
  }

  return out;
}

} // namespace turbo_ocr::pipeline
