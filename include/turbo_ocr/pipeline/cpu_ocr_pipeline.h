#pragma once

#include <memory>
#include <vector>

#include "turbo_ocr/classification/cpu_paddle_cls.h"
#include "turbo_ocr/common/types.h"
#include "turbo_ocr/detection/cpu_paddle_det.h"
#include "turbo_ocr/layout/cpu_paddle_layout.h"
#include "turbo_ocr/layout/layout_types.h"
#include "turbo_ocr/pipeline/i_ocr_pipeline.h"
#include "turbo_ocr/pipeline/pipeline_result.h"
#include "turbo_ocr/recognition/cpu_paddle_rec.h"

namespace turbo_ocr::pipeline {

class CpuOcrPipeline : public IOcrPipeline {
public:
  CpuOcrPipeline();
  ~CpuOcrPipeline() noexcept override = default;

  [[nodiscard]] bool init(const std::string &det_model, const std::string &rec_model,
                          const std::string &rec_dict, const std::string &cls_model = "") override;

  [[nodiscard]] bool load_layout_model(const std::string &onnx_path);

  void warmup() override;

  [[nodiscard]] std::vector<OCRResultItem> run(const cv::Mat &img) override;

  /// Run OCR + optional layout detection.
  [[nodiscard]] OcrPipelineResult run_with_layout(const cv::Mat &img,
                                                   bool want_layout = false);

private:
  std::unique_ptr<detection::CpuPaddleDet> det_;
  std::unique_ptr<classification::CpuPaddleCls> cls_;
  std::unique_ptr<recognition::CpuPaddleRec> rec_;
  std::unique_ptr<layout::CpuPaddleLayout> layout_;

  bool use_cls_ = false;
};

} // namespace turbo_ocr::pipeline
