#pragma once

#include <memory>
#include <vector>

#include "turbo_ocr/classification/cpu_paddle_cls.h"
#include "turbo_ocr/common/types.h"
#include "turbo_ocr/detection/cpu_paddle_det.h"
#include "turbo_ocr/pipeline/i_ocr_pipeline.h"
#include "turbo_ocr/recognition/cpu_paddle_rec.h"

namespace turbo_ocr::pipeline {

class CpuOcrPipeline : public IOcrPipeline {
public:
  CpuOcrPipeline();
  ~CpuOcrPipeline() noexcept override = default;

  [[nodiscard]] bool init(const std::string &det_model, const std::string &rec_model,
                          const std::string &rec_dict, const std::string &cls_model = "") override;

  void warmup() override;

  [[nodiscard]] std::vector<OCRResultItem> run(const cv::Mat &img) override;

private:
  std::unique_ptr<detection::CpuPaddleDet> det_;
  std::unique_ptr<classification::CpuPaddleCls> cls_;
  std::unique_ptr<recognition::CpuPaddleRec> rec_;

  bool use_cls_ = false;
};

} // namespace turbo_ocr::pipeline
