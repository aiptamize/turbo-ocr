#pragma once

#include "turbo_ocr/common/types.h"
#include <opencv2/core/mat.hpp>
#include <string>
#include <vector>

namespace turbo_ocr::pipeline {

/// Abstract interface for OCR pipelines (GPU and CPU).
/// GPU-specific overloads (with cudaStream_t, GpuImage, batch) remain
/// as non-virtual extensions on OcrPipeline.
class IOcrPipeline {
public:
  virtual ~IOcrPipeline() = default;

  [[nodiscard]] virtual bool init(const std::string &det_model,
                                  const std::string &rec_model,
                                  const std::string &rec_dict,
                                  const std::string &cls_model = "") = 0;

  virtual void warmup() = 0;

  [[nodiscard]] virtual std::vector<OCRResultItem> run(const cv::Mat &img) = 0;
};

} // namespace turbo_ocr::pipeline
