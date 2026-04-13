#pragma once

#include <format>
#include <iostream>
#include <memory>
#include <stdexcept>
#include <vector>

#include "turbo_ocr/pipeline/ocr_pipeline.h"
#include "turbo_ocr/pipeline/pipeline_pool.h"
#include "turbo_ocr/common/cuda_check.h"
#include "turbo_ocr/common/errors.h"

namespace turbo_ocr::pipeline {

/// OcrPipeline + its dedicated CUDA stream, managed as a single poolable unit.
struct GpuPipelineEntry {
  std::unique_ptr<OcrPipeline> pipeline;
  cudaStream_t stream = nullptr;

  GpuPipelineEntry() = default;

  GpuPipelineEntry(std::unique_ptr<OcrPipeline> p, cudaStream_t s)
      : pipeline(std::move(p)), stream(s) {}

  ~GpuPipelineEntry() noexcept {
    if (stream)
      cudaStreamDestroy(stream);
  }

  // Move-only
  GpuPipelineEntry(GpuPipelineEntry &&o) noexcept
      : pipeline(std::move(o.pipeline)), stream(o.stream) {
    o.stream = nullptr;
  }
  GpuPipelineEntry &operator=(GpuPipelineEntry &&o) noexcept {
    if (this != &o) {
      if (stream) cudaStreamDestroy(stream);
      pipeline = std::move(o.pipeline);
      stream = o.stream;
      o.stream = nullptr;
    }
    return *this;
  }
  GpuPipelineEntry(const GpuPipelineEntry &) = delete;
  GpuPipelineEntry &operator=(const GpuPipelineEntry &) = delete;
};

/// Convenience alias — a PipelinePool of GpuPipelineEntry.
using GpuPipelinePool = PipelinePool<GpuPipelineEntry>;

/// Factory: create, init, warmup GPU pipelines and return a pool.
/// `layout_model` is an optional TRT engine path — pass "" to disable the
/// layout stage entirely (zero added cost at runtime).
[[nodiscard]] inline std::unique_ptr<GpuPipelinePool> make_gpu_pipeline_pool(
    int pool_size, const std::string &det_model, const std::string &rec_model,
    const std::string &rec_dict, const std::string &cls_model = "",
    const std::string &layout_model = "") {

  if (pool_size <= 0) [[unlikely]]
    throw std::invalid_argument(
        std::format("[Pool] Invalid pool_size={}, must be > 0", pool_size));

  std::vector<std::unique_ptr<GpuPipelineEntry>> entries;
  for (int i = 0; i < pool_size; ++i) {
    auto pipeline = std::make_unique<OcrPipeline>();
    if (!pipeline->init(det_model, rec_model, rec_dict, cls_model)) {
      std::cerr << std::format("[Pool] Failed to init GPU pipeline {} of {}", i, pool_size) << '\n';
      continue;
    }
    if (!layout_model.empty()) {
      if (!pipeline->load_layout_model(layout_model)) {
        // Fail hard: mixing layout-on / layout-off pipelines in the same
        // pool would make response shape non-deterministic depending on
        // which handle the request happened to acquire.
        throw turbo_ocr::ModelLoadError(std::format(
            "[Pool] Failed to load layout model for pipeline {} of {}",
            i, pool_size));
      }
    }
    cudaStream_t stream;
    CUDA_CHECK(cudaStreamCreateWithFlags(&stream, cudaStreamNonBlocking));
    entries.push_back(std::make_unique<GpuPipelineEntry>(std::move(pipeline), stream));
  }

  if (entries.empty()) [[unlikely]]
    throw turbo_ocr::ModelLoadError(
        std::format("[Pool] All {} GPU pipelines failed to initialize", pool_size));

  std::cout << std::format("Warming up {} pipelines...", entries.size()) << '\n';
  for (auto &e : entries) {
    e->pipeline->warmup_gpu(e->stream);
  }
  std::cout << "Pipeline warmup complete." << '\n';

  return std::make_unique<GpuPipelinePool>(std::move(entries));
}

} // namespace turbo_ocr::pipeline
