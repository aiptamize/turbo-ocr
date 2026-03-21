#pragma once

#include <format>
#include <iostream>
#include <memory>
#include <string>

#include <NvInfer.h>

namespace turbo_ocr::engine {

class Logger : public nvinfer1::ILogger {
  void log(Severity severity, const char *msg) noexcept override {
    if (severity != Severity::kINFO) {
      std::cerr << std::format("[TRT] {}", msg) << '\n';
    }
  }
};

/// TensorRT inference engine wrapper (name-based API, TRT 10+).
class TrtEngine {
public:
  /// Construct with path to a serialized TensorRT engine file (.trt).
  explicit TrtEngine(const std::string &model_path);
  ~TrtEngine() noexcept = default;

  /// Deserialize and load the engine. Must be called before any inference.
  [[nodiscard]] bool load();

  void bind_io(void *input, void *output);

  // Infer with explicit full input dims (for dynamic shapes).
  // Skips setInputShape if dims haven't changed.
  [[nodiscard]] bool infer_dynamic(const nvinfer1::Dims &input_dims, cudaStream_t stream = 0);

  // Switch optimization profile. Calls setOptimizationProfileAsync if the
  // requested profile differs from the current one. Multi-profile engines
  // (e.g. rec with small/large profiles) use this for better kernel selection.
  void select_profile(int profile_idx, cudaStream_t stream = 0);

  // Returns the number of optimization profiles in the engine.
  [[nodiscard]] int num_profiles() const noexcept;

  [[nodiscard]] nvinfer1::Dims get_output_dims() const noexcept;

  void probe_output_dims(const nvinfer1::Dims &input_dims,
                         int &out_seq_len, int &out_num_classes);

private:
  std::string model_path_;
  Logger logger_;
  std::unique_ptr<nvinfer1::IRuntime> runtime_;
  std::unique_ptr<nvinfer1::ICudaEngine> engine_;

  std::string input_name_;
  std::string output_name_;
  std::unique_ptr<nvinfer1::IExecutionContext> context_;

  nvinfer1::Dims last_input_dims_{};
  int current_profile_ = 0;
  void *bound_input_ = nullptr;
  void *bound_output_ = nullptr;
};

} // namespace turbo_ocr::engine
