#pragma once

#include <onnxruntime_cxx_api.h>
#include <string>
#include <vector>

// CPU inference engine using ONNX Runtime.
// Loads ONNX models and runs them on CPU with all optimizations enabled.
// Drop-in replacement for TrtEngine when no GPU is available.

namespace turbo_ocr::engine {

/// ONNX Runtime CPU inference engine (drop-in replacement for TrtEngine).
class CpuEngine {
public:
  /// Construct with path to an ONNX model file (.onnx).
  explicit CpuEngine(const std::string &model_path);
  ~CpuEngine() noexcept = default;

  /// Load the ONNX model and create an inference session.
  [[nodiscard]] bool load();

  // Run inference. Input is a flat float buffer with given shape.
  // Returns output as a flat float vector + output shape.
  struct InferResult {
    std::vector<float> data;
    std::vector<int64_t> shape;

    [[nodiscard]] bool empty() const noexcept { return data.empty(); }
  };

  [[nodiscard]] InferResult infer(const float *input_data,
                                  const std::vector<int64_t> &input_shape);

  // Probe output dims for a given input shape (runs a dummy inference)
  void probe_output_dims(const std::vector<int64_t> &input_shape,
                         int &out_dim1, int &out_dim2);

private:
  std::string model_path_;
  Ort::Env env_{ORT_LOGGING_LEVEL_WARNING, "CpuEngine"};
  std::unique_ptr<Ort::Session> session_;
  Ort::SessionOptions session_options_;

  std::string input_name_;
  std::string output_name_;
  Ort::AllocatorWithDefaultOptions allocator_;
};

} // namespace turbo_ocr::engine
