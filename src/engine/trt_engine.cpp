#include "turbo_ocr/engine/trt_engine.h"

#include <format>
#include <fstream>
#include <vector>

using namespace turbo_ocr::engine;

TrtEngine::TrtEngine(const std::string &model_path) : model_path_(model_path) {}

bool TrtEngine::load() {
  std::ifstream file(model_path_, std::ios::binary);
  if (!file.good()) [[unlikely]] {
    std::cerr << std::format("[TRT] Error loading engine file: {}", model_path_) << '\n';
    return false;
  }

  file.seekg(0, file.end);
  auto pos = file.tellg();
  if (pos < 0) [[unlikely]] {
    std::cerr << std::format("[TRT] Error reading engine file size: {}", model_path_) << '\n';
    return false;
  }
  auto size = static_cast<size_t>(pos);
  file.seekg(0, file.beg);

  std::vector<char> trtModelStream(size);
  file.read(trtModelStream.data(), size);

  runtime_.reset(nvinfer1::createInferRuntime(logger_));
  if (!runtime_) [[unlikely]] {
    std::cerr << std::format("[TRT] Failed to create runtime for: {}", model_path_) << '\n';
    return false;
  }

  engine_.reset(runtime_->deserializeCudaEngine(trtModelStream.data(), size));
  if (!engine_) [[unlikely]] {
    std::cerr << std::format("[TRT] Failed to deserialize engine: {}", model_path_) << '\n';
    return false;
  }

  context_.reset(engine_->createExecutionContext());
  if (!context_) [[unlikely]] {
    std::cerr << std::format("[TRT] Failed to create execution context: {}", model_path_) << '\n';
    return false;
  }

  auto nbIO = engine_->getNbIOTensors();
  for (int i = 0; i < nbIO; ++i) {
    const char *name = engine_->getIOTensorName(i);
    if (engine_->getTensorIOMode(name) == nvinfer1::TensorIOMode::kINPUT)
      input_name_ = name;
    else
      output_name_ = name;
  }

  return true;
}

void TrtEngine::bind_io(void *input, void *output) {
  if (!context_) [[unlikely]]
    return;
  bound_input_ = input;
  bound_output_ = output;
  context_->setTensorAddress(input_name_.c_str(), input);
  context_->setTensorAddress(output_name_.c_str(), output);
}

static constexpr bool dims_equal(const nvinfer1::Dims &a, const nvinfer1::Dims &b) {
  if (a.nbDims != b.nbDims) return false;
  for (int i = 0; i < a.nbDims; ++i)
    if (a.d[i] != b.d[i]) return false;
  return true;
}

bool TrtEngine::infer_dynamic(const nvinfer1::Dims &input_dims,
                              cudaStream_t stream) {
  if (!context_) [[unlikely]]
    return false;
  if (!dims_equal(input_dims, last_input_dims_)) {
    // TRT 10.14: all profiles share the same base tensor names.
    // setInputShape applies to whichever profile is currently active.
    if (!context_->setInputShape(input_name_.c_str(), input_dims)) [[unlikely]] {
      std::cerr << std::format("[TRT] setInputShape FAILED for input=({},{},{},{}) profile={} on {}",
                               input_dims.d[0], input_dims.d[1], input_dims.d[2], input_dims.d[3],
                               current_profile_, model_path_) << '\n';
      return false;
    }
    last_input_dims_ = input_dims;
  }
  return context_->enqueueV3(stream);
}

void TrtEngine::select_profile(int profile_idx, cudaStream_t stream) {
  if (profile_idx == current_profile_)
    return;

  if (profile_idx < 0 || profile_idx >= engine_->getNbOptimizationProfiles()) {
    std::cerr << std::format("[TRT] Invalid profile index {} (engine has {})",
                             profile_idx, engine_->getNbOptimizationProfiles())
              << '\n';
    return;
  }

  context_->setOptimizationProfileAsync(profile_idx, stream);
  current_profile_ = profile_idx;
  // Invalidate cached dims — new profile requires setInputShape again
  last_input_dims_ = {};

  // Re-bind I/O addresses for the new profile (TRT clears them on profile switch)
  if (bound_input_) context_->setTensorAddress(input_name_.c_str(), bound_input_);
  if (bound_output_) context_->setTensorAddress(output_name_.c_str(), bound_output_);
}

int TrtEngine::num_profiles() const noexcept {
  return engine_ ? engine_->getNbOptimizationProfiles() : 1;
}

nvinfer1::Dims TrtEngine::get_output_dims() const noexcept {
  if (!context_) [[unlikely]]
    return {};
  return context_->getTensorShape(output_name_.c_str());
}

void TrtEngine::probe_output_dims(const nvinfer1::Dims &input_dims,
                                   int &out_seq_len, int &out_num_classes) {
  if (!context_) [[unlikely]]
    return;
  if (!context_->setInputShape(input_name_.c_str(), input_dims)) {
    std::cerr << "[TRT] probe_output_dims: setInputShape failed\n";
    return;
  }
  nvinfer1::Dims od = context_->getTensorShape(output_name_.c_str());
  if (od.nbDims >= 3) {
    out_seq_len = od.d[1];
    out_num_classes = od.d[2];
  }
  // Invalidate cached dims so the next infer_dynamic() call will re-set the
  // actual inference shape (probe may have used a different shape).
  last_input_dims_ = {};
}
