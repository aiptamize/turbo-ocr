#include "turbo_ocr/engine/cpu_engine.h"

#include <cstring>
#include <format>
#include <iostream>
#include <numeric>

// CoreML support on macOS (Apple Neural Engine + GPU acceleration)
#ifdef __APPLE__
#include <coreml_provider_factory.h>
#endif

using namespace turbo_ocr::engine;

CpuEngine::CpuEngine(const std::string &model_path) : model_path_(model_path) {
  // CPU optimizations — balance threads per inference vs concurrency
  if (const char *env = std::getenv("ORT_NUM_THREADS"))
    session_options_.SetIntraOpNumThreads(std::atoi(env));
  else
    session_options_.SetIntraOpNumThreads(4);
  session_options_.SetInterOpNumThreads(1);
  session_options_.SetGraphOptimizationLevel(
      GraphOptimizationLevel::ORT_ENABLE_ALL);
  session_options_.EnableCpuMemArena();
  session_options_.SetExecutionMode(ExecutionMode::ORT_SEQUENTIAL);

  // On macOS: use CoreML for Neural Engine + GPU acceleration
  // Supported ops run on ANE/GPU, unsupported fall back to CPU automatically
#ifdef __APPLE__
  bool use_coreml = true;
  if (const char *env = std::getenv("DISABLE_COREML"))
    use_coreml = (std::strcmp(env, "1") != 0);
  if (use_coreml) {
    // COREML_FLAG_USE_CPU_AND_GPU = 0x020 routes to GPU + Neural Engine
    // COREML_FLAG_ONLY_ENABLE_DEVICE_WITH_ANE = 0x004 requires ANE
    uint32_t coreml_flags = 0;
    if (const char *env = std::getenv("COREML_FLAGS"))
      coreml_flags = std::strtoul(env, nullptr, 0);
    else
      coreml_flags = 0x020; // CPU + GPU (includes Neural Engine when available)
    OrtSessionOptionsAppendExecutionProvider_CoreML(session_options_, coreml_flags);
    std::cout << "[CpuEngine] CoreML enabled (flags=0x" << std::hex
              << coreml_flags << std::dec << ")\n";
  }
#endif
}

bool CpuEngine::load() {
  try {
    session_ =
        std::make_unique<Ort::Session>(env_, model_path_.c_str(), session_options_);
  } catch (const Ort::Exception &e) {
    std::cerr << std::format("[CpuEngine] Failed to load ONNX model: {} - {}", model_path_, e.what()) << '\n';
    return false;
  }

  // Get input/output names
  auto input_name = session_->GetInputNameAllocated(0, allocator_);
  input_name_ = input_name.get();

  auto output_name = session_->GetOutputNameAllocated(0, allocator_);
  output_name_ = output_name.get();

  std::cout << std::format("[CpuEngine] Loaded: {} (input={}, output={})",
                          model_path_, input_name_, output_name_) << '\n';
  return true;
}

CpuEngine::InferResult
CpuEngine::infer(const float *input_data,
                 const std::vector<int64_t> &input_shape) {
  InferResult result;
  if (!session_)
    return result;

  int64_t input_count =
      std::accumulate(input_shape.begin(), input_shape.end(), int64_t{1},
                      std::multiplies<int64_t>());

  // Cached MemoryInfo — avoid recreating every call
  static const auto memory_info =
      Ort::MemoryInfo::CreateCpu(OrtArenaAllocator, OrtMemTypeDefault);

  Ort::Value input_tensor = Ort::Value::CreateTensor<float>(
      memory_info, const_cast<float *>(input_data), input_count,
      input_shape.data(), input_shape.size());

  const char *input_names[] = {input_name_.c_str()};
  const char *output_names[] = {output_name_.c_str()};

  auto output_tensors = session_->Run(Ort::RunOptions{nullptr}, input_names,
                                       &input_tensor, 1, output_names, 1);

  auto &output_tensor = output_tensors.front();
  auto type_info = output_tensor.GetTensorTypeAndShapeInfo();
  result.shape = type_info.GetShape();

  int64_t output_count = type_info.GetElementCount();
  const float *output_data = output_tensor.GetTensorData<float>();
  result.data.assign(output_data, output_data + output_count);

  return result;
}

void CpuEngine::probe_output_dims(const std::vector<int64_t> &input_shape,
                                   int &out_dim1, int &out_dim2) {
  if (!session_)
    return;

  int64_t input_count =
      std::accumulate(input_shape.begin(), input_shape.end(), int64_t{1},
                      std::multiplies<int64_t>());

  std::vector<float> dummy(input_count, 0.0f);
  auto result = infer(dummy.data(), input_shape);

  if (result.shape.size() >= 3) {
    out_dim1 = static_cast<int>(result.shape[1]);
    out_dim2 = static_cast<int>(result.shape[2]);
  }
}
