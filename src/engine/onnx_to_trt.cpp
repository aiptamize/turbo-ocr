#include "turbo_ocr/engine/onnx_to_trt.h"
#include "turbo_ocr/common/cuda_check.h"
#include "turbo_ocr/detection/det_config.h"
#include "turbo_ocr/server/env_utils.h"

#include <NvInfer.h>
#include <NvOnnxParser.h>

#include <cuda_runtime.h>

#include <algorithm>
#include <cerrno>
#include <chrono>
#include <cstdlib>
#include <cstring>
#include <filesystem>
#include <unistd.h>
#include <fstream>
#include <functional>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <system_error>

namespace fs = std::filesystem;

namespace turbo_ocr::engine {

using turbo_ocr::detection::read_det_max_side;
using turbo_ocr::detection::kDetMaxSideMin;

// TensorRT builder optimization level: 0..5 (TRT 10 range). Higher = better
// kernel selection at the cost of build time. Default 5 — same as before
// this knob existed. Operators on small instances or with strict cold-start
// budgets can drop to 3 (build ~3-5× faster, runtime regression typically
// <5%). Read once on first call; subsequent calls hit the function-static.
[[nodiscard]] static int read_trt_opt_level() {
  static const int v = server::env_int("TRT_OPT_LEVEL", 5, 0, 5);
  return v;
}

static class BuildLogger : public nvinfer1::ILogger {
public:
  void log(Severity severity, const char *msg) noexcept override {
    if (severity <= Severity::kWARNING) {
      // Concurrent TRT builds (e.g. det/rec/cls/layout in parallel during
      // warmup) can interleave on std::cerr. Build phase only — no impact on
      // request hot path.
      static std::mutex log_mu;
      std::lock_guard<std::mutex> lock(log_mu);
      std::cerr << "[TRT Build] " << msg << '\n';
    }
  }
} s_logger;

std::string get_engine_cache_dir() {
  // User override
  if (auto *env = std::getenv("TRT_ENGINE_CACHE"))
    return env;

  // Try ~/.cache/turbo-ocr/
  if (auto *home = std::getenv("HOME")) {
    auto dir = std::string(home) + "/.cache/turbo-ocr";
    fs::create_directories(dir);
    return dir;
  }

  // Fallback to /tmp
  auto dir = std::string("/tmp/turbo-ocr-engines");
  fs::create_directories(dir);
  return dir;
}

std::string get_cached_engine_path(const std::string &onnx_path,
                                   const std::string &type) {
  auto cache_dir = get_engine_cache_dir();

  // Build a cache key from: onnx file size + mtime + TRT version
  auto onnx_size = fs::file_size(onnx_path);
  auto onnx_mtime = fs::last_write_time(onnx_path).time_since_epoch().count();

  int trt_major = 0, trt_minor = 0, trt_patch = 0;
#ifdef NV_TENSORRT_MAJOR
  trt_major = NV_TENSORRT_MAJOR;
  trt_minor = NV_TENSORRT_MINOR;
  trt_patch = NV_TENSORRT_PATCH;
#endif

  // GPU compute capability (engines are GPU-architecture specific)
  int gpu_major = 0, gpu_minor = 0;
  CUDA_CHECK(cudaDeviceGetAttribute(&gpu_major, cudaDevAttrComputeCapabilityMajor, 0));
  CUDA_CHECK(cudaDeviceGetAttribute(&gpu_minor, cudaDevAttrComputeCapabilityMinor, 0));

  // CUDA driver + runtime versions: a host driver upgrade can invalidate
  // engines cached under the previous driver (cryptic CUDA errors at
  // deserialize time). TRT version covers majors but not driver minor
  // patches, so include both integers (e.g. 13020 for CUDA 13.2) directly.
  int cuda_driver = 0, cuda_runtime = 0;
  CUDA_CHECK(cudaDriverGetVersion(&cuda_driver));
  CUDA_CHECK(cudaRuntimeGetVersion(&cuda_runtime));

  // Cache key includes: onnx identity, TRT version, GPU arch, and profile version.
  // Bump kProfileVersion when optimization profiles change for det/rec/cls.
  // Adding a NEW model type (e.g. "layout") does NOT require a bump because
  // the cache key includes `type` — new types live in their own hash space.
  // 2026-04-26: bumped because the det profile MAX now tracks DET_MAX_SIDE
  // instead of being hardcoded at 960.
  static constexpr int kProfileVersion = 20260426;

  auto key = "v" + std::to_string(kProfileVersion) + ":" + type + ":" +
      onnx_path + ":" + std::to_string(onnx_size) + ":" +
      std::to_string(onnx_mtime) + ":" + std::to_string(trt_major) + "." +
      std::to_string(trt_minor) + "." + std::to_string(trt_patch) + ":sm" +
      std::to_string(gpu_major) + "." + std::to_string(gpu_minor) +
      ":drv" + std::to_string(cuda_driver) +
      ":rt" + std::to_string(cuda_runtime);
  // For det, the MAX dim of the optimization profile depends on DET_MAX_SIDE,
  // so each operator config gets its own engine (Triton/vLLM pattern).
  if (type == "det")
    key += ":dms" + std::to_string(read_det_max_side());
  // TRT_OPT_LEVEL changes which kernels TensorRT picks, so the produced
  // engine differs. Operators that toggle the level get separate cached
  // engines instead of silently reusing a stale one.
  key += ":opt" + std::to_string(read_trt_opt_level());
  auto hash = std::hash<std::string>{}(key);

  return cache_dir + "/" + type + "_" + std::to_string(hash) + ".trt";
}

static bool build_engine(const std::string &onnx_path,
                          const std::string &trt_path,
                          const std::string &type) {
  std::cout << "Building TRT engine: " << onnx_path << " -> " << trt_path;
  if (type == "det") std::cout << " (DET_MAX_SIDE=" << read_det_max_side() << ")";
  std::cout << '\n';

  auto builder = std::unique_ptr<nvinfer1::IBuilder>(
      nvinfer1::createInferBuilder(s_logger));
  if (!builder) return false;

  // Pass 0 — no NetworkDefinitionCreationFlag bits. In TRT 10 networks are
  // always explicit batch (kEXPLICIT_BATCH is value 0, deprecated, ignored),
  // and bit 0 now means kSTRONGLY_TYPED. The legacy `1U << kEXPLICIT_BATCH`
  // expression therefore evaluated to 1 — silently selecting a strongly-
  // typed network, which forbids setFlag(kFP16) below. TRT then silently
  // returns a null serialized plan with no kERROR log, surfacing only as
  // "Failed to build engine from <onnx_path>" on cold builds.
  auto network = std::unique_ptr<nvinfer1::INetworkDefinition>(
      builder->createNetworkV2(0U));
  if (!network) return false;

  auto parser = std::unique_ptr<nvonnxparser::IParser>(
      nvonnxparser::createParser(*network, s_logger));
  if (!parser || !parser->parseFromFile(onnx_path.c_str(),
      static_cast<int>(nvinfer1::ILogger::Severity::kWARNING)))
    return false;

  auto config = std::unique_ptr<nvinfer1::IBuilderConfig>(
      builder->createBuilderConfig());
  // Layout needs more workspace than det/rec/cls because PP-DocLayoutV3 is a
  // DETR-family model with large attention matmuls; 1 GB is fine for the
  // others. Det workspace scales with DET_MAX_SIDE: at 4096 the activation
  // tensors are ~17× larger than at 960, so 1 GB can be tight. Capped at
  // 4 GB (same ceiling as layout) so even 4096-side builds fit on a 16 GB
  // card alongside rec/cls/layout.
  size_t workspace_bytes;
  if (type == "layout") {
    workspace_bytes = 4ULL << 30;          // 4 GiB
  } else if (type == "det") {
    // (det_max/960)² × 1 GiB, capped at 4 GiB. At default 960 → 1 GiB
    // (unchanged). At 2048 → ~4 GiB. At 4096 → 4 GiB (cap).
    int det_max = read_det_max_side();
    double scale = (static_cast<double>(det_max) / 960.0) *
                   (static_cast<double>(det_max) / 960.0);
    size_t scaled = static_cast<size_t>((1ULL << 30) * scale);
    workspace_bytes = std::min(scaled, size_t(4ULL << 30));
    if (workspace_bytes < (1ULL << 30)) workspace_bytes = 1ULL << 30;
  } else {
    workspace_bytes = 1ULL << 30;          // rec, cls
  }
  config->setMemoryPoolLimit(nvinfer1::MemoryPoolType::kWORKSPACE, workspace_bytes);
  config->setFlag(nvinfer1::BuilderFlag::kFP16);
  // TRT_OPT_LEVEL: 0..5, default 5. Lower trades runtime perf for build time.
  const int opt_level = read_trt_opt_level();
  config->setBuilderOptimizationLevel(opt_level);
  std::cout << "[TRT] builder optimization level: " << opt_level << '\n';

  auto profile = builder->createOptimizationProfile();
  auto input = network->getInput(0);

  if (type == "det") {
    // MAX tracks DET_MAX_SIDE (default 960). MIN is the floor (32, after
    // round-down-to-32). OPT is the sweet-spot for typical inputs, capped
    // by MAX so a small DET_MAX_SIDE doesn't violate MIN<=OPT<=MAX.
    int det_max = read_det_max_side();
    int det_opt = std::min(640, det_max);
    profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMIN,
        nvinfer1::Dims4{1, 3, kDetMaxSideMin, kDetMaxSideMin});
    profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kOPT,
        nvinfer1::Dims4{1, 3, det_opt, det_opt});
    profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMAX,
        nvinfer1::Dims4{1, 3, det_max, det_max});
  } else if (type == "rec") {
    profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMIN,
        nvinfer1::Dims4{1, 3, 48, 48});
    profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kOPT,
        nvinfer1::Dims4{32, 3, 48, 320});
    profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMAX,
        nvinfer1::Dims4{32, 3, 48, 4000});
  } else if (type == "layout") {
    // PP-DocLayoutV3 has 3 inputs: image [B,3,800,800], im_shape [B,2],
    // scale_factor [B,2]. paddle2onnx does not guarantee input ordering, so
    // dispatch by name. Batch profile: min=1, opt=4, max=8 (measured sweet
    // spot on RTX 5090: 1.18 ms / image at B=4).
    for (int i = 0; i < network->getNbInputs(); ++i) {
      auto *in = network->getInput(i);
      std::string name = in->getName();
      if (name == "image") {
        profile->setDimensions(name.c_str(), nvinfer1::OptProfileSelector::kMIN,
            nvinfer1::Dims4{1, 3, 800, 800});
        profile->setDimensions(name.c_str(), nvinfer1::OptProfileSelector::kOPT,
            nvinfer1::Dims4{4, 3, 800, 800});
        profile->setDimensions(name.c_str(), nvinfer1::OptProfileSelector::kMAX,
            nvinfer1::Dims4{8, 3, 800, 800});
      } else if (name == "im_shape" || name == "scale_factor") {
        profile->setDimensions(name.c_str(), nvinfer1::OptProfileSelector::kMIN,
            nvinfer1::Dims2{1, 2});
        profile->setDimensions(name.c_str(), nvinfer1::OptProfileSelector::kOPT,
            nvinfer1::Dims2{4, 2});
        profile->setDimensions(name.c_str(), nvinfer1::OptProfileSelector::kMAX,
            nvinfer1::Dims2{8, 2});
      } else {
        std::cerr << "[TRT] Unexpected input for layout model: " << name << '\n';
        return false;
      }
    }
  } else {
    // cls: PP-OCRv5 textline orientation classifier (PP-LCNet_x0_25), input
    // 80x160. Must match kClsImageH/kClsImageW in classification/paddle_cls.h.
    profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMIN,
        nvinfer1::Dims4{1, 3, 80, 160});
    profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kOPT,
        nvinfer1::Dims4{32, 3, 80, 160});
    profile->setDimensions(input->getName(), nvinfer1::OptProfileSelector::kMAX,
        nvinfer1::Dims4{128, 3, 80, 160});
  }
  config->addOptimizationProfile(profile);

  auto plan = std::unique_ptr<nvinfer1::IHostMemory>(
      builder->buildSerializedNetwork(*network, *config));
  if (!plan || plan->size() == 0) return false;

  std::error_code ec;
  fs::create_directories(fs::path(trt_path).parent_path(), ec);
  if (ec) {
    std::cerr << "[TRT] Failed to create cache dir for " << trt_path << ": "
              << ec.message() << '\n';
    return false;
  }

  // Write to a temp file first, then atomic rename (prevents corruption from
  // concurrent builds in multi-replica Docker deployments).
  //
  // Both the open and the close-after-write check log on failure with the
  // strerror text. Earlier the open path was a silent `return false`, and
  // a uid-mismatch on a bind-mounted engine cache (host uid 1000, container
  // ocr uid 1001) hit it as EACCES — which surfaced only as "Failed to
  // build engine from <onnx>" from the caller, despite TRT having built
  // the engine successfully. Hours of bisection later, ergo this errno log.
  const auto tmp_path = trt_path + ".tmp." + std::to_string(getpid());
  std::ofstream file(tmp_path, std::ios::binary);
  if (!file) {
    std::cerr << "[TRT] Failed to open temp file " << tmp_path << ": "
              << std::strerror(errno) << '\n';
    return false;
  }
  file.write(static_cast<const char *>(plan->data()), plan->size());
  file.close();
  if (!file) {
    std::cerr << "[TRT] Failed to write temp file " << tmp_path
              << " (disk full?): " << std::strerror(errno) << '\n';
    fs::remove(tmp_path, ec);
    return false;
  }
  fs::rename(tmp_path, trt_path, ec);
  if (ec) {
    std::cerr << "[TRT] Failed to rename " << tmp_path << " -> " << trt_path
              << ": " << ec.message() << '\n';
    fs::remove(tmp_path, ec);
    return false;
  }

  std::cout << "Built: " << trt_path << " ("
            << static_cast<double>(plan->size()) / (1024 * 1024) << " MB)\n";
  return true;
}

void sweep_orphan_engine_temps(int min_age_seconds) {
  std::error_code ec;
  const auto cache_dir = get_engine_cache_dir();
  if (!fs::exists(cache_dir, ec) || ec) return;

  const auto now = fs::file_time_type::clock::now();
  int removed = 0;
  for (const auto &entry : fs::directory_iterator(cache_dir, ec)) {
    if (ec) break;
    if (!entry.is_regular_file(ec)) continue;
    const auto name = entry.path().filename().string();
    if (name.find(".tmp.") == std::string::npos) continue;

    const auto mtime = fs::last_write_time(entry.path(), ec);
    if (ec) { ec.clear(); continue; }
    const auto age = std::chrono::duration_cast<std::chrono::seconds>(
        now - mtime).count();
    if (age < min_age_seconds) continue;

    fs::remove(entry.path(), ec);
    if (!ec) ++removed;
    ec.clear();
  }
  if (removed > 0)
    std::cout << "[TRT] Removed " << removed
              << " orphan engine temp file(s) from " << cache_dir << '\n';
}

std::string ensure_trt_engine(const std::string &onnx_path,
                               const std::string &type) {
  if (!fs::exists(onnx_path)) {
    std::cerr << "[TRT] ONNX not found: " << onnx_path << '\n';
    return {};
  }

  auto trt_path = get_cached_engine_path(onnx_path, type);

  if (fs::exists(trt_path)) {
    std::cout << "Using cached engine: " << trt_path;
    if (type == "det") std::cout << " (DET_MAX_SIDE=" << read_det_max_side() << ")";
    std::cout << '\n';
    return trt_path;
  }

  if (!build_engine(onnx_path, trt_path, type)) {
    std::cerr << "[TRT] Failed to build engine from: " << onnx_path << '\n';
    return {};
  }

  return trt_path;
}

} // namespace turbo_ocr::engine
