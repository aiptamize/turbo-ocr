#pragma once

#include <algorithm>
#include <cstdlib>

namespace turbo_ocr::detection {

// Single source of truth for DET_MAX_SIDE.
//
// Read by:
//   - paddle_det.cpp (GPU detector — sizes pinned input buffers)
//   - cpu_paddle_det.cpp (CPU detector — same role)
//   - onnx_to_trt.cpp (TRT engine builder — sizes the optimization profile
//     MAX, and is included in the engine cache key)
//
// All three call sites MUST get the same value or the engine and the
// runtime will silently disagree. The function is `inline` and reads from
// `std::getenv` on every call, but the env is conventionally set once at
// process start. Bounds are kept symmetric with the integer-overflow
// safeguard in paddle_det.cpp:32 (`max_pixels = kMaxSideLen_²`).
inline constexpr int kDetMaxSideMin = 32;
inline constexpr int kDetMaxSideMax = 4096;
inline constexpr int kDetMaxSideDefault = 960;

[[nodiscard]] inline int read_det_max_side() {
  int v = kDetMaxSideDefault;
  if (const char *env = std::getenv("DET_MAX_SIDE"))
    v = std::atoi(env);
  return std::clamp(v, kDetMaxSideMin, kDetMaxSideMax);
}

} // namespace turbo_ocr::detection
