#pragma once

#include "turbo_ocr/server/env_utils.h"

namespace turbo_ocr::decode {

// Single source of truth for the MAX_IMAGE_DIM cap (px). Read once on first
// call; subsequent calls hit the function-static. Same env var used by every
// image-accepting route (HTTP /ocr, /ocr/raw, /ocr/batch, /ocr/pixels and
// gRPC Recognize/RecognizeBatch) so operators have one knob.
[[nodiscard]] inline int max_image_dim() {
  static const int v = server::env_int("MAX_IMAGE_DIM", 16384, 64, 65535);
  return v;
}

} // namespace turbo_ocr::decode
