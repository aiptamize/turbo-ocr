#pragma once

#include <climits>
#include <format>
#include <functional>
#include <string>
#include <vector>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include "turbo_ocr/common/encoding.h"
#include "turbo_ocr/common/errors.h"
#include "turbo_ocr/common/serialization.h"
#include "turbo_ocr/common/types.h"
#include "turbo_ocr/decode/fast_png_decoder.h"
#include "turbo_ocr/layout/layout_types.h"
#include "crow/crow_all.h"

namespace turbo_ocr::server {

/// Combined result of one inference: text OCR results + optional layout.
struct InferResult {
  std::vector<OCRResultItem>       results;
  std::vector<layout::LayoutBox>   layout;
};

/// Image decoder: (raw_bytes_ptr, length) -> cv::Mat
using ImageDecoder = std::function<cv::Mat(const unsigned char *data, size_t len)>;

/// Inference function: given cv::Mat + layout flag, run OCR pipeline.
using InferFunc = std::function<InferResult(const cv::Mat &, bool want_layout)>;

/// Default CPU-only image decoder: JPEG (OpenCV) and PNG (Wuffs).
[[nodiscard]] inline cv::Mat cpu_decode_image(const unsigned char *data, size_t len) {
  if (len >= 2 && data[0] == 0xFF && data[1] == 0xD8) {
    if (len > static_cast<size_t>(INT_MAX)) return {};
    return cv::imdecode(
        cv::Mat(1, static_cast<int>(len), CV_8UC1,
                const_cast<unsigned char *>(data)),
        cv::IMREAD_COLOR);
  }
  if (decode::FastPngDecoder::is_png(data, len))
    return decode::FastPngDecoder::decode(data, len);
  return {};
}

/// Parse `?layout=0|1|on|off|true|false|yes|no` from a Crow request.
[[nodiscard]] inline std::string parse_layout_query(const crow::request &req,
                                                     bool layout_available,
                                                     bool *out) {
  *out = false;
  const char *v = req.url_params.get("layout");
  if (!v || !*v) return {};
  std::string s(v);
  for (char &c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  bool on;
  if (s == "1" || s == "true" || s == "on" || s == "yes")       on = true;
  else if (s == "0" || s == "false" || s == "off" || s == "no") on = false;
  else return std::format("Invalid layout param: '{}' (expected 0|1)", s);
  if (on && !layout_available) {
    return std::string("Layout requested but server was not started with "
                       "ENABLE_LAYOUT=1 — restart the server with that env var "
                       "to enable layout detection.");
  }
  *out = on;
  return {};
}

} // namespace turbo_ocr::server
