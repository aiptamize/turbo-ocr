#pragma once

#include <chrono>
#include <climits>
#include <cstring>
#include <format>
#include <functional>
#include <memory>
#include <random>
#include <string>
#include <vector>

#include "turbo_ocr/common/logger.h"

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include <drogon/HttpRequest.h>
#include <drogon/HttpResponse.h>

#include "turbo_ocr/common/encoding.h"
#include "turbo_ocr/common/errors.h"
#include "turbo_ocr/common/serialization.h"
#include "turbo_ocr/common/types.h"
#include "turbo_ocr/decode/fast_png_decoder.h"
#include "turbo_ocr/layout/layout_types.h"
#include "turbo_ocr/server/metrics.h"

namespace turbo_ocr::server {

/// Combined result of one inference: text OCR results + optional layout.
struct InferResult {
  std::vector<OCRResultItem>            results;
  std::vector<layout::LayoutBox>        layout;
  std::vector<int>                      reading_order;
};

/// Per-request feature flags parsed from query parameters.
struct InferOptions {
  bool want_layout = false;
  bool want_reading_order = false;
  // ?as_blocks=1 — emit a `blocks` array (paragraph-level aggregate,
  // one entry per non-empty layout cell, mirrors PaddleX's
  // PP-StructureV3 parsing_res_list granularity). Auto-enables layout
  // and reading_order since aggregation needs both.
  bool want_blocks = false;
};

/// Image decoder: (raw_bytes_ptr, length) -> cv::Mat
using ImageDecoder = std::function<cv::Mat(const unsigned char *data, size_t len)>;

/// Inference function: given cv::Mat + feature flags, run OCR pipeline.
using InferFunc = std::function<InferResult(const cv::Mat &, const InferOptions &)>;

/// Drogon callback alias.
using DrogonCallback = std::function<void(const drogon::HttpResponsePtr &)>;

// ── UUID v7 (timestamp-ordered, ~50ns) ──────────────────────────────────

[[nodiscard]] inline std::string generate_uuid_v7() {
  auto ms = static_cast<uint64_t>(
      std::chrono::duration_cast<std::chrono::milliseconds>(
          std::chrono::system_clock::now().time_since_epoch())
          .count());

  thread_local std::mt19937_64 rng(std::random_device{}());
  uint64_t rand_a = rng();
  uint64_t rand_b = rng();

  uint8_t u[16];
  u[0]  = (ms >> 40) & 0xFF;
  u[1]  = (ms >> 32) & 0xFF;
  u[2]  = (ms >> 24) & 0xFF;
  u[3]  = (ms >> 16) & 0xFF;
  u[4]  = (ms >> 8)  & 0xFF;
  u[5]  = ms & 0xFF;
  std::memcpy(u + 6, &rand_a, 2);
  std::memcpy(u + 8, &rand_b, 8);
  u[6] = (u[6] & 0x0F) | 0x70;   // version 7
  u[8] = (u[8] & 0x3F) | 0x80;   // variant 10

  char buf[37];
  std::snprintf(buf, sizeof(buf),
      "%02x%02x%02x%02x-%02x%02x-%02x%02x-%02x%02x-%02x%02x%02x%02x%02x%02x",
      u[0],u[1],u[2],u[3],u[4],u[5],u[6],u[7],
      u[8],u[9],u[10],u[11],u[12],u[13],u[14],u[15]);
  return std::string(buf, 36);
}

// ── Response helpers ────────────────────────────────────────────────────

/// Structured JSON error response: {"error":{"code":"...","message":"..."}}
[[nodiscard]] inline drogon::HttpResponsePtr error_response(
    drogon::HttpStatusCode status, const char *code, std::string message) {
  std::string body;
  body.reserve(64 + std::strlen(code) + message.size());
  body += R"({"error":{"code":")";
  body += code;
  body += R"(","message":")";
  // Escape quotes in message
  for (char c : message) {
    if (c == '"') body += "\\\"";
    else if (c == '\\') body += "\\\\";
    else body += c;
  }
  body += R"("}})";
  auto resp = drogon::HttpResponse::newHttpResponse();
  resp->setStatusCode(status);
  resp->setBody(std::move(body));
  resp->setContentTypeString("application/json");
  return resp;
}

/// Plain-text response (for /health and non-error uses).
[[nodiscard]] inline drogon::HttpResponsePtr make_response(
    drogon::HttpStatusCode code, std::string body) {
  auto resp = drogon::HttpResponse::newHttpResponse();
  resp->setStatusCode(code);
  resp->setBody(std::move(body));
  return resp;
}

/// JSON success response.
[[nodiscard]] inline drogon::HttpResponsePtr json_response(std::string json_str) {
  auto resp = drogon::HttpResponse::newHttpResponse();
  resp->setStatusCode(drogon::k200OK);
  resp->setBody(std::move(json_str));
  resp->setContentTypeString("application/json");
  return resp;
}

// ── Error handling wrapper ──────────────────────────────────────────────

template <typename F>
void run_with_error_handling(DrogonCallback &cb, const char *route, F &&fn) {
  try {
    fn();
  } catch (const turbo_ocr::PoolExhaustedError &e) {
    cb(error_response(drogon::k503ServiceUnavailable, "SERVER_BUSY", e.what()));
  } catch (const turbo_ocr::ImageDecodeError &e) {
    cb(error_response(drogon::k400BadRequest, "IMAGE_DECODE_FAILED", e.what()));
  } catch (const std::exception &e) {
    TOCR_LOG_ERROR_RL("Inference error", "route", std::string_view(route), "error", std::string_view(e.what()));
    cb(error_response(drogon::k500InternalServerError, "INFERENCE_ERROR", "Inference error"));
  } catch (...) {
    TOCR_LOG_ERROR_RL("Inference error: unknown exception", "route", std::string_view(route));
    cb(error_response(drogon::k500InternalServerError, "INFERENCE_ERROR", "Inference error"));
  }
}

} // namespace turbo_ocr::server

#include "turbo_ocr/server/work_pool.h"

namespace turbo_ocr::server {

// ── Work submission ─────────────────────────────────────────────────────

/// Submit blocking work to a WorkPool safely.
/// Callback is wrapped in shared_ptr so it survives if submit() throws.
/// Observability headers (X-Request-Id, X-Inference-Time-Ms, Retry-After)
/// are injected by the middleware registered in register_observability_middleware().
template <typename F>
void submit_work(WorkPool &pool, DrogonCallback &&callback, F &&work) {
  auto cb = std::make_shared<DrogonCallback>(std::move(callback));
  try {
    pool.submit([cb, w = std::forward<F>(work)]() mutable { w(*cb); });
  } catch (const turbo_ocr::PoolExhaustedError &e) {
    Metrics::instance().record_pool_exhaustion();
    (*cb)(error_response(drogon::k503ServiceUnavailable, "SERVER_BUSY", e.what()));
  }
}

/// Register Drogon middleware for observability headers and metrics.
/// Call once before drogon::app().run().
///
/// Pre-handling:  generates X-Request-Id (or propagates from client),
///                records request start time in request attributes.
/// Post-handling: injects X-Request-Id, X-Inference-Time-Ms, Retry-After
///                headers; records metrics.
inline void register_observability_middleware() {
  // Pre-request: assign request ID + start time
  drogon::app().registerPreHandlingAdvice(
      [](const drogon::HttpRequestPtr &req) {
        auto id = req->getHeader("X-Request-Id");
        if (id.empty()) id = generate_uuid_v7();
        req->addHeader("X-Request-Id", id);  // store for post-handler
        // Store start time as attribute (nanoseconds since epoch)
        auto now = std::chrono::steady_clock::now().time_since_epoch().count();
        req->addHeader("X-Start-Ns", std::to_string(now));
      });

  // Post-request: inject response headers + record metrics
  drogon::app().registerPostHandlingAdvice(
      [](const drogon::HttpRequestPtr &req,
         const drogon::HttpResponsePtr &resp) {
        // X-Request-Id
        auto req_id = req->getHeader("X-Request-Id");
        if (!req_id.empty())
          resp->addHeader("X-Request-Id", req_id);

        // X-Inference-Time-Ms
        auto start_ns_str = req->getHeader("X-Start-Ns");
        double duration_s = 0.0;
        if (!start_ns_str.empty()) {
          try {
            auto start_ns = std::stoll(start_ns_str);
            auto now_ns = std::chrono::steady_clock::now().time_since_epoch().count();
            auto ms = (now_ns - start_ns) / 1'000'000;
            resp->addHeader("X-Inference-Time-Ms", std::to_string(ms));
            duration_s = static_cast<double>(now_ns - start_ns) / 1e9;
          } catch (...) {}
        }

        // Retry-After on 503
        if (resp->statusCode() == drogon::k503ServiceUnavailable)
          resp->addHeader("Retry-After", "1");

        // Metrics
        auto path = req->path();
        if (path != "/metrics") {
          auto route = Metrics::route_from_path(path);
          int status = static_cast<int>(resp->statusCode());
          Metrics::instance().record_request(route, status, duration_s);
          Metrics::instance().record_request_size(req->body().size());
        }
      });
}

// ── Utilities ───────────────────────────────────────────────────────────

[[nodiscard]] inline cv::Mat cpu_decode_image(const unsigned char *data, size_t len) {
  // PNG via Wuffs (fast path); every other format (JPEG, WebP, BMP, TIFF,
  // GIF, …) via cv::imdecode. OpenCV's imgcodecs is linked to libwebp /
  // libtiff so it covers the rest.
  auto opencv_decode = [&]() -> cv::Mat {
    if (len > static_cast<size_t>(INT_MAX)) return {};
    return cv::imdecode(
        cv::Mat(1, static_cast<int>(len), CV_8UC1,
                const_cast<unsigned char *>(data)),
        cv::IMREAD_COLOR);
  };
  if (decode::FastPngDecoder::is_png(data, len))
    return decode::FastPngDecoder::decode(data, len);
  return opencv_decode();
}

[[nodiscard]] inline std::string parse_layout_query(const drogon::HttpRequestPtr &req,
                                                     bool layout_available,
                                                     bool *out) {
  *out = false;
  auto v = req->getParameter("layout");
  if (v.empty()) return {};
  std::string s(v);
  for (char &c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  bool on;
  if (s == "1" || s == "true" || s == "on" || s == "yes")       on = true;
  else if (s == "0" || s == "false" || s == "off" || s == "no") on = false;
  else return std::format("Invalid layout param: '{}' (expected 0|1)", s);
  if (on && !layout_available) {
    return std::string("Layout requested but the layout model is not loaded. "
                       "Either models/layout/layout.onnx is missing from the "
                       "image, or the server was started with DISABLE_LAYOUT=1.");
  }
  *out = on;
  return {};
}

// Parse a generic boolean query param ("1"/"true"/"on"/"yes" etc.).
// Returns empty string on success and writes to *out; otherwise returns an
// error message. When the parameter is absent, *out is set to false and an
// empty string is returned.
[[nodiscard]] inline std::string parse_bool_query(const drogon::HttpRequestPtr &req,
                                                   const char *key,
                                                   bool *out) {
  *out = false;
  auto v = req->getParameter(key);
  if (v.empty()) return {};
  std::string s(v);
  for (char &c : s) c = static_cast<char>(std::tolower(static_cast<unsigned char>(c)));
  if (s == "1" || s == "true" || s == "on" || s == "yes")       { *out = true; return {}; }
  if (s == "0" || s == "false" || s == "off" || s == "no")      { *out = false; return {}; }
  return std::format("Invalid {} param: '{}' "
                     "(expected 0/1, true/false, on/off, or yes/no)",
                     key, s);
}

// Parse the full set of opt-in query parameters for inference routes.
// `layout` and `reading_order` both default to 0; either being set to 1
// without the underlying layout model causes a 400 with a descriptive
// error code (LAYOUT_DISABLED).
struct ParseOptionsResult {
  std::string error;     // empty on success
  std::string error_code; // populated when error is non-empty
};
[[nodiscard]] inline ParseOptionsResult
parse_query_options(const drogon::HttpRequestPtr &req,
                    bool layout_available,
                    InferOptions *out) {
  *out = {};
  if (auto err = parse_layout_query(req, layout_available, &out->want_layout);
      !err.empty())
    return {err, "INVALID_PARAMETER"};

  if (auto err = parse_bool_query(req, "reading_order",
                                   &out->want_reading_order);
      !err.empty())
    return {err, "INVALID_PARAMETER"};
  if (out->want_reading_order && !layout_available) {
    // Reading order is derived from layout boxes — without the model
    // there's nothing to derive from. Reject the request explicitly so
    // clients don't silently get the y/x fallback they didn't ask for.
    return {"reading_order=1 requires the layout model: start the server "
            "without DISABLE_LAYOUT=1 (layout is on by default)",
            "LAYOUT_DISABLED"};
  }
  if (out->want_reading_order && !out->want_layout) {
    // Reading order auto-enables layout so /ocr behaves as documented:
    // ?reading_order=1 alone yields a populated reading_order array.
    out->want_layout = true;
  }

  if (auto err = parse_bool_query(req, "as_blocks", &out->want_blocks);
      !err.empty())
    return {err, "INVALID_PARAMETER"};
  if (out->want_blocks && !layout_available) {
    return {"as_blocks=1 requires the layout model: start the server "
            "without DISABLE_LAYOUT=1 (layout is on by default)",
            "LAYOUT_DISABLED"};
  }
  if (out->want_blocks) {
    // Aggregation needs reading_order (and reading_order needs layout).
    out->want_reading_order = true;
    out->want_layout = true;
  }

  return {};
}

} // namespace turbo_ocr::server
