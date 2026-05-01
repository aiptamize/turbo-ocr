#include "turbo_ocr/routes/common_routes.h"
#include "turbo_ocr/decode/image_config.h"
#include "turbo_ocr/decode/image_dims.h"
#include "turbo_ocr/server/env_utils.h"

#include <drogon/HttpAppFramework.h>
#include <json/json.h>

#include <format>

namespace turbo_ocr::routes {

namespace {
using decode::max_image_dim;

// Two-stage check for /ocr and /ocr/raw:
//   1. Pre-decode header sniff (PNG / JPEG): refuses oversized inputs
//      without ever calling the decoder, defending against decompression
//      bombs (a 1 KB PNG can claim 100k×100k → 30 GB decode buffer).
//   2. Post-decode check on the resulting cv::Mat: catches formats we
//      don't sniff (BMP, TIFF, WEBP).
// Returns true if the request was rejected (caller should return).
[[nodiscard]] inline bool reject_if_too_large_pre(
    const unsigned char *data, size_t len, server::DrogonCallback &cb) {
  if (auto d = decode::peek_image_dimensions(data, len)) {
    if (d->width > max_image_dim() || d->height > max_image_dim()) {
      cb(server::error_response(drogon::k400BadRequest, "DIMENSIONS_TOO_LARGE",
          std::format("Image dimensions {}x{} exceed maximum of {}x{}",
                      d->width, d->height, max_image_dim(), max_image_dim())));
      return true;
    }
  }
  return false;
}
[[nodiscard]] inline bool reject_if_too_large_post(
    const cv::Mat &img, server::DrogonCallback &cb) {
  if (img.cols > max_image_dim() || img.rows > max_image_dim()) {
    cb(server::error_response(drogon::k400BadRequest, "DIMENSIONS_TOO_LARGE",
        std::format("Image dimensions {}x{} exceed maximum of {}x{}",
                    img.cols, img.rows, max_image_dim(), max_image_dim())));
    return true;
  }
  return false;
}
} // namespace

void register_health_route(std::function<bool()> readiness_check) {
  auto health_ok = [](const drogon::HttpRequestPtr &,
                      std::function<void(const drogon::HttpResponsePtr &)> &&callback) {
    callback(server::make_response(drogon::k200OK, "ok"));
  };
  drogon::app().registerHandler("/health", health_ok, {drogon::Get});
  drogon::app().registerHandler("/health/live", health_ok, {drogon::Get});

  // /health/ready — verifies the pipeline is actually responsive
  auto ready_check = std::make_shared<std::function<bool()>>(std::move(readiness_check));
  drogon::app().registerHandler(
      "/health/ready",
      [ready_check](const drogon::HttpRequestPtr &,
                    std::function<void(const drogon::HttpResponsePtr &)> &&callback) {
        if (*ready_check && !(*ready_check)()) {
          callback(server::error_response(drogon::k503ServiceUnavailable,
              "NOT_READY", "Pipeline not ready"));
          return;
        }
        callback(server::make_response(drogon::k200OK, "ok"));
      },
      {drogon::Get});
}

void register_ocr_base64_route(server::WorkPool &pool,
                                const server::InferFunc &infer,
                                const server::ImageDecoder &decode,
                                bool layout_available) {
  drogon::app().registerHandler(
      "/ocr",
      [&pool, &infer, &decode, layout_available](
          const drogon::HttpRequestPtr &req,
          std::function<void(const drogon::HttpResponsePtr &)> &&callback) {

        server::InferOptions opts;
        if (auto r = server::parse_query_options(req, layout_available, &opts);
            !r.error.empty()) {
          callback(server::error_response(drogon::k400BadRequest,
                                           r.error_code.c_str(), r.error));
          return;
        }

        auto json = req->getJsonObject();
        if (!json) {
          callback(server::error_response(drogon::k400BadRequest, "INVALID_JSON", "Invalid JSON"));
          return;
        }
        if (!json->isMember("image") || !(*json)["image"].isString()
            || (*json)["image"].asString().empty()) {
          callback(server::error_response(drogon::k400BadRequest, "MISSING_IMAGE", "Empty or missing image field"));
          return;
        }

        auto b64_str = std::make_shared<std::string>((*json)["image"].asString());

        server::submit_work(pool, std::move(callback),
            [b64_str, &infer, &decode, opts](server::DrogonCallback &cb) {
          server::run_with_error_handling(cb, "/ocr", [&] {
            std::string decoded_bytes = turbo_ocr::base64_decode(*b64_str);
            if (decoded_bytes.empty()) {
              cb(server::error_response(drogon::k400BadRequest, "BASE64_DECODE_FAILED", "Failed to decode base64"));
              return;
            }

            const auto *bytes = reinterpret_cast<const unsigned char *>(decoded_bytes.data());
            if (reject_if_too_large_pre(bytes, decoded_bytes.size(), cb)) return;

            cv::Mat img = decode(bytes, decoded_bytes.size());
            if (img.empty()) {
              cb(server::error_response(drogon::k400BadRequest, "IMAGE_DECODE_FAILED", "Failed to decode image"));
              return;
            }
            if (reject_if_too_large_post(img, cb)) return;

            auto inf = infer(img, opts);
            cb(server::json_response(
                turbo_ocr::emit_results_json(inf.results, inf.layout, inf.reading_order, opts.want_blocks)));
          });
        });
      },
      {drogon::Post});
}

void register_ocr_raw_route(server::WorkPool &pool,
                             const server::InferFunc &infer,
                             const server::ImageDecoder &decode,
                             bool layout_available) {
  drogon::app().registerHandler(
      "/ocr/raw",
      [&pool, &infer, &decode, layout_available](
          const drogon::HttpRequestPtr &req,
          std::function<void(const drogon::HttpResponsePtr &)> &&callback) {

        if (req->body().empty()) {
          callback(server::error_response(drogon::k400BadRequest, "EMPTY_BODY", "Empty body"));
          return;
        }

        server::InferOptions opts;
        if (auto r = server::parse_query_options(req, layout_available, &opts);
            !r.error.empty()) {
          callback(server::error_response(drogon::k400BadRequest,
                                           r.error_code.c_str(), r.error));
          return;
        }

        server::submit_work(pool, std::move(callback),
            [req, &infer, &decode, opts](server::DrogonCallback &cb) {
          server::run_with_error_handling(cb, "/ocr/raw", [&] {
            const auto *data = reinterpret_cast<const unsigned char *>(req->body().data());
            size_t len = req->body().size();

            if (reject_if_too_large_pre(data, len, cb)) return;

            cv::Mat img = decode(data, len);
            if (img.empty()) {
              cb(server::error_response(drogon::k400BadRequest, "IMAGE_DECODE_FAILED", "Failed to decode image"));
              return;
            }
            if (reject_if_too_large_post(img, cb)) return;

            auto inf = infer(img, opts);
            cb(server::json_response(
                turbo_ocr::emit_results_json(inf.results, inf.layout, inf.reading_order, opts.want_blocks)));
          });
        });
      },
      {drogon::Post});
}

void register_common_routes(server::WorkPool &pool,
                             const server::InferFunc &infer,
                             const server::ImageDecoder &decode,
                             bool layout_available,
                             std::function<bool()> readiness_check) {
  register_health_route(std::move(readiness_check));
  register_ocr_base64_route(pool, infer, decode, layout_available);
  register_ocr_raw_route(pool, infer, decode, layout_available);
}

} // namespace turbo_ocr::routes
