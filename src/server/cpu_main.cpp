#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdlib>
#include <filesystem>
#include <format>
#include <iostream>
#include <string>
#include <thread>
#include <vector>

#include <drogon/HttpAppFramework.h>
#include <json/json.h>

#include "turbo_ocr/common/logger.h"
#include "turbo_ocr/decode/image_dims.h"
#include "turbo_ocr/decode/image_config.h"

#include "turbo_ocr/pdf/pdf_extraction_mode.h"
#include "turbo_ocr/pdf/pdf_text_layer.h"
#include "turbo_ocr/pipeline/cpu_pipeline_pool.h"
#include "turbo_ocr/render/pdf_renderer.h"
#include "turbo_ocr/server/env_utils.h"
#include "turbo_ocr/server/grpc_service.h"
#include "turbo_ocr/server/language_paths.h"
#include "turbo_ocr/server/metrics.h"
#include "turbo_ocr/server/server_types.h"
#include "turbo_ocr/server/work_pool.h"
#include "turbo_ocr/routes/common_routes.h"
#include "turbo_ocr/routes/pdf_routes.h"

using turbo_ocr::Box;
using turbo_ocr::OCRResultItem;
using turbo_ocr::base64_decode;
using turbo_ocr::results_to_json;
using turbo_ocr::emit_results_json;
using turbo_ocr::server::env_or;

namespace {
std::atomic<bool> g_shutdown_requested{false};
turbo_ocr::server::WorkPool *g_work_pool_for_drain = nullptr;

int shutdown_grace_seconds() {
  return turbo_ocr::server::env_int("SHUTDOWN_GRACE_SECONDS", 30, 0, 600);
}

// Mirrors the GPU binary: graceful drain of WorkPool inflight before
// app().quit() so K8s SIGTERM doesn't truncate inflight responses.
void begin_graceful_shutdown(const char *signal_name) {
  if (g_shutdown_requested.exchange(true)) return;
  TOCR_LOG_INFO("Graceful shutdown requested",
                "signal", std::string_view(signal_name),
                "grace_seconds", shutdown_grace_seconds());
  std::thread([signal_name]() {
    if (g_work_pool_for_drain) {
      auto deadline = std::chrono::seconds(shutdown_grace_seconds());
      bool drained = g_work_pool_for_drain->wait_drain(deadline);
      TOCR_LOG_INFO("Inflight work drain complete",
                    "drained", drained, "signal", std::string_view(signal_name));
    }
    drogon::app().quit();
  }).detach();
}
} // namespace

int main() {
  TOCR_LOG_INFO("PaddleOCR CPU-Only Mode (ONNX Runtime)");

  auto rec_paths = turbo_ocr::server::resolve_rec_paths("REC_MODEL");
  if (auto lang = turbo_ocr::server::ocr_lang(); !lang.empty())
    TOCR_LOG_INFO("Language selected via OCR_LANG",
                  "lang",  std::string_view(lang),
                  "rec",   std::string_view(rec_paths.rec),
                  "dict",  std::string_view(rec_paths.dict));
  auto det_model = env_or("DET_MODEL", "models/det.onnx");
  auto rec_model = rec_paths.rec;
  auto rec_dict = rec_paths.dict;
  auto cls_model = env_or("CLS_MODEL", "models/cls.onnx");

  // Validate model paths up front so a missing models/ tree fails fast
  // with a clear error rather than tripping a confusing ORT load failure
  // deep in pipeline construction. Dict file is also required on CPU.
  auto require_model = [](const std::string &path, const char *purpose) {
    if (!std::filesystem::exists(path)) {
      TOCR_LOG_ERROR("Model file missing",
                     "purpose", std::string_view(purpose),
                     "path", std::string_view(path));
      std::cerr << "[FATAL] " << purpose << " file not found at: " << path
                << "\n        Run scripts/download_models.sh or set "
                << purpose << "_MODEL env var.\n";
      std::exit(1);
    }
  };
  require_model(det_model, "DET");
  require_model(rec_model, "REC");
  require_model(rec_dict, "REC_DICT");
  require_model(cls_model, "CLS");

  if (turbo_ocr::server::env_enabled("DISABLE_ANGLE_CLS")) {
    cls_model.clear();
    TOCR_LOG_INFO("Angle classification disabled via DISABLE_ANGLE_CLS=1");
  }

  // Layout model (CPU via ONNX Runtime) — on by default. Optional: a
  // missing layout.onnx soft-disables the stage below rather than aborting.
  std::string layout_model = env_or("LAYOUT_ONNX", "models/layout/layout.onnx");
  bool layout_disabled = turbo_ocr::server::env_enabled("DISABLE_LAYOUT");
  bool layout_available = false;

  int pool_size = 4;
  if (const char *env = std::getenv("PIPELINE_POOL_SIZE"))
    pool_size = std::max(1, std::atoi(env));

  TOCR_LOG_INFO("CPU pipeline pool size", "pool_size", pool_size);
  auto pool = turbo_ocr::pipeline::make_cpu_pipeline_pool(
      pool_size, det_model, rec_model, rec_dict, cls_model);

  // Load layout model into each pipeline if enabled
  if (!layout_disabled && !layout_model.empty()) {
    bool all_ok = true;
    for (size_t i = 0; i < static_cast<size_t>(pool_size); ++i) {
      auto handle = pool->acquire();
      if (!handle->load_layout_model(layout_model)) {
        TOCR_LOG_WARN("Layout model not found; layout disabled");
        all_ok = false;
        break;
      }
    }
    if (all_ok) {
      layout_available = true;
      TOCR_LOG_INFO("Layout detection enabled (CPU/ONNX Runtime)");
    }
  } else if (layout_disabled) {
    TOCR_LOG_INFO("Layout detection disabled");
  }

  turbo_ocr::server::InferFunc infer =
      [&pool](const cv::Mat &img,
              const turbo_ocr::server::InferOptions &opts)
          -> turbo_ocr::server::InferResult {
    auto handle = pool->acquire();
    auto out = handle->run_with_layout(img, opts.want_layout,
                                        opts.want_reading_order);
    return turbo_ocr::server::InferResult{
        .results       = std::move(out.results),
        .layout        = std::move(out.layout),
        .reading_order = std::move(out.reading_order),
    };
  };

  turbo_ocr::server::ImageDecoder decode = turbo_ocr::server::cpu_decode_image;

  // Work pool for offloading blocking inference from Drogon event loop
  int work_threads = std::max(pool_size * 32, 128);
  turbo_ocr::server::WorkPool work_pool(work_threads);

  turbo_ocr::server::Metrics::instance().set_pool_size(pool_size);
  turbo_ocr::server::register_observability_middleware();
  turbo_ocr::server::register_metrics_route();

  // Single readiness probe shared by HTTP /health/ready and gRPC Health.
  // Acquires a pool handle to verify the work pool isn't wedged; cheap
  // and matches what the GPU binary does at /health/ready.
  auto readiness = [&pool]() -> bool {
    try {
      auto handle = pool->acquire();
      (void)handle;
      return true;
    } catch (...) {
      return false;
    }
  };
  turbo_ocr::routes::register_common_routes(work_pool, infer, decode,
                                             layout_available, readiness);

  // --- /ocr/pixels endpoint (raw BGR pixel data, zero decode overhead) ---
  drogon::app().registerHandler(
      "/ocr/pixels",
      [&work_pool, &infer, layout_available](
          const drogon::HttpRequestPtr &req,
          std::function<void(const drogon::HttpResponsePtr &)> &&callback) {
        turbo_ocr::server::InferOptions opts;
        if (auto r = turbo_ocr::server::parse_query_options(
                req, layout_available, &opts);
            !r.error.empty()) {
          callback(turbo_ocr::server::error_response(
              drogon::k400BadRequest, r.error_code.c_str(), r.error));
          return;
        }

        auto w_str = req->getHeader("X-Width");
        auto h_str = req->getHeader("X-Height");
        auto c_str = req->getHeader("X-Channels");

        if (w_str.empty() || h_str.empty()) {
          callback(turbo_ocr::server::error_response(drogon::k400BadRequest,
                                    "MISSING_HEADER", "Missing X-Width or X-Height headers"));
          return;
        }

        int width, height, channels;
        try {
          width = std::stoi(w_str);
          height = std::stoi(h_str);
          channels = c_str.empty() ? 3 : std::stoi(c_str);
        } catch (const std::exception &) {
          callback(turbo_ocr::server::error_response(drogon::k400BadRequest,
              "INVALID_HEADER", "Invalid X-Width, X-Height, or X-Channels header value"));
          return;
        }

        if (width <= 0 || height <= 0 || (channels != 1 && channels != 3)) {
          callback(turbo_ocr::server::error_response(drogon::k400BadRequest,
              "INVALID_DIMENSIONS", "Invalid dimensions or channels"));
          return;
        }

        // Configurable via MAX_IMAGE_DIM (default 16384). Same env var as
        // the GPU path so both servers share one knob.
        const int kMaxPixelDim = turbo_ocr::decode::max_image_dim();
        if (width > kMaxPixelDim || height > kMaxPixelDim) {
          callback(turbo_ocr::server::error_response(drogon::k400BadRequest,
              "DIMENSIONS_TOO_LARGE", std::format("Dimensions {}x{} exceed maximum of {}x{}",
                          width, height, kMaxPixelDim, kMaxPixelDim)));
          return;
        }

        size_t expected = static_cast<size_t>(width) * height * channels;
        if (req->body().size() != expected) {
          callback(turbo_ocr::server::error_response(drogon::k400BadRequest,
              "BODY_SIZE_MISMATCH", std::format("Body size mismatch: expected {} bytes ({}x{}x{}), got {}",
                          expected, width, height, channels, req->body().size())));
          return;
        }

        turbo_ocr::server::submit_work(work_pool, std::move(callback),
            [req, &infer, width, height, channels, opts](turbo_ocr::server::DrogonCallback &cb) {
          turbo_ocr::server::run_with_error_handling(cb, "/ocr/pixels", [&] {
            cv::Mat img(height, width,
                        channels == 3 ? CV_8UC3 : CV_8UC1,
                        const_cast<char *>(req->body().data()));
            auto inf = infer(img, opts);
            cb(turbo_ocr::server::json_response(
                turbo_ocr::emit_results_json(inf.results, inf.layout, inf.reading_order, opts.want_blocks)));
          });
        });
      },
      {drogon::Post});

  // --- /ocr/pdf endpoint (CPU: sequential page OCR) ---
  int pdf_daemons = 4, pdf_workers = 2;
  if (const char *env = std::getenv("PDF_DAEMONS"))
    pdf_daemons = std::max(1, std::atoi(env));
  if (const char *env = std::getenv("PDF_WORKERS"))
    pdf_workers = std::max(1, std::atoi(env));
  turbo_ocr::render::PdfRenderer pdf_renderer(pdf_daemons, pdf_workers);
  TOCR_LOG_INFO("PDF renderer initialized", "daemons", pdf_daemons, "workers", pdf_workers);
  turbo_ocr::pdf::ensure_pdfium_initialized();

  turbo_ocr::pdf::PdfMode default_pdf_mode = turbo_ocr::pdf::PdfMode::Ocr;
  if (auto *m = std::getenv("ENABLE_PDF_MODE"); m && *m)
    default_pdf_mode = turbo_ocr::pdf::parse_pdf_mode(m);

  turbo_ocr::routes::register_pdf_route(work_pool, infer, pdf_renderer, default_pdf_mode, layout_available);

  // --- /ocr/batch endpoint (CPU version) ---
  drogon::app().registerHandler(
      "/ocr/batch",
      [&work_pool, &pool, pool_size, &decode, layout_available](
          const drogon::HttpRequestPtr &req,
          std::function<void(const drogon::HttpResponsePtr &)> &&callback) {

        turbo_ocr::server::InferOptions opts;
        if (auto r = turbo_ocr::server::parse_query_options(
                req, layout_available, &opts);
            !r.error.empty()) {
          callback(turbo_ocr::server::error_response(
              drogon::k400BadRequest, r.error_code.c_str(), r.error));
          return;
        }
        const bool want_layout = opts.want_layout;

        auto json = req->getJsonObject();
        if (!json) {
          callback(turbo_ocr::server::error_response(drogon::k400BadRequest, "INVALID_JSON", "Invalid JSON"));
          return;
        }
        if (!json->isMember("images") || !(*json)["images"].isArray()) {
          callback(turbo_ocr::server::error_response(drogon::k400BadRequest, "INVALID_JSON", "Missing images array"));
          return;
        }

        auto &images_json = (*json)["images"];
        size_t n = images_json.size();
        if (n == 0) {
          callback(turbo_ocr::server::error_response(drogon::k400BadRequest, "EMPTY_BATCH", "Empty images array"));
          return;
        }

        // Pre-decode base64
        auto raw_bytes = std::make_shared<std::vector<std::string>>(n);
        for (size_t i = 0; i < n; ++i)
          (*raw_bytes)[i] = base64_decode(images_json[static_cast<int>(i)].asString());

        turbo_ocr::server::submit_work(work_pool, std::move(callback),
            [raw_bytes, n, &pool, pool_size, &decode, opts](turbo_ocr::server::DrogonCallback &cb) {
          const bool want_layout = opts.want_layout;
          // Same MAX_IMAGE_DIM cap as /ocr, /ocr/raw, /ocr/pixels.
          const int kMaxImageDim = turbo_ocr::decode::max_image_dim();

          // Per-slot batch state. Cardinality MUST equal the input array
          // length so callers can correlate batch_results[i] with their
          // images[i]; failed slots get tagged in `error`, never dropped.
          struct BatchItem {
            std::vector<OCRResultItem> results;
            std::vector<turbo_ocr::layout::LayoutBox> layout;
            std::vector<int> reading_order;
            std::string error;          // empty when the slot succeeded
          };
          std::vector<BatchItem> batch_items(n);
          std::vector<cv::Mat> imgs(n);

          // Empty inputs short-circuit so we never pay decode cost on
          // slots the caller never filled in.
          for (size_t i = 0; i < n; ++i) {
            if ((*raw_bytes)[i].empty()) batch_items[i].error = "empty";
          }

          // Pre-decode dim sniff: refuses oversized PNG/JPEG/WebP per-slot
          // before the decoder allocates a buffer. Mirrors GPU /ocr/batch.
          for (size_t i = 0; i < n; ++i) {
            if (!batch_items[i].error.empty()) continue;
            const auto &raw = (*raw_bytes)[i];
            if (auto d = turbo_ocr::decode::peek_image_dimensions(
                    reinterpret_cast<const unsigned char *>(raw.data()), raw.size())) {
              if (d->width > kMaxImageDim || d->height > kMaxImageDim) {
                batch_items[i].error = std::format(
                    "dimensions_too_large ({}x{} > {}x{})",
                    d->width, d->height, kMaxImageDim, kMaxImageDim);
              }
            }
          }

          // Decode every still-valid slot; tag decode failures per-slot.
          for (size_t i = 0; i < n; ++i) {
            if (!batch_items[i].error.empty()) continue;
            const auto &raw = (*raw_bytes)[i];
            imgs[i] = decode(
                reinterpret_cast<const unsigned char *>(raw.data()),
                raw.size());
            if (imgs[i].empty()) {
              batch_items[i].error = "decode_failed";
              continue;
            }
            // Post-decode safety net for BMP/TIFF/GIF (non-sniffed formats).
            if (imgs[i].cols > kMaxImageDim || imgs[i].rows > kMaxImageDim) {
              batch_items[i].error = std::format(
                  "dimensions_too_large ({}x{} > {}x{})",
                  imgs[i].cols, imgs[i].rows, kMaxImageDim, kMaxImageDim);
              imgs[i].release();
            }
          }

          // Build the work index list (slots that survived all checks).
          std::vector<size_t> valid_indices;
          valid_indices.reserve(n);
          for (size_t i = 0; i < n; ++i) {
            if (batch_items[i].error.empty() && !imgs[i].empty())
              valid_indices.push_back(i);
          }

          std::atomic<size_t> next_valid{0};
          int num_workers = valid_indices.empty()
              ? 0
              : std::min(static_cast<int>(valid_indices.size()), pool_size);
          {
            std::vector<std::jthread> threads;
            threads.reserve(num_workers);
            for (int w = 0; w < num_workers; ++w) {
              threads.emplace_back([&]() {
                // Acquire the pool handle outside the per-image loop —
                // pool exhaustion is a fatal worker-level error (no point
                // trying again), so it tags every remaining slot.
                std::unique_ptr<decltype(pool->acquire())> handle_holder;
                try {
                  handle_holder = std::make_unique<decltype(pool->acquire())>(
                      pool->acquire());
                } catch (const turbo_ocr::PoolExhaustedError &e) {
                  TOCR_LOG_ERROR("Batch worker pool exhausted", "route", "/ocr/batch");
                  // Tag every UNCLAIMED valid slot as failed so callers see it.
                  while (true) {
                    size_t k = next_valid.fetch_add(1);
                    if (k >= valid_indices.size()) break;
                    batch_items[valid_indices[k]].error = "pool_exhausted";
                  }
                  return;
                }
                auto &handle = *handle_holder;
                while (true) {
                  size_t k = next_valid.fetch_add(1);
                  if (k >= valid_indices.size()) break;
                  size_t idx = valid_indices[k];
                  // Per-image try/catch so one image failing does NOT
                  // leave all later slots silently empty with HTTP 200.
                  try {
                    auto out = handle->run_with_layout(imgs[idx], want_layout,
                                                       opts.want_reading_order);
                    batch_items[idx].results = std::move(out.results);
                    batch_items[idx].layout = std::move(out.layout);
                    batch_items[idx].reading_order = std::move(out.reading_order);
                  } catch (const std::exception &e) {
                    TOCR_LOG_ERROR("Batch image error", "route", "/ocr/batch",
                                   "image_index", idx, "error",
                                   std::string_view(e.what()));
                    batch_items[idx].error = e.what();
                  } catch (...) {
                    TOCR_LOG_ERROR("Batch image error: unknown",
                                   "route", "/ocr/batch", "image_index", idx);
                    batch_items[idx].error = "unknown";
                  }
                }
              });
            }
          } // jthreads auto-join here

          // Emit per-image results AND a sibling errors array so clients
          // can see which slots failed without having to compare lengths.
          std::string json_str;
          json_str.reserve(batch_items.size() * 1024);
          json_str += "{\"batch_results\":[";
          for (size_t i = 0; i < batch_items.size(); ++i) {
            if (i > 0) json_str += ',';
            json_str += emit_results_json(batch_items[i].results, batch_items[i].layout, batch_items[i].reading_order, opts.want_blocks);
          }
          json_str += "],\"errors\":[";
          for (size_t i = 0; i < batch_items.size(); ++i) {
            if (i > 0) json_str += ',';
            const auto &e = batch_items[i].error;
            if (e.empty()) {
              json_str += "null";
            } else {
              json_str += '"';
              for (char c : e) {
                if (c == '"' || c == '\\') json_str += '\\';
                json_str += c;
              }
              json_str += '"';
            }
          }
          json_str += "]}";
          cb(turbo_ocr::server::json_response(std::move(json_str)));
        });
      },
      {drogon::Post});

  // gRPC server
  int grpc_port = 50051;
  if (const char *env = std::getenv("GRPC_PORT"))
    grpc_port = std::max(1, std::atoi(env));
  auto grpc_handle = turbo_ocr::server::start_grpc_server(
      infer, grpc_port, &pdf_renderer, default_pdf_mode, layout_available,
      readiness);

  // HTTP server (Drogon)
  int port = 8080;
  if (const char *env = std::getenv("PORT"))
    port = std::max(1, std::atoi(env));

  // MAX_BODY_MB caps the largest accepted upload (default 100).
  // MAX_BODY_MEMORY_MB tunes the in-memory buffer threshold (default
  // 1024 MB / 1 GiB); bodies above it spill to /tmp. The cap is
  // clamped to MAX_BODY_MB below, so the default keeps every body in
  // RAM until the operator raises MAX_BODY_MB past 1 GiB. See main.cpp
  // for the full rationale — same knob on both CPU and GPU servers.
  int max_body_mb = turbo_ocr::server::env_int("MAX_BODY_MB", 100, 1, 102400);
  int max_body_mem_mb = turbo_ocr::server::env_int(
      "MAX_BODY_MEMORY_MB", 1024, 1, 102400);
  if (max_body_mem_mb > max_body_mb) max_body_mem_mb = max_body_mb;
  size_t max_body_bytes = static_cast<size_t>(max_body_mb) * 1024 * 1024;
  size_t max_mem_bytes  = static_cast<size_t>(max_body_mem_mb) * 1024 * 1024;

  TOCR_LOG_INFO("Starting CPU-Only OCR Server", "port", port, "grpc_port",
                grpc_port, "body_cap_mb_drogon", max_body_mb,
                "body_cap_mb_nginx", max_body_mb, "body_mem_mb", max_body_mem_mb);

  // Graceful shutdown on SIGTERM (Docker / K8s) and SIGINT (Ctrl-C):
  // drain WorkPool inflight up to SHUTDOWN_GRACE_SECONDS before quit().
  g_work_pool_for_drain = &work_pool;
  drogon::app()
      .setTermSignalHandler([] { begin_graceful_shutdown("SIGTERM"); })
      .setIntSignalHandler([]  { begin_graceful_shutdown("SIGINT");  })
      .addListener("0.0.0.0", port)
      .setThreadNum(4)
      .setIdleConnectionTimeout(120)
      .setClientMaxBodySize(max_body_bytes)
      .setClientMaxMemoryBodySize(max_mem_bytes)
      .run();

  TOCR_LOG_INFO("HTTP server stopped, shutting down gRPC");
  grpc_handle.server->Shutdown();
  TOCR_LOG_INFO("Shutdown complete");
  return 0;
}
