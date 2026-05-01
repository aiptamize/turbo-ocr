#include <atomic>
#include <chrono>
#include <csignal>
#include <cstdlib>
#include <filesystem>
#include <format>
#include <iostream>
#include <memory>
#include <mutex>
#include <string>
#include <thread>

#include "turbo_ocr/common/logger.h"

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include <drogon/HttpAppFramework.h>

#include "turbo_ocr/decode/nvjpeg_decoder.h"
#include "turbo_ocr/engine/onnx_to_trt.h"
#include "turbo_ocr/pdf/pdf_extraction_mode.h"
#include "turbo_ocr/pdf/pdf_text_layer.h"
#include "turbo_ocr/pipeline/pipeline_dispatcher.h"
#include "turbo_ocr/render/pdf_renderer.h"
#include "turbo_ocr/server/env_utils.h"
#include "turbo_ocr/server/grpc_service.h"
#include "turbo_ocr/server/language_paths.h"
#include "turbo_ocr/server/metrics.h"
#include "turbo_ocr/server/server_types.h"
#include "turbo_ocr/server/work_pool.h"
#include "turbo_ocr/routes/common_routes.h"
#include "turbo_ocr/routes/image_routes.h"
#include "turbo_ocr/routes/pdf_routes.h"

using turbo_ocr::decode::FastPngDecoder;
using turbo_ocr::decode::NvJpegDecoder;
using turbo_ocr::render::PdfRenderer;
using turbo_ocr::server::env_or;

namespace {
std::atomic<bool> g_shutdown_requested{false};
turbo_ocr::server::WorkPool *g_work_pool_for_drain = nullptr;

// Default 30s ceiling — long enough for a typical PDF inflight to finish,
// short enough to stay below the standard 30s SIGKILL grace in K8s minus
// signal-propagation slop. Override with SHUTDOWN_GRACE_SECONDS.
int shutdown_grace_seconds() {
  return turbo_ocr::server::env_int("SHUTDOWN_GRACE_SECONDS", 30, 0, 600);
}

// Runs from Drogon's main loop (registered via setTermSignalHandler) —
// safe to start a thread, log, and call app().quit(). The detached
// drainer waits for the WorkPool to quiesce before tearing down Drogon
// so inflight requests get to send their response.
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
  auto rec_paths = turbo_ocr::server::resolve_rec_paths("REC_ONNX");
  if (auto lang = turbo_ocr::server::ocr_lang(); !lang.empty())
    TOCR_LOG_INFO("Language selected via OCR_LANG",
                  "lang",  std::string_view(lang),
                  "rec",   std::string_view(rec_paths.rec),
                  "dict",  std::string_view(rec_paths.dict));
  auto rec_dict = rec_paths.dict;

  // Validate model paths up front so a missing models/ tree fails fast with
  // a clear error rather than tripping a confusing CUDA/TRT error deep in
  // pipeline construction. ensure_trt_engine() returns "" on missing ONNX,
  // which the dispatcher only notices much later.
  auto require_model = [](const std::string &path, const char *purpose) {
    if (!std::filesystem::exists(path)) {
      TOCR_LOG_ERROR("Model file missing",
                     "purpose", std::string_view(purpose),
                     "path", std::string_view(path));
      std::cerr << "[FATAL] " << purpose << " model not found at: " << path
                << "\n        Run scripts/download_models.sh or set "
                << purpose << "_ONNX env var.\n";
      std::exit(1);
    }
  };
  require_model(env_or("DET_ONNX", "models/det.onnx"), "DET");
  require_model(rec_paths.rec, "REC");
  require_model(env_or("CLS_ONNX", "models/cls.onnx"), "CLS");

  // Auto-build TRT engines from ONNX (cached by TRT version + model hash)
  // Sweep orphan .trt.tmp.* files left by previous crashed processes; safe
  // because in-progress builds by sibling replicas are protected by the
  // 60-second min-age window inside the sweeper.
  turbo_ocr::engine::sweep_orphan_engine_temps();
  auto det_model = turbo_ocr::engine::ensure_trt_engine(
      env_or("DET_ONNX", "models/det.onnx"), "det");
  auto rec_model = turbo_ocr::engine::ensure_trt_engine(rec_paths.rec, "rec");
  auto cls_model = turbo_ocr::engine::ensure_trt_engine(
      env_or("CLS_ONNX", "models/cls.onnx"), "cls");
  if (turbo_ocr::server::env_enabled("DISABLE_ANGLE_CLS")) {
    cls_model.clear();
    TOCR_LOG_INFO("Angle classification disabled via DISABLE_ANGLE_CLS=1");
  }

  // Optional PP-DocLayoutV3 stage. ON by default — users can disable with
  // DISABLE_LAYOUT=1 to save ~300-500 MB VRAM. Also accepts legacy
  // ENABLE_LAYOUT=0 to disable.
  std::string layout_model;
  bool layout_disabled = turbo_ocr::server::env_enabled("DISABLE_LAYOUT") ||
                          (std::getenv("ENABLE_LAYOUT") && !turbo_ocr::server::env_enabled("ENABLE_LAYOUT"));
  if (!layout_disabled) {
    if (auto *pre = std::getenv("LAYOUT_TRT"); pre && *pre) {
      layout_model = pre;
      TOCR_LOG_INFO("Layout detection enabled", "engine", std::string_view(layout_model));
    } else {
      // Layout ONNX is optional — soft-disable with a warning if missing
      // rather than hard-failing, so installs without layout still serve.
      layout_model = turbo_ocr::engine::ensure_trt_engine(
          env_or("LAYOUT_ONNX", "models/layout/layout.onnx"), "layout");
      if (layout_model.empty()) {
        TOCR_LOG_WARN("Layout model (layout.onnx) not found; layout stage will be disabled");
      } else {
        TOCR_LOG_INFO("Layout detection enabled");
      }
    }
  } else {
    TOCR_LOG_INFO("Layout detection disabled (set DISABLE_LAYOUT=0 to enable)");
  }

  // PDF extraction mode default
  turbo_ocr::pdf::PdfMode default_pdf_mode = turbo_ocr::pdf::PdfMode::Ocr;
  if (auto *m = std::getenv("ENABLE_PDF_MODE"); m && *m) {
    default_pdf_mode = turbo_ocr::pdf::parse_pdf_mode(m);
    TOCR_LOG_INFO("PDF extraction default mode configured", "mode", turbo_ocr::pdf::mode_name(default_pdf_mode));
  } else {
    TOCR_LOG_INFO("PDF extraction default mode: ocr (override per-request with /ocr/pdf?mode=<geometric|auto|auto_verified>)");
  }
  turbo_ocr::pdf::ensure_pdfium_initialized();

  // Pipeline pool
  int pool_size = 4;
  if (const char *env = std::getenv("PIPELINE_POOL_SIZE")) {
    pool_size = std::max(1, std::atoi(env));
  } else {
    size_t free_mem = 0, total_mem = 0;
    if (cudaMemGetInfo(&free_mem, &total_mem) == cudaSuccess) {
      int vram_gb = static_cast<int>(total_mem >> 30);
      if (vram_gb >= 14) pool_size = 5;
      else if (vram_gb >= 12) pool_size = 3;
      else if (vram_gb >= 8)  pool_size = 2;
      else                     pool_size = 1;
      TOCR_LOG_INFO("Auto-detected pipeline pool size", "pool_size", pool_size, "vram_gb", vram_gb);
    }
  }

  auto dispatcher = turbo_ocr::pipeline::make_pipeline_dispatcher(
      pool_size, det_model, rec_model, rec_dict, cls_model, layout_model);

  // PDF renderer
  int pdf_daemons = 16, pdf_workers = 4;
  if (const char *env = std::getenv("PDF_DAEMONS"))
    pdf_daemons = std::max(1, std::atoi(env));
  if (const char *env = std::getenv("PDF_WORKERS"))
    pdf_workers = std::max(1, std::atoi(env));
  PdfRenderer pdf_renderer(pdf_daemons, pdf_workers);
  TOCR_LOG_INFO("PDF renderer initialized", "daemons", pdf_daemons, "workers", pdf_workers);

  // nvJPEG
  TOCR_LOG_INFO("Initializing nvJPEG decoders");
  thread_local NvJpegDecoder tl_nvjpeg;
  bool nvjpeg_available = tl_nvjpeg.available();
  if (nvjpeg_available)
    TOCR_LOG_INFO("nvJPEG GPU-accelerated JPEG decode enabled");
  else
    TOCR_LOG_WARN("nvJPEG not available, using OpenCV JPEG decode");

  // Image decoder: JPEG via nvJPEG (GPU), PNG via Wuffs (fast path), every
  // other format (WebP, BMP, TIFF, GIF, …) via cv::imdecode. OpenCV's
  // imgcodecs is linked to libwebp/libtiff so cv::imdecode covers the rest.
  turbo_ocr::server::ImageDecoder decode =
      [nvjpeg_available](const unsigned char *data, size_t len) -> cv::Mat {
    auto opencv_decode = [&]() -> cv::Mat {
      if (len > static_cast<size_t>(INT_MAX)) return {};
      return cv::imdecode(
          cv::Mat(1, static_cast<int>(len), CV_8UC1,
                  const_cast<unsigned char *>(data)),
          cv::IMREAD_COLOR);
    };
    if (len >= 2 && data[0] == 0xFF && data[1] == 0xD8) {
      if (nvjpeg_available) {
        cv::Mat img = tl_nvjpeg.decode(data, len);
        if (!img.empty()) return img;
      }
      return opencv_decode();
    }
    if (FastPngDecoder::is_png(data, len))
      return FastPngDecoder::decode(data, len);
    return opencv_decode();
  };

  // Inference function for shared routes (/ocr base64)
  const bool layout_available = !layout_model.empty();
  turbo_ocr::server::InferFunc infer =
      [&dispatcher](const cv::Mat &img,
                    const turbo_ocr::server::InferOptions &opts)
          -> turbo_ocr::server::InferResult {
    auto out = dispatcher->submit([&img, &opts](auto &e) {
      return e.pipeline->run_with_layout(img, e.stream, opts.want_layout,
                                          opts.want_reading_order);
    }).get();
    return turbo_ocr::server::InferResult{
        .results       = std::move(out.results),
        .layout        = std::move(out.layout),
        .reading_order = std::move(out.reading_order),
    };
  };

  // Work pool for offloading blocking inference from Drogon event loop
  int work_threads = std::max(pool_size * 32, 128);
  if (const char *env = std::getenv("HTTP_THREADS"))
    work_threads = std::max(1, std::atoi(env));
  turbo_ocr::server::WorkPool work_pool(work_threads);

  // --- Register all routes ---
  turbo_ocr::server::Metrics::instance().set_pool_size(pool_size);
  turbo_ocr::server::register_observability_middleware();
  turbo_ocr::server::register_metrics_route();
  // Single readiness probe shared by HTTP /health/ready and gRPC Health
  // so k8s probes behave identically across protocols.
  //
  // The probe runs a tiny real inference (48x48 dummy) through the
  // pipeline so a corrupt TRT engine, missing layout, or wedged stream
  // surfaces here instead of on the first real client request. To keep
  // probe traffic from eating GPU, we cache the result for 5 seconds —
  // probes within that window get the cached verdict without GPU work.
  struct ProbeState {
    std::mutex mu;
    std::atomic<int64_t> last_check_ms{0};
    std::atomic<bool> ok{false};
  };
  auto probe = std::make_shared<ProbeState>();
  auto readiness = [&dispatcher, probe, layout_available]() -> bool {
    using namespace std::chrono;
    const int64_t now_ms = duration_cast<milliseconds>(
        steady_clock::now().time_since_epoch()).count();
    if (now_ms - probe->last_check_ms.load(std::memory_order_acquire) < 5000)
      return probe->ok.load(std::memory_order_acquire);

    std::lock_guard lock(probe->mu);
    // Recheck under lock — another thread may have refreshed the cache
    // while we waited. Avoids stampedes during probe spikes.
    if (now_ms - probe->last_check_ms.load(std::memory_order_acquire) < 5000)
      return probe->ok.load(std::memory_order_acquire);

    bool ok = false;
    try {
      dispatcher->submit([layout_available](auto &e) {
        cv::Mat dummy(48, 48, CV_8UC3, cv::Scalar(255, 255, 255));
        (void)e.pipeline->run_with_layout(dummy, e.stream,
                                          /*want_layout=*/layout_available,
                                          /*want_reading_order=*/false);
      }).get();
      ok = true;
    } catch (...) {}
    probe->ok.store(ok, std::memory_order_release);
    probe->last_check_ms.store(now_ms, std::memory_order_release);
    return ok;
  };
  turbo_ocr::routes::register_health_route(readiness);
  turbo_ocr::routes::register_ocr_base64_route(work_pool, infer, decode, layout_available);
  turbo_ocr::routes::register_image_routes(work_pool, *dispatcher, decode, nvjpeg_available, layout_available);
  turbo_ocr::routes::register_pdf_route(work_pool, *dispatcher, pdf_renderer, default_pdf_mode, layout_available);

  // gRPC
  int grpc_port = 50051;
  if (const char *env = std::getenv("GRPC_PORT"))
    grpc_port = std::max(1, std::atoi(env));
  auto grpc_handle = turbo_ocr::server::start_grpc_server(
      *dispatcher, grpc_port, &pdf_renderer, default_pdf_mode, layout_available,
      readiness);

  // HTTP (Drogon) — behind nginx (port 8000), direct access on 8080
  int port = 8080;
  if (const char *env = std::getenv("PORT"))
    port = std::max(1, std::atoi(env));
  int io_threads = std::max(pool_size, 4);

  // MAX_BODY_MB caps the largest accepted upload (default 100). Same env
  // var the entrypoint uses to render the nginx body cap, so the two
  // layers always agree. MAX_BODY_MEMORY_MB controls how much of each
  // body Drogon buffers in memory before spilling to a temp file —
  // larger values cut /tmp I/O at the cost of letting RSS grow with
  // concurrent connections. Default 1024 MB (1 GiB) effectively keeps
  // every body in RAM until MAX_BODY_MB is raised above 1 GiB, since
  // the memory cap is clamped to the total cap below. Lower it on
  // memory-constrained hosts (e.g. MAX_BODY_MEMORY_MB=50 caps body
  // buffer RSS at ~50 MB × concurrent_requests).
  int max_body_mb = turbo_ocr::server::env_int("MAX_BODY_MB", 100, 1, 102400);
  int max_body_mem_mb = turbo_ocr::server::env_int(
      "MAX_BODY_MEMORY_MB", 1024, 1, 102400);
  if (max_body_mem_mb > max_body_mb) max_body_mem_mb = max_body_mb;
  size_t max_body_bytes = static_cast<size_t>(max_body_mb) * 1024 * 1024;
  size_t max_mem_bytes  = static_cast<size_t>(max_body_mem_mb) * 1024 * 1024;

  TOCR_LOG_INFO("HTTP server starting", "port", port, "io_threads", io_threads,
           "work_threads", work_threads, "pool_size", dispatcher->worker_count(),
           "body_cap_mb_drogon", max_body_mb, "body_cap_mb_nginx", max_body_mb,
           "body_mem_mb", max_body_mem_mb);

  // Graceful shutdown on SIGTERM (Docker / K8s) and SIGINT (Ctrl-C):
  // drain WorkPool inflight up to SHUTDOWN_GRACE_SECONDS before quit().
  g_work_pool_for_drain = &work_pool;
  drogon::app()
      .setTermSignalHandler([] { begin_graceful_shutdown("SIGTERM"); })
      .setIntSignalHandler([]  { begin_graceful_shutdown("SIGINT");  })
      .addListener("0.0.0.0", port)
      .setThreadNum(io_threads)
      .setIdleConnectionTimeout(120)
      .setClientMaxBodySize(max_body_bytes)
      .setClientMaxMemoryBodySize(max_mem_bytes)
      .run();

  TOCR_LOG_INFO("HTTP server stopped, shutting down gRPC");
  grpc_handle.server->Shutdown();
  TOCR_LOG_INFO("Shutdown complete");
  return 0;
}
