#include "turbo_ocr/routes/pdf_routes.h"

#include <format>
#include <future>

#include "turbo_ocr/common/logger.h"

#ifndef USE_CPU_ONLY
#include "turbo_ocr/pipeline/pipeline_dispatcher.h"
#endif
#include <mutex>

#include <opencv2/core.hpp>

#include <drogon/HttpAppFramework.h>
#include <drogon/utils/Utilities.h>
#include <json/json.h>

#include "turbo_ocr/common/serialization.h"
#include "turbo_ocr/pdf/pdf_text_layer.h"
#include "turbo_ocr/server/server_types.h"

using turbo_ocr::OCRResultItem;
using turbo_ocr::results_to_json;

namespace turbo_ocr::routes {

namespace {

int max_pdf_pages() {
  static int val = [] {
    if (auto *env = std::getenv("MAX_PDF_PAGES"))
      return std::max(1, std::atoi(env));
    return 2000;
  }();
  return val;
}

// ---------------------------------------------------------------------------
// PDF text-layer helpers — shared by CPU and GPU /ocr/pdf routes.
//
// `geometric` mode reads the PDF's native text layer (extracted by pdfium)
// directly — no rendering, no OCR. `auto` falls back to OCR only when the
// text layer is unusable (image-only PDF, garbage encoding, rotated page).
// `text_layer_quality_for` decides whether the layer is trustworthy:
//   absent   — no chars, no lines, or fewer than 10 chars (looks scanned)
//   rejected — non-zero rotation, too many U+FFFD replacement chars,
//              too many non-printable characters
//   trusted  — usable text we can return without OCRing the page
// `fill_from_text_layer_pt` copies that text + bounding boxes into the
// page result at point coordinates (DPI=72); the caller scales to pixel
// coordinates only when the resolved mode is geometric (so layout output,
// which is in pixel space, doesn't get rescaled twice).
struct PdfPageResultBase {
  std::vector<OCRResultItem> results;
  std::vector<layout::LayoutBox> layout;
  std::vector<int> reading_order;
  int width = 0, height = 0, effective_dpi = 0;
  pdf::PdfMode resolved_mode = pdf::PdfMode::Ocr;
  std::string_view text_layer_quality = "absent";
};

template <typename PageResult>
void fill_from_text_layer_pt(PageResult &pg, const pdf::PdfPageText &text) {
  pg.width  = static_cast<int>(std::round(text.page_width_pt));
  pg.height = static_cast<int>(std::round(text.page_height_pt));
  pg.effective_dpi = 72;
  pg.results.reserve(text.lines.size());
  for (const auto &line : text.lines) {
    OCRResultItem item;
    item.source = "pdf";
    item.confidence = 1.0f;
    item.text = line.text;
    int ix0 = static_cast<int>(std::round(line.x0_pt));
    int iy0 = static_cast<int>(std::round(line.y0_pt));
    int ix1 = static_cast<int>(std::round(line.x1_pt));
    int iy1 = static_cast<int>(std::round(line.y1_pt));
    item.box[0] = {ix0, iy0};
    item.box[1] = {ix1, iy0};
    item.box[2] = {ix1, iy1};
    item.box[3] = {ix0, iy1};
    pg.results.push_back(std::move(item));
  }
}

std::string_view text_layer_quality_for(const pdf::PdfPageText &text) {
  if (text.char_count == 0)         return "absent";
  if (text.rotation_deg != 0)       return "rejected";
  if (text.char_count < 10)         return "absent";
  if (text.fffd_count * 20 > text.char_count)     return "rejected";
  if (text.nonprint_count * 10 > text.char_count) return "rejected";
  if (text.lines.empty())           return "absent";
  return "trusted";
}

/// Helper: extract PDF bytes from a Drogon request (raw, base64 JSON, multipart).
/// Returns true on success, fills pdf_ptr/pdf_len and may fill decoded_buf.
/// On failure, calls cb with 400 and returns false.
bool extract_pdf_bytes(const drogon::HttpRequestPtr &req,
                       std::string &decoded_buf,
                       const char *&pdf_ptr, size_t &pdf_len,
                       const std::function<void(const drogon::HttpResponsePtr &)> &cb) {
  auto ct = req->getHeader("Content-Type");
  if (ct.find("multipart/form-data") != std::string::npos) {
    drogon::MultiPartParser parser;
    if (parser.parse(req) != 0) {
      cb(server::error_response(drogon::k400BadRequest, "INVALID_MULTIPART", "Failed to parse multipart body"));
      return false;
    }
    for (auto &file : parser.getFiles()) {
      auto name = file.getItemName();
      if (name == "file" || name == "pdf") {
        decoded_buf.assign(file.fileData(), file.fileLength());
        break;
      }
    }
    if (decoded_buf.empty()) {
      cb(server::error_response(drogon::k400BadRequest, "MISSING_FILE",
          "Multipart request must contain a 'file' or 'pdf' form field"));
      return false;
    }
    pdf_ptr = decoded_buf.data();
    pdf_len = decoded_buf.size();
  } else if (ct.find("application/json") != std::string::npos) {
    auto json = req->getJsonObject();
    if (!json || !json->isMember("pdf")) {
      cb(server::error_response(drogon::k400BadRequest, "MISSING_PDF",
          R"(JSON body must contain {"pdf": "<base64>"})"));
      return false;
    }
    auto b64 = (*json)["pdf"].asString();
    decoded_buf = turbo_ocr::base64_decode(b64);
    if (decoded_buf.empty()) {
      cb(server::error_response(drogon::k400BadRequest, "BASE64_DECODE_FAILED", "Failed to decode base64 PDF"));
      return false;
    }
    pdf_ptr = decoded_buf.data();
    pdf_len = decoded_buf.size();
  } else {
    if (req->body().empty()) {
      cb(server::error_response(drogon::k400BadRequest, "EMPTY_BODY", "Empty body"));
      return false;
    }
    pdf_ptr = req->body().data();
    pdf_len = req->body().size();
  }
  return true;
}

// Common request-time params we parse off the wire: layout flag, dpi, mode.
// Both GPU and CPU paths share the same query-string contract.
struct PdfRequestParams {
  bool layout_enabled = false;
  int dpi = 100;
  pdf::PdfMode mode;
};

// Parse and validate /ocr/pdf query params. Calls callback with a 400 on
// any validation failure and returns false. Caller must `return` on false.
bool parse_pdf_request_params(
    const drogon::HttpRequestPtr &req,
    bool layout_available,
    pdf::PdfMode default_pdf_mode,
    PdfRequestParams &out,
    const std::function<void(const drogon::HttpResponsePtr &)> &callback) {
  if (auto err = server::parse_layout_query(req, layout_available,
                                             &out.layout_enabled);
      !err.empty()) {
    callback(server::error_response(drogon::k400BadRequest, "INVALID_PARAMETER", err));
    return false;
  }

  out.dpi = 100;
  auto dpi_str = req->getParameter("dpi");
  if (!dpi_str.empty()) out.dpi = std::atoi(dpi_str.c_str());
  if (out.dpi < 50 || out.dpi > 600) {
    callback(server::error_response(drogon::k400BadRequest, "INVALID_DPI",
        "Image DPI must be between 50 and 600"));
    return false;
  }

  out.mode = default_pdf_mode;
  auto mode_str = req->getParameter("mode");
  if (!mode_str.empty())
    out.mode = pdf::parse_pdf_mode(mode_str.c_str(), default_pdf_mode);
  return true;
}

// Inline page-count guard: emits PDF_TOO_LARGE if the document exceeds
// MAX_PDF_PAGES. Returns true on guard-trip (caller should abort).
bool reject_if_too_many_pages(const uint8_t *pdf_data, size_t pdf_len_local,
                               server::DrogonCallback &cb) {
  pdf::PdfDocument check_doc(pdf_data, pdf_len_local);
  if (!check_doc.ok()) return false;
  int np = check_doc.page_count();
  int limit = max_pdf_pages();
  if (np > limit) {
    cb(server::error_response(drogon::k400BadRequest, "PDF_TOO_LARGE",
        std::format("PDF has {} pages, maximum is {} (set MAX_PDF_PAGES to increase)",
                    np, limit)));
    return true;
  }
  return false;
}

// Open the PDF and pre-extract per-page text only when the chosen mode
// actually needs the text layer. mode=ocr skips this. On open failure we
// downgrade to mode=ocr and clear the doc.
void open_pdf_for_text_layer(const uint8_t *pdf_data, size_t pdf_len_local,
                              pdf::PdfMode &mode,
                              std::unique_ptr<pdf::PdfDocument> &pdf_doc,
                              std::vector<pdf::PdfPageText> &page_text_cache) {
  if (mode == pdf::PdfMode::Ocr) return;
  pdf_doc = std::make_unique<pdf::PdfDocument>(pdf_data, pdf_len_local);
  if (!pdf_doc->ok()) {
    TOCR_LOG_WARN("Failed to open PDF for text-layer lookup; falling back to mode=ocr",
                  "route", "/ocr/pdf");
    mode = pdf::PdfMode::Ocr;
    pdf_doc.reset();
    return;
  }
  int np = pdf_doc->page_count();
  page_text_cache.reserve(static_cast<size_t>(std::max(0, np)));
  for (int p = 0; p < np; ++p)
    page_text_cache.push_back(pdf_doc->extract_page(p));
}

// Decide per-page resolved_mode and whether each page needs rendering,
// based on text-layer quality. mode=ocr always renders, so this is only
// called for the non-ocr modes. AutoVerified is GPU-only — on CPU it's
// aliased to Auto before this is called.
template <typename PageResult>
void prepopulate_pages(pdf::PdfMode mode,
                       bool layout_or_want_layout,
                       const std::vector<pdf::PdfPageText> &page_text_cache,
                       std::vector<PageResult> &page_results,
                       std::vector<uint8_t> &need_render,
                       bool *any_need_render) {
  int np = static_cast<int>(page_text_cache.size());
  page_results.resize(static_cast<size_t>(np));
  need_render.assign(static_cast<size_t>(np), 0);

  for (int p = 0; p < np; ++p) {
    const auto &text = page_text_cache[static_cast<size_t>(p)];
    auto &pg = page_results[static_cast<size_t>(p)];
    pg.text_layer_quality = text_layer_quality_for(text);
    bool has_good_layer = (pg.text_layer_quality == "trusted");

    switch (mode) {
      case pdf::PdfMode::Geometric:
        pg.resolved_mode = pdf::PdfMode::Geometric;
        if (has_good_layer) {
          fill_from_text_layer_pt(pg, text);
        } else {
          pg.width = static_cast<int>(std::round(text.page_width_pt));
          pg.height = static_cast<int>(std::round(text.page_height_pt));
          pg.effective_dpi = 72;
        }
        if (layout_or_want_layout) {
          need_render[static_cast<size_t>(p)] = 1;
          if (any_need_render) *any_need_render = true;
        }
        break;
      case pdf::PdfMode::Auto:
        if (has_good_layer) {
          pg.resolved_mode = pdf::PdfMode::Geometric;
          fill_from_text_layer_pt(pg, text);
          if (layout_or_want_layout) {
            need_render[static_cast<size_t>(p)] = 1;
            if (any_need_render) *any_need_render = true;
          }
        } else {
          pg.resolved_mode = pdf::PdfMode::Ocr;
          need_render[static_cast<size_t>(p)] = 1;
          if (any_need_render) *any_need_render = true;
        }
        break;
      case pdf::PdfMode::AutoVerified:
        pg.resolved_mode = pdf::PdfMode::AutoVerified;
        need_render[static_cast<size_t>(p)] = 1;
        if (any_need_render) *any_need_render = true;
        break;
      default: break;
    }
  }
}

// Build the final {pages: [...]} JSON. Shared by CPU and GPU paths. The
// per-result + per-page byte estimate keeps dense pages from reallocating
// and tiny pages from over-allocating.
template <typename PageResult>
std::string emit_pdf_response(std::vector<PageResult> &page_results,
                               int request_dpi,
                               bool want_blocks = false) {
  size_t n_pages = page_results.size();
  size_t total_results = 0;
  for (size_t i = 0; i < n_pages; ++i)
    total_results += page_results[i].results.size() + page_results[i].layout.size();
  std::string json_str;
  json_str.reserve(total_results * 256 + n_pages * 256 + 64);
  json_str += "{\"pages\":[";
  for (size_t i = 0; i < n_pages; ++i) {
    if (i > 0) json_str += ',';
    auto &pg = page_results[i];
    int page_dpi = pg.effective_dpi > 0 ? pg.effective_dpi : request_dpi;
    json_str += "{\"page\":";
    json_str += std::to_string(i + 1);
    json_str += ",\"page_index\":";
    json_str += std::to_string(i);
    json_str += ",\"dpi\":";
    json_str += std::to_string(page_dpi);
    json_str += ",\"width\":";
    json_str += std::to_string(pg.width);
    json_str += ",\"height\":";
    json_str += std::to_string(pg.height);
    json_str += ',';
    auto page_json = !pg.reading_order.empty()
                         ? emit_results_json(pg.results, pg.layout,
                                              pg.reading_order, want_blocks)
                         : results_to_json(pg.results, pg.layout);
    json_str.append(page_json.data() + 1, page_json.size() - 2);
    json_str += ",\"mode\":\"";
    json_str += pdf::mode_name(pg.resolved_mode);
    json_str += "\",\"text_layer_quality\":\"";
    json_str += pg.text_layer_quality;
    json_str += "\"}";
  }
  json_str += "]}";
  return json_str;
}

#ifndef USE_CPU_ONLY
// GPU page-result type carries the same fields as the base. The
// per-page-future render loop resolves each rendered page on the dispatcher
// thread pool and writes back into this shared vector under results_mutex.
struct GpuPdfPageResult : public PdfPageResultBase {};

// GPU streamed render callback: decodes PPM, runs layout-only or full
// pipeline + AutoVerified verification depending on resolved mode, and
// writes results into `page_results[page_idx]` under `results_mutex`. The
// callback is invoked from inside pdf_renderer.render_streamed().
void run_streamed_render_gpu(
    pipeline::PipelineDispatcher &dispatcher,
    render::PdfRenderer &pdf_renderer,
    const uint8_t *pdf_data, size_t pdf_len_local,
    int dpi, bool layout_enabled, bool want_reading_order,
    pdf::PdfMode mode,
    pdf::PdfDocument *pdf_doc,
    const std::vector<pdf::PdfPageText> &page_text_cache,
    std::mutex &results_mutex,
    std::vector<GpuPdfPageResult> &page_results,
    std::vector<uint8_t> &need_render,
    std::vector<std::future<void>> &page_futures,
    std::mutex &futures_mutex,
    int &num_pages_out) {
  auto stream_handle = pdf_renderer.render_streamed(pdf_data, pdf_len_local, dpi,
      [&](int page_idx, std::string ppm_path) {
        {
          std::lock_guard<std::mutex> rlock(results_mutex);
          if (page_idx >= static_cast<int>(page_results.size())) {
            page_results.resize(page_idx + 1);
            if (mode != pdf::PdfMode::Ocr &&
                page_idx >= static_cast<int>(need_render.size()))
              need_render.resize(page_idx + 1, 1);
          }
          if (mode != pdf::PdfMode::Ocr &&
              page_idx < static_cast<int>(need_render.size()) &&
              !need_render[page_idx])
            return;
        }

        std::future<void> fut;
        try {
          fut = dispatcher.submit(
              [&, page_idx, path = std::move(ppm_path)](auto &e) {
            cv::Mat img = render::PdfRenderer::decode_ppm(path);
            if (img.empty()) {
              TOCR_LOG_ERROR("Failed to decode PPM for page", "route", "/ocr/pdf", "page", page_idx);
              return;
            }
            int pw = img.cols, ph = img.rows;

            pdf::PdfMode page_mode;
            {
              std::lock_guard<std::mutex> rlock(results_mutex);
              page_mode = (page_idx < static_cast<int>(page_results.size()))
                  ? page_results[page_idx].resolved_mode
                  : pdf::PdfMode::Ocr;
            }

            std::vector<OCRResultItem> rec_results;
            std::vector<layout::LayoutBox> layout_snapshot;

            if (page_mode == pdf::PdfMode::Geometric) {
              auto lo = e.pipeline->run_layout_only(img, e.stream);
              layout_snapshot = std::move(lo.layout);
            } else {
              auto pipeline_out = e.pipeline->run_with_layout(
                  img, e.stream, layout_enabled, want_reading_order);
              rec_results = std::move(pipeline_out.results);
              layout_snapshot = std::move(pipeline_out.layout);
              for (auto &it : rec_results) it.source = "ocr";
              if (want_reading_order) {
                std::lock_guard<std::mutex> rlock(results_mutex);
                page_results[page_idx].reading_order =
                    std::move(pipeline_out.reading_order);
              }
            }

            if (page_mode == pdf::PdfMode::AutoVerified &&
                page_idx < static_cast<int>(page_text_cache.size()) && pdf_doc) {
              for (auto &item : rec_results) {
                const float px_to_pt = 72.0f / static_cast<float>(dpi);
                auto [ix0, iy0, ix1, iy1] = turbo_ocr::aabb(item.box);
                float x0 = ix0 * px_to_pt, y0 = iy0 * px_to_pt;
                float x1 = ix1 * px_to_pt, y1 = iy1 * px_to_pt;
                std::string native =
                    pdf_doc->text_in_rect_pt(page_idx, x0, y0, x1, y1);
                auto verdict = pdf::passes_sanity_check(
                    native, x1 - x0, y1 - y0);
                if (verdict.accept) {
                  item.text = std::move(native);
                  item.source = "pdf";
                  item.confidence = 1.0f;
                }
              }
            }

            std::lock_guard<std::mutex> rlock(results_mutex);
            auto &slot = page_results[page_idx];
            if (page_mode == pdf::PdfMode::Geometric) {
              const float pt_to_px = static_cast<float>(dpi) / 72.0f;
              for (auto &item : slot.results) {
                for (int k = 0; k < 4; ++k) {
                  item.box[k][0] = static_cast<int>(
                      std::round(item.box[k][0] * pt_to_px));
                  item.box[k][1] = static_cast<int>(
                      std::round(item.box[k][1] * pt_to_px));
                }
              }
            } else {
              slot.results = std::move(rec_results);
            }
            slot.layout        = std::move(layout_snapshot);
            slot.width         = pw;
            slot.height        = ph;
            slot.effective_dpi = dpi;
            if (page_mode == pdf::PdfMode::Ocr)
              slot.resolved_mode = pdf::PdfMode::Ocr;
          });
        } catch (const turbo_ocr::PoolExhaustedError &) {
          TOCR_LOG_WARN("GPU queue full, skipping page", "route", "/ocr/pdf", "page", page_idx);
          return;
        }
        std::lock_guard lock(futures_mutex);
        page_futures.push_back(std::move(fut));
      });
  num_pages_out = stream_handle.num_pages;
}
#endif // !USE_CPU_ONLY

// CPU streamed render callback. Sequential: decode PPM, run the InferFunc
// inline. mode==Ocr means we never visited prepopulate_pages, so resolved
// mode is pinned here. Geometric pages keep their layer-derived text and
// just rescale point→pixel coords.
int run_streamed_render_cpu(
    const server::InferFunc &infer,
    render::PdfRenderer &pdf_renderer,
    const uint8_t *pdf_data, size_t pdf_len_local,
    int dpi, bool want_layout, bool want_reading_order, pdf::PdfMode mode,
    std::vector<PdfPageResultBase> &page_results,
    const std::vector<uint8_t> &need_render) {
  auto stream_handle = pdf_renderer.render_streamed(pdf_data, pdf_len_local, dpi,
      [&](int page_idx, std::string ppm_path) {
        if (mode != pdf::PdfMode::Ocr &&
            page_idx < static_cast<int>(need_render.size()) &&
            !need_render[static_cast<size_t>(page_idx)])
          return;

        cv::Mat img = render::PdfRenderer::decode_ppm(ppm_path);
        if (img.empty()) return;

        if (page_idx >= static_cast<int>(page_results.size()))
          page_results.resize(page_idx + 1);
        auto &pg = page_results[static_cast<size_t>(page_idx)];

        turbo_ocr::server::InferOptions inf_opts;
        inf_opts.want_layout = want_layout;
        inf_opts.want_reading_order = want_reading_order;
        // mode==Ocr means the per-page resolved_mode wasn't set up
        // front (we skipped the text-layer pre-pass entirely), so
        // pin it to Ocr here. For non-ocr modes pg.resolved_mode is
        // already set per page.
        if (mode == pdf::PdfMode::Ocr)
          pg.resolved_mode = pdf::PdfMode::Ocr;

        if (pg.resolved_mode == pdf::PdfMode::Geometric) {
          // Text already filled from layer in pt-space; only run
          // layout (when requested) and rescale text boxes from
          // points to pixel space matching the rendered image.
          if (want_layout) {
            auto inf = infer(img, inf_opts);
            pg.layout = std::move(inf.layout);
          }
          pg.width = img.cols;
          pg.height = img.rows;
          pg.effective_dpi = dpi;
          const float pt_to_px = static_cast<float>(dpi) / 72.0f;
          for (auto &item : pg.results) {
            for (int k = 0; k < 4; ++k) {
              item.box[k][0] = static_cast<int>(
                  std::round(item.box[k][0] * pt_to_px));
              item.box[k][1] = static_cast<int>(
                  std::round(item.box[k][1] * pt_to_px));
            }
          }
        } else {
          // Ocr branch: full pipeline, results from rec.
          auto inf = infer(img, inf_opts);
          pg.results = std::move(inf.results);
          pg.layout = std::move(inf.layout);
          pg.reading_order = std::move(inf.reading_order);
          pg.width = img.cols;
          pg.height = img.rows;
          pg.effective_dpi = dpi;
          for (auto &item : pg.results) item.source = "ocr";
        }
      });
  return stream_handle.num_pages;
}

} // namespace

#ifndef USE_CPU_ONLY
void register_pdf_route(server::WorkPool &pool,
                        pipeline::PipelineDispatcher &dispatcher,
                        render::PdfRenderer &pdf_renderer,
                        pdf::PdfMode default_pdf_mode,
                        bool layout_available) {

  drogon::app().registerHandler(
      "/ocr/pdf",
      [&pool, &dispatcher, &pdf_renderer, default_pdf_mode, layout_available](
          const drogon::HttpRequestPtr &req,
          std::function<void(const drogon::HttpResponsePtr &)> &&callback) {

    // Extract PDF bytes (lightweight, on event loop)
    auto pdf_buf = std::make_shared<std::string>();
    const char *pdf_ptr = nullptr;
    size_t pdf_len = 0;

    if (!extract_pdf_bytes(req, *pdf_buf, pdf_ptr, pdf_len, callback))
      return;

    server::InferOptions opts;
    if (auto r = server::parse_query_options(req, layout_available, &opts);
        !r.error.empty()) {
      callback(server::error_response(drogon::k400BadRequest,
                                       r.error_code.c_str(), r.error));
      return;
    }
    const bool layout_enabled = opts.want_layout;
    const bool want_reading_order = opts.want_reading_order;
    const bool want_blocks = opts.want_blocks;

    int dpi = 100;
    auto dpi_str = req->getParameter("dpi");
    if (!dpi_str.empty()) dpi = std::atoi(dpi_str.c_str());
    if (dpi < 50 || dpi > 600) {
      callback(server::error_response(drogon::k400BadRequest, "INVALID_DPI", "DPI must be between 50 and 600"));
      return;
    }

    pdf::PdfMode req_mode = default_pdf_mode;
    auto mode_str = req->getParameter("mode");
    if (!mode_str.empty())
      req_mode = pdf::parse_pdf_mode(mode_str.c_str(), default_pdf_mode);

    // For raw body case, pdf_ptr points into req->body() — copy into pdf_buf
    if (pdf_buf->empty())
      pdf_buf->assign(pdf_ptr, pdf_len);

    server::submit_work(pool, std::move(callback),
        [pdf_buf, req, &dispatcher, &pdf_renderer,
         layout_enabled, want_reading_order, want_blocks,
         dpi, req_mode](server::DrogonCallback &cb) {
      const auto *pdf_data = reinterpret_cast<const uint8_t *>(pdf_buf->data());
      size_t pdf_len_local = pdf_buf->size();

      if (reject_if_too_many_pages(pdf_data, pdf_len_local, cb)) return;

      // Open PDF for text-layer modes
      pdf::PdfMode mode = req_mode;
      std::unique_ptr<pdf::PdfDocument> pdf_doc;
      std::vector<pdf::PdfPageText> page_text_cache;
      open_pdf_for_text_layer(pdf_data, pdf_len_local, mode,
                              pdf_doc, page_text_cache);

      // Shared state — uses the file-scope `fill_from_text_layer_pt` and
      // `text_layer_quality_for` helpers shared with the CPU route.
      std::mutex results_mutex;
      std::vector<GpuPdfPageResult> page_results;

      // Pre-populate pages that don't need rendering
      std::vector<uint8_t> need_render;
      bool any_need_render = (mode == pdf::PdfMode::Ocr);

      if (mode != pdf::PdfMode::Ocr) {
        prepopulate_pages(mode, layout_enabled, page_text_cache,
                          page_results, need_render, &any_need_render);
      }

      // Streamed render + OCR
      std::mutex futures_mutex;
      std::vector<std::future<void>> page_futures;
      int num_pages = 0;

      if (any_need_render) {
        try {
          run_streamed_render_gpu(dispatcher, pdf_renderer,
                                   pdf_data, pdf_len_local,
                                   dpi, layout_enabled, want_reading_order, mode,
                                   pdf_doc.get(), page_text_cache,
                                   results_mutex, page_results, need_render,
                                   page_futures, futures_mutex, num_pages);
        } catch (const std::exception &e) {
          for (auto &f : page_futures) { try { f.get(); } catch (...) {} }
          TOCR_LOG_ERROR("PDF render failed", "route", "/ocr/pdf", "error", std::string_view(e.what()));
          cb(server::error_response(drogon::k400BadRequest, "PDF_RENDER_FAILED", "PDF render failed"));
          return;
        }
      } else {
        num_pages = pdf_doc ? pdf_doc->page_count() : 0;
      }

      {
        std::lock_guard<std::mutex> rlock(results_mutex);
        if (static_cast<int>(page_results.size()) < num_pages)
          page_results.resize(num_pages);
      }

      for (auto &f : page_futures) {
        try { f.get(); } catch (const std::exception &e) {
          TOCR_LOG_ERROR("PDF page error", "route", "/ocr/pdf", "error", std::string_view(e.what()));
        }
      }

      if (num_pages == 0) {
        cb(server::error_response(drogon::k400BadRequest, "EMPTY_PDF", "PDF contains no pages"));
        return;
      }

      // num_pages may have grown the vector under page_futures completion;
      // trim to its actual number reported by the renderer.
      std::vector<GpuPdfPageResult> trimmed;
      {
        std::lock_guard<std::mutex> rlock(results_mutex);
        trimmed.reserve(num_pages);
        for (int i = 0; i < num_pages && i < static_cast<int>(page_results.size()); ++i)
          trimmed.push_back(std::move(page_results[i]));
      }
      cb(server::json_response(emit_pdf_response(trimmed, dpi, want_blocks)));
    });
  }, {drogon::Post});
}
#endif // !USE_CPU_ONLY

// --- CPU overload: sequential page OCR via InferFunc ---
void register_pdf_route(server::WorkPool &pool,
                        const server::InferFunc &infer,
                        render::PdfRenderer &pdf_renderer,
                        pdf::PdfMode default_pdf_mode,
                        bool layout_available) {

  drogon::app().registerHandler(
      "/ocr/pdf",
      [&pool, &infer, &pdf_renderer, default_pdf_mode, layout_available](
          const drogon::HttpRequestPtr &req,
          std::function<void(const drogon::HttpResponsePtr &)> &&callback) {

    std::string decoded_buf;
    const char *pdf_ptr = nullptr;
    size_t pdf_len = 0;

    if (!extract_pdf_bytes(req, decoded_buf, pdf_ptr, pdf_len, callback))
      return;

    server::InferOptions opts;
    if (auto r = server::parse_query_options(req, layout_available, &opts);
        !r.error.empty()) {
      callback(server::error_response(drogon::k400BadRequest,
                                       r.error_code.c_str(), r.error));
      return;
    }
    const bool want_layout = opts.want_layout;
    const bool want_reading_order = opts.want_reading_order;
    const bool want_blocks = opts.want_blocks;

    int dpi = 100;
    auto dpi_str = req->getParameter("dpi");
    if (!dpi_str.empty()) dpi = std::atoi(dpi_str.c_str());
    if (dpi < 50 || dpi > 600) {
      callback(server::error_response(drogon::k400BadRequest, "INVALID_DPI", "DPI must be between 50 and 600"));
      return;
    }

    pdf::PdfMode req_mode = default_pdf_mode;
    auto mode_str = req->getParameter("mode");
    if (!mode_str.empty())
      req_mode = pdf::parse_pdf_mode(mode_str.c_str(), default_pdf_mode);

    auto pdf_buf = std::make_shared<std::string>(pdf_ptr, pdf_len);

    server::submit_work(pool, std::move(callback),
        [pdf_buf, &infer, &pdf_renderer, want_layout,
         want_reading_order, want_blocks, dpi,
         req_mode](server::DrogonCallback &cb) {
      const auto *pdf_data = reinterpret_cast<const uint8_t *>(pdf_buf->data());
      size_t pdf_len_local = pdf_buf->size();

      if (reject_if_too_many_pages(pdf_data, pdf_len_local, cb)) return;

      // CPU server runs sequentially; the GPU AutoVerified path
      // cross-checks every OCR detection against the text layer in
      // parallel. Doing the same on CPU would require an extra pdfium
      // text_in_rect call per detection per page, doubling latency on a
      // single-thread pipeline. Honest behavior: alias auto_verified to
      // auto on CPU and emit the actually-resolved per-page mode in the
      // response, so clients who set auto_verified get auto's text-layer
      // fast-path without us claiming verification we didn't perform.
      pdf::PdfMode mode = req_mode;
      if (mode == pdf::PdfMode::AutoVerified) mode = pdf::PdfMode::Auto;

      // Open PDF for text-layer extraction when the resolved mode needs it.
      // For mode=ocr we skip this entirely (matches the legacy CPU path).
      std::unique_ptr<pdf::PdfDocument> pdf_doc;
      std::vector<pdf::PdfPageText> page_text_cache;
      open_pdf_for_text_layer(pdf_data, pdf_len_local, mode,
                              pdf_doc, page_text_cache);

      std::vector<PdfPageResultBase> page_results;
      std::vector<uint8_t> need_render;

      // Decide per-page resolved mode up front. mode=ocr always renders;
      // mode=geometric / mode=auto consult the text layer first and only
      // render when necessary.
      if (mode != pdf::PdfMode::Ocr) {
        prepopulate_pages(mode, want_layout, page_text_cache,
                          page_results, need_render, /*any_need_render=*/nullptr);
      }

      // Render + OCR pass. mode=ocr runs every page; non-ocr modes only
      // render pages that flagged need_render (image-only pages, layout=1
      // requests, auto-fallback OCR pages, auto_verified pages).
      try {
        bool any_need_render = (mode == pdf::PdfMode::Ocr) ||
            std::any_of(need_render.begin(), need_render.end(),
                        [](uint8_t v) { return v != 0; });

        if (any_need_render) {
          int num_pages = run_streamed_render_cpu(infer, pdf_renderer,
              pdf_data, pdf_len_local, dpi, want_layout,
              want_reading_order, mode,
              page_results, need_render);
          if (static_cast<int>(page_results.size()) < num_pages)
            page_results.resize(num_pages);
        }
      } catch (const std::exception &e) {
        cb(server::error_response(drogon::k400BadRequest, "PDF_RENDER_FAILED",
            std::format("PDF render failed: {}", e.what())));
        return;
      }

      if (page_results.empty()) {
        cb(server::error_response(drogon::k400BadRequest, "EMPTY_PDF",
            "PDF contains no pages"));
        return;
      }

      cb(server::json_response(emit_pdf_response(page_results, dpi, want_blocks)));
    });
  }, {drogon::Post});
}

} // namespace turbo_ocr::routes
