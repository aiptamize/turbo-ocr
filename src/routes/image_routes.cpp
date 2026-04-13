#include "turbo_ocr/routes/image_routes.h"

#include <format>
#include <iostream>

#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include "turbo_ocr/common/serialization.h"
#include "turbo_ocr/decode/nvjpeg_decoder.h"

using turbo_ocr::OCRResultItem;
using turbo_ocr::base64_decode;
using turbo_ocr::results_to_json;
using turbo_ocr::decode::NvJpegDecoder;

namespace turbo_ocr::routes {

void register_image_routes(crow::SimpleApp &app,
                           pipeline::PipelineDispatcher &dispatcher,
                           const server::ImageDecoder &decode,
                           bool nvjpeg_available,
                           bool layout_available) {

  // --- /ocr/raw: GPU-direct JPEG decode, Wuffs PNG ---
  CROW_ROUTE(app, "/ocr/raw")
      .methods(crow::HTTPMethod::Post)(
          [&dispatcher, &decode, nvjpeg_available, layout_available](const crow::request &req) {

    if (req.body.empty())
      return crow::response(400, "Empty body");

    bool want_layout = false;
    if (auto err = server::parse_layout_query(req, layout_available, &want_layout); !err.empty())
      return crow::response(400, err);

    const auto *data = reinterpret_cast<const unsigned char *>(req.body.data());
    size_t len = req.body.size();

    try {
      // JPEG with nvJPEG: submit GPU-direct decode + infer as one work item
      if (nvjpeg_available && NvJpegDecoder::is_jpeg(data, len)) {
        auto out = dispatcher.submit([data, len, want_layout](auto &e) {
          thread_local NvJpegDecoder nvjpeg;
          auto [w, h] = nvjpeg.get_dimensions(data, len);
          if (w > 0 && h > 0) {
            auto [d_buf, pitch] = e.pipeline->ensure_gpu_buf(h, w);
            if (nvjpeg.decode_to_gpu(data, len, d_buf, pitch, w, h, e.stream)) {
              turbo_ocr::GpuImage gpu_img{.data = d_buf, .step = pitch, .rows = h, .cols = w};
              try {
                return e.pipeline->run_with_layout(gpu_img, e.stream, want_layout);
              } catch (const std::exception &) {
                // GPU-direct path failed (e.g. invalid pitch for small images).
                // Fall through to CPU decode below.
              }
            }
          }
          cv::Mat img = nvjpeg.decode(data, len);
          if (img.empty()) {
            if (len <= static_cast<size_t>(INT_MAX))
              img = cv::imdecode(
                  cv::Mat(1, static_cast<int>(len), CV_8UC1,
                          const_cast<unsigned char *>(data)),
                  cv::IMREAD_COLOR);
          }
          if (img.empty())
            throw turbo_ocr::ImageDecodeError("Failed to decode JPEG");
          return e.pipeline->run_with_layout(img, e.stream, want_layout);
        }).get();
        auto json_str = results_to_json(out.results, out.layout);
        auto resp = crow::response(200, std::move(json_str));
        resp.set_header("Content-Type", "application/json");
        return resp;
      }

      // Non-JPEG (PNG, etc.) or nvJPEG not available
      cv::Mat img = decode(data, len);
      if (img.empty())
        return crow::response(400, "Failed to decode image");

      auto out = dispatcher.submit([&img, want_layout](auto &e) {
        return e.pipeline->run_with_layout(img, e.stream, want_layout);
      }).get();
      auto json_str = results_to_json(out.results, out.layout);
      auto resp = crow::response(200, std::move(json_str));
      resp.set_header("Content-Type", "application/json");
      return resp;
    } catch (const turbo_ocr::ImageDecodeError &) {
      return crow::response(400, "Failed to decode image");
    } catch (const std::exception &e) {
      std::cerr << std::format("[/ocr/raw] Inference error: {}\n", e.what());
      return crow::response(500, "Inference error");
    } catch (...) {
      std::cerr << "[/ocr/raw] Inference error: unknown exception\n";
      return crow::response(500, "Inference error");
    }
  });

  // --- /ocr/batch: nvJPEG batch decode + parallel pipeline ---
  CROW_ROUTE(app, "/ocr/batch")
      .methods(crow::HTTPMethod::Post)(
          [&dispatcher, &decode, nvjpeg_available, layout_available](const crow::request &req) {
    bool want_layout = false;
    if (auto err = server::parse_layout_query(req, layout_available, &want_layout); !err.empty())
      return crow::response(400, err);
    if (want_layout)
      return crow::response(400,
          "Layout is not supported on /ocr/batch in v1 — use /ocr or /ocr/raw "
          "with ?layout=1, or /ocr/pdf?layout=1.");

    auto x = crow::json::load(req.body);
    if (!x)
      return crow::response(400, "Invalid JSON");
    if (!x.has("images"))
      return crow::response(400, "Missing images array");

    auto &images_json = x["images"];
    size_t n = images_json.size();
    if (n == 0)
      return crow::response(400, "Empty images array");

    std::vector<std::string> raw_bytes(n);
    for (size_t i = 0; i < n; ++i)
      raw_bytes[i] = base64_decode(images_json[i].s());

    std::vector<cv::Mat> imgs(n);
    std::vector<size_t> jpeg_indices;
    std::vector<std::pair<const unsigned char *, size_t>> jpeg_buffers;

    if (nvjpeg_available) {
      for (size_t i = 0; i < n; ++i) {
        auto &raw = raw_bytes[i];
        if (raw.size() >= 2 &&
            static_cast<unsigned char>(raw[0]) == 0xFF &&
            static_cast<unsigned char>(raw[1]) == 0xD8) {
          jpeg_indices.push_back(i);
          jpeg_buffers.emplace_back(
              reinterpret_cast<const unsigned char *>(raw.data()), raw.size());
        }
      }
    }

    if (jpeg_buffers.size() >= 2) {
      thread_local NvJpegDecoder tl_nvjpeg;
      auto batch_mats = tl_nvjpeg.batch_decode(jpeg_buffers);
      for (size_t j = 0; j < jpeg_indices.size(); ++j)
        imgs[jpeg_indices[j]] = std::move(batch_mats[j]);
    }

    for (size_t i = 0; i < n; ++i) {
      if (!imgs[i].empty()) continue;
      auto &raw = raw_bytes[i];
      if (raw.empty()) continue;
      imgs[i] = decode(
          reinterpret_cast<const unsigned char *>(raw.data()), raw.size());
    }

    std::vector<cv::Mat> valid_imgs;
    std::vector<size_t> valid_indices;
    valid_imgs.reserve(n);
    valid_indices.reserve(n);
    for (size_t i = 0; i < n; ++i) {
      if (!imgs[i].empty()) {
        valid_imgs.push_back(std::move(imgs[i]));
        valid_indices.push_back(i);
      }
    }
    if (valid_imgs.empty())
      return crow::response(400, "No valid images");

    constexpr int kMaxBatch = 8;
    std::vector<std::vector<OCRResultItem>> all_results(n);

    try {
      dispatcher.submit([&](auto &e) {
        for (size_t offset = 0; offset < valid_imgs.size(); offset += kMaxBatch) {
          size_t end = std::min(offset + kMaxBatch, valid_imgs.size());
          std::vector<cv::Mat> chunk(
              std::make_move_iterator(valid_imgs.begin() + offset),
              std::make_move_iterator(valid_imgs.begin() + end));
          auto chunk_results = e.pipeline->run_batch(chunk, e.stream);
          for (size_t j = 0; j < chunk_results.size(); ++j)
            all_results[valid_indices[offset + j]] = std::move(chunk_results[j]);
        }
      }).get();
    } catch (const std::exception &e) {
      std::cerr << std::format("[Batch] run_batch error: {}", e.what()) << '\n';
      return crow::response(500, "Inference error");
    } catch (...) {
      std::cerr << "[Batch] run_batch error: unknown exception" << '\n';
      return crow::response(500, "Inference error");
    }

    std::string json_str;
    json_str.reserve(n * 1024);
    json_str += "{\"batch_results\":[";
    for (size_t i = 0; i < n; ++i) {
      if (i > 0) json_str += ',';
      json_str += results_to_json(all_results[i]);
    }
    json_str += "]}";
    auto resp = crow::response(200, std::move(json_str));
    resp.set_header("Content-Type", "application/json");
    return resp;
  });

  // --- /ocr/pixels: raw BGR pixel data, zero decode overhead ---
  CROW_ROUTE(app, "/ocr/pixels")
      .methods(crow::HTTPMethod::Post)(
          [&dispatcher, layout_available](const crow::request &req) {
    bool want_layout = false;
    if (auto err = server::parse_layout_query(req, layout_available, &want_layout); !err.empty())
      return crow::response(400, err);

    auto w_str = req.get_header_value("X-Width");
    auto h_str = req.get_header_value("X-Height");
    auto c_str = req.get_header_value("X-Channels");

    if (w_str.empty() || h_str.empty())
      return crow::response(400, "Missing X-Width or X-Height headers");

    int width, height, channels;
    try {
      width = std::stoi(w_str);
      height = std::stoi(h_str);
      channels = c_str.empty() ? 3 : std::stoi(c_str);
    } catch (const std::exception &) {
      return crow::response(400, "Invalid X-Width, X-Height, or X-Channels header value");
    }

    if (width <= 0 || height <= 0 || (channels != 1 && channels != 3))
      return crow::response(400, "Invalid dimensions or channels");

    constexpr int kMaxPixelDim = 16384;
    if (width > kMaxPixelDim || height > kMaxPixelDim)
      return crow::response(400, std::format("Dimensions {}x{} exceed maximum of {}x{}", width, height, kMaxPixelDim, kMaxPixelDim));

    size_t expected = static_cast<size_t>(width) * height * channels;
    if (req.body.size() != expected)
      return crow::response(400,
          std::format("Body size mismatch: expected {} bytes ({}x{}x{}), got {}",
                      expected, width, height, channels, req.body.size()));

    cv::Mat img(height, width, channels == 3 ? CV_8UC3 : CV_8UC1,
                const_cast<char *>(req.body.data()));

    try {
      auto out = dispatcher.submit([&img, want_layout](auto &e) {
        return e.pipeline->run_with_layout(img, e.stream, want_layout);
      }).get();
      auto json_str = results_to_json(out.results, out.layout);
      auto resp = crow::response(200, std::move(json_str));
      resp.set_header("Content-Type", "application/json");
      return resp;
    } catch (const std::exception &e) {
      std::cerr << std::format("[/ocr/pixels] Inference error: {}\n", e.what());
      return crow::response(500, "Inference error");
    } catch (...) {
      std::cerr << "[/ocr/pixels] Inference error: unknown exception\n";
      return crow::response(500, "Inference error");
    }
  });
}

} // namespace turbo_ocr::routes
