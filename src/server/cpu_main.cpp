#include <atomic>
#include <cstdlib>
#include <format>
#include <string>
#include <thread>
#include <vector>

#include "turbo_ocr/pipeline/cpu_pipeline_pool.h"
#include "turbo_ocr/server/env_utils.h"
#include "turbo_ocr/server/http_routes.h"

using turbo_ocr::Box;
using turbo_ocr::OCRResultItem;
using turbo_ocr::base64_decode;
using turbo_ocr::results_to_json;
using turbo_ocr::server::env_or;

int main() {
  std::cout << "=== PaddleOCR CPU-Only Mode (ONNX Runtime) ===" << '\n';

  auto det_model = env_or("DET_MODEL", "models/det.onnx");
  auto rec_model = env_or("REC_MODEL", "models/rec.onnx");
  auto rec_dict = env_or("REC_DICT", "models/keys.txt");
  auto cls_model = env_or("CLS_MODEL", "models/cls.onnx");
  if (turbo_ocr::server::env_enabled("DISABLE_ANGLE_CLS")) {
    cls_model.clear();
    std::cout << "Angle classification disabled via DISABLE_ANGLE_CLS=1"
              << '\n';
  }

  int pool_size = 4;
  if (const char *env = std::getenv("PIPELINE_POOL_SIZE"))
    pool_size = std::max(1, std::atoi(env));

  std::cout << "CPU pipeline pool size: " << pool_size << '\n';
  auto pool = turbo_ocr::pipeline::make_cpu_pipeline_pool(
      pool_size, det_model, rec_model, rec_dict, cls_model);

  // Inference function for shared routes
  turbo_ocr::server::InferFunc infer = [&pool](const cv::Mat &img) {
    auto handle = pool->acquire();
    return handle->run(img);
  };

  // Image decoder (CPU only: Wuffs for PNG, OpenCV for everything else)
  turbo_ocr::server::ImageDecoder decode = turbo_ocr::server::cpu_decode_image;

  crow::SimpleApp app;

  // Register shared routes: /health, /ocr, /ocr/raw
  turbo_ocr::server::register_common_routes(app, infer, decode);

  // --- /ocr/batch endpoint (CPU version: simple sequential decode) ---
  CROW_ROUTE(app, "/ocr/batch")
      .methods(crow::HTTPMethod::Post)(
          [&pool, pool_size, &decode](const crow::request &req) {
            auto x = crow::json::load(req.body);
            if (!x)
              return crow::response(400, "Invalid JSON");

            if (!x.has("images"))
              return crow::response(400, "Missing images array");

            auto &images_json = x["images"];
            size_t n = images_json.size();
            if (n == 0)
              return crow::response(400, "Empty images array");

            std::vector<cv::Mat> imgs;
            imgs.reserve(n);
            for (size_t i = 0; i < n; ++i) {
              std::string decoded = base64_decode(images_json[i].s());
              if (decoded.empty())
                continue;
              cv::Mat img = decode(
                  reinterpret_cast<const unsigned char *>(decoded.data()),
                  decoded.size());
              if (!img.empty())
                imgs.push_back(img);
            }

            if (imgs.empty())
              return crow::response(400, "No valid images");

            std::vector<std::vector<OCRResultItem>> batch_results(imgs.size());
            std::atomic<size_t> next_idx{0};

            int num_workers =
                std::min(static_cast<int>(imgs.size()), pool_size);
            {
              std::vector<std::jthread> threads;
              threads.reserve(num_workers);
              for (int w = 0; w < num_workers; ++w) {
                threads.emplace_back([&]() {
                  try {
                    auto handle = pool->acquire();
                    while (true) {
                      size_t idx = next_idx.fetch_add(1);
                      if (idx >= imgs.size())
                        break;
                      batch_results[idx] = handle->run(imgs[idx]);
                    }
                  } catch (const turbo_ocr::PoolExhaustedError &) {
                    std::cerr << "[Batch] Worker error: pool exhausted\n";
                  } catch (const std::exception &e) {
                    std::cerr << std::format("[Batch] Worker error: {}", e.what())
                              << '\n';
                  } catch (...) {
                    std::cerr << "[Batch] Worker error: unknown exception" << '\n';
                  }
                });
              }
            } // jthreads auto-join here

            std::string json_str;
            json_str.reserve(batch_results.size() * 1024);
            json_str += "{\"batch_results\":[";
            for (size_t i = 0; i < batch_results.size(); ++i) {
              if (i > 0)
                json_str += ',';
              json_str += results_to_json(batch_results[i]);
            }
            json_str += "]}";
            auto resp = crow::response(200, std::move(json_str));
            resp.set_header("Content-Type", "application/json");
            return resp;
          });

  int port = 8000;
  if (const char *env = std::getenv("PORT"))
    port = std::max(1, std::atoi(env));

  std::cout << "Starting CPU-Only OCR Server on port " << port << "..."
            << '\n';
  app.port(port).multithreaded().run();

  return 0;
}
