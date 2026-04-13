#include "turbo_ocr/routes/common_routes.h"

#include <format>
#include <iostream>

namespace turbo_ocr::routes {

void register_health_route(crow::SimpleApp &app) {
  CROW_ROUTE(app, "/health")
      .methods(crow::HTTPMethod::Get)([](const crow::request &) {
    return crow::response(200, "ok");
  });
}

void register_ocr_base64_route(crow::SimpleApp &app,
                                const server::InferFunc &infer,
                                const server::ImageDecoder &decode,
                                bool layout_available) {
  CROW_ROUTE(app, "/ocr")
      .methods(crow::HTTPMethod::Post)(
          [&infer, &decode, layout_available](const crow::request &req) {
    bool want_layout = false;
    if (auto err = server::parse_layout_query(req, layout_available, &want_layout); !err.empty())
      return crow::response(400, err);

    auto x = crow::json::load(req.body);
    if (!x)
      return crow::response(400, "Invalid JSON");

    if (!x.has("image") || x["image"].t() != crow::json::type::String
        || x["image"].s().size() == 0)
      return crow::response(400, "Empty or missing image field");

    std::string decoded = turbo_ocr::base64_decode(x["image"].s());
    if (decoded.empty())
      return crow::response(400, "Failed to decode base64");

    cv::Mat img = decode(
        reinterpret_cast<const unsigned char *>(decoded.data()),
        decoded.size());

    if (img.empty())
      return crow::response(400, "Failed to decode image");

    try {
      auto inf = infer(img, want_layout);
      auto json_str = turbo_ocr::results_to_json(inf.results, inf.layout);
      auto resp = crow::response(200, std::move(json_str));
      resp.set_header("Content-Type", "application/json");
      return resp;
    } catch (const turbo_ocr::PoolExhaustedError &) {
      return crow::response(503, "Service overloaded, try again later");
    } catch (const std::exception &e) {
      std::cerr << std::format("[/ocr] Inference error: {}\n", e.what());
      return crow::response(500, "Inference error");
    } catch (...) {
      std::cerr << "[/ocr] Inference error: unknown exception\n";
      return crow::response(500, "Inference error");
    }
  });
}

void register_ocr_raw_route(crow::SimpleApp &app,
                             const server::InferFunc &infer,
                             const server::ImageDecoder &decode,
                             bool layout_available) {
  CROW_ROUTE(app, "/ocr/raw")
      .methods(crow::HTTPMethod::Post)(
          [&infer, &decode, layout_available](const crow::request &req) {
    if (req.body.empty())
      return crow::response(400, "Empty body");

    bool want_layout = false;
    if (auto err = server::parse_layout_query(req, layout_available, &want_layout); !err.empty())
      return crow::response(400, err);

    cv::Mat img = decode(
        reinterpret_cast<const unsigned char *>(req.body.data()),
        req.body.size());

    if (img.empty())
      return crow::response(400, "Failed to decode image");

    try {
      auto inf = infer(img, want_layout);
      auto json_str = turbo_ocr::results_to_json(inf.results, inf.layout);
      auto resp = crow::response(200, std::move(json_str));
      resp.set_header("Content-Type", "application/json");
      return resp;
    } catch (const turbo_ocr::PoolExhaustedError &) {
      return crow::response(503, "Service overloaded, try again later");
    } catch (const std::exception &e) {
      std::cerr << std::format("[/ocr/raw] Inference error: {}\n", e.what());
      return crow::response(500, "Inference error");
    } catch (...) {
      std::cerr << "[/ocr/raw] Inference error: unknown exception\n";
      return crow::response(500, "Inference error");
    }
  });
}

void register_common_routes(crow::SimpleApp &app,
                             const server::InferFunc &infer,
                             const server::ImageDecoder &decode,
                             bool layout_available) {
  register_health_route(app);
  register_ocr_base64_route(app, infer, decode, layout_available);
  register_ocr_raw_route(app, infer, decode, layout_available);
}

} // namespace turbo_ocr::routes
