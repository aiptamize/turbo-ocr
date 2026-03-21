#pragma once

#include <cstring>
#include <format>
#include <iostream>
#include <string_view>

#include <grpcpp/grpcpp.h>
#include <opencv2/core.hpp>
#include <opencv2/imgcodecs.hpp>

#include "turbo_ocr/common/box.h"
#include "turbo_ocr/common/encoding.h"
#include "turbo_ocr/common/errors.h"
#include "turbo_ocr/common/serialization.h"
#include "turbo_ocr/common/types.h"
#include "turbo_ocr/decode/fast_png_decoder.h"
#include "turbo_ocr/decode/nvjpeg_decoder.h"
#include "turbo_ocr/pipeline/gpu_pipeline_pool.h"
#include "ocr.grpc.pb.h"

namespace turbo_ocr::server {

enum class GrpcResponseMode { json_bytes, structured };

inline cv::Mat grpc_decode_image(std::string_view image_data) {
  auto *data = reinterpret_cast<const unsigned char *>(image_data.data());
  auto len = image_data.size();
  if (len >= 2 && data[0] == 0xFF && data[1] == 0xD8) {
    thread_local decode::NvJpegDecoder tl_nvjpeg;
    if (tl_nvjpeg.available()) {
      cv::Mat img = tl_nvjpeg.decode(data, len);
      if (!img.empty()) return img;
    }
    if (len > static_cast<size_t>(INT_MAX)) return {};
    return cv::imdecode(cv::Mat(1, static_cast<int>(len), CV_8UC1,
                                const_cast<unsigned char *>(data)),
                        cv::IMREAD_COLOR);
  }
  if (decode::FastPngDecoder::is_png(data, len))
    return decode::FastPngDecoder::decode(data, len);
  return {};
}

class OCRServiceImpl final : public ocr::OCRService::Service {
public:
  OCRServiceImpl(pipeline::GpuPipelinePool &pool, GrpcResponseMode mode)
      : pool_(&pool), mode_(mode) {}

  grpc::Status Recognize(grpc::ServerContext *,
                         const ocr::OCRRequest *request,
                         ocr::OCRResponse *response) override {
    if (request->image().empty()) [[unlikely]]
      return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT, "Empty image");

    cv::Mat img = grpc_decode_image(request->image());
    if (img.empty()) [[unlikely]]
      return grpc::Status(grpc::StatusCode::INVALID_ARGUMENT, "Decode failed");

    try {
      auto handle = pool_->acquire();
      auto results = handle->pipeline->run(img, handle->stream);
      fill_response(response, results);
      return grpc::Status::OK;
    } catch (const PoolExhaustedError &) {
      return grpc::Status(grpc::StatusCode::RESOURCE_EXHAUSTED,
                          "Pipeline pool exhausted");
    } catch (const std::exception &e) {
      std::cerr << std::format("[gRPC] Inference error: {}\n", e.what());
      return grpc::Status(grpc::StatusCode::INTERNAL, "Inference error");
    } catch (...) {
      std::cerr << "[gRPC] Inference error: unknown exception\n";
      return grpc::Status(grpc::StatusCode::INTERNAL, "Inference error");
    }
  }

  grpc::Status RecognizeBatch(grpc::ServerContext *, const ocr::OCRBatchRequest *,
                              ocr::OCRBatchResponse *) override {
    return grpc::Status(grpc::StatusCode::UNIMPLEMENTED,
                        "Use concurrent Recognize calls instead");
  }

private:
  void fill_response(ocr::OCRResponse *response,
                     const std::vector<OCRResultItem> &results) {
    response->set_num_detections(static_cast<int>(results.size()));
    if (mode_ == GrpcResponseMode::json_bytes) {
      response->set_json_response(results_to_json(results));
    } else {
      response->mutable_results()->Reserve(static_cast<int>(results.size()));
      for (const auto &item : results) {
        auto *result = response->add_results();
        result->set_text(item.text);
        result->set_confidence(item.confidence);
        result->mutable_bounding_box()->Reserve(4);
        for (int k = 0; k < 4; ++k) {
          auto *bbox = result->add_bounding_box();
          bbox->mutable_x()->Reserve(1);
          bbox->mutable_y()->Reserve(1);
          bbox->add_x(static_cast<float>(item.box[k][0]));
          bbox->add_y(static_cast<float>(item.box[k][1]));
        }
      }
    }
  }

  pipeline::GpuPipelinePool *pool_ = nullptr;
  GrpcResponseMode mode_;
};

/// Start gRPC server on a background thread. Returns the server and thread.
/// Caller must keep both alive. Call server->Shutdown() to stop.
struct GrpcHandle {
  std::unique_ptr<grpc::Server> server;
  std::jthread thread;
};

inline GrpcHandle start_grpc_server(pipeline::GpuPipelinePool &pool, int port) {
  auto mode = GrpcResponseMode::json_bytes;
  if (const char *env = std::getenv("GRPC_RESPONSE_MODE")) {
    if (std::strcmp(env, "structured") == 0)
      mode = GrpcResponseMode::structured;
  }

  auto service = std::make_shared<OCRServiceImpl>(pool, mode);

  constexpr int kMaxMsg = 100 * 1024 * 1024;
  int cqs = 10;
  if (const char *env = std::getenv("GRPC_CQS"))
    cqs = std::max(1, std::atoi(env));

  auto address = std::format("0.0.0.0:{}", port);

  grpc::ServerBuilder builder;
  builder.AddListeningPort(address, grpc::InsecureServerCredentials());
  builder.RegisterService(service.get());
  builder.SetMaxReceiveMessageSize(kMaxMsg);
  builder.SetSyncServerOption(grpc::ServerBuilder::SyncServerOption::NUM_CQS, cqs);
  builder.SetSyncServerOption(grpc::ServerBuilder::SyncServerOption::MIN_POLLERS, cqs);
  builder.SetSyncServerOption(grpc::ServerBuilder::SyncServerOption::MAX_POLLERS, cqs * 2);
  builder.AddChannelArgument(GRPC_ARG_ALLOW_REUSEPORT, 1);
  builder.AddChannelArgument(GRPC_ARG_MINIMAL_STACK, 1);

  auto server = builder.BuildAndStart();
  std::cout << std::format("gRPC server listening on {}\n", address);

  // Run in background thread — gRPC::Wait() blocks, so we need a thread.
  // The shared_ptr to service is captured to keep it alive.
  auto thread = std::jthread([srv = server.get(), svc = service]() {
    srv->Wait();
  });

  return {std::move(server), std::move(thread)};
}

} // namespace turbo_ocr::server
