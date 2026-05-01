#pragma once

#include <format>
#include <iostream>
#include <utility>
#include <vector>

#include <cuda_runtime.h>
#include <nvjpeg.h>
#include <opencv2/core.hpp>

// GPU-accelerated JPEG decoder using nvJPEG.
// Decodes JPEG bytes -> cv::Mat (BGR, on CPU) significantly faster than cv::imdecode.
// Falls back to cv::imdecode for non-JPEG formats.

namespace turbo_ocr::decode {

class NvJpegDecoder {
public:
  NvJpegDecoder() {
    // Try NVDEC hardware decoder first (offloads Huffman to dedicated HW)
    nvjpegStatus_t st = nvjpegCreateEx(NVJPEG_BACKEND_HARDWARE, nullptr, nullptr, 0, &handle_);
    if (st == NVJPEG_STATUS_SUCCESS) {
      std::cerr << "[NvJpeg] Using HARDWARE (NVDEC) backend\n";
    } else {
      // Fallback to GPU_HYBRID (GPU-assisted Huffman, frees compute cores)
      st = nvjpegCreateEx(NVJPEG_BACKEND_GPU_HYBRID, nullptr, nullptr, 0, &handle_);
      if (st == NVJPEG_STATUS_SUCCESS) {
        std::cerr << "[NvJpeg] Using GPU_HYBRID backend\n";
      } else {
        // Final fallback to simple (default hybrid CPU+GPU)
        st = nvjpegCreateSimple(&handle_);
        if (st == NVJPEG_STATUS_SUCCESS) {
          std::cerr << "[NvJpeg] Using default (simple) backend\n";
        }
      }
    }
    if (st != NVJPEG_STATUS_SUCCESS) {
      std::cerr << std::format("[NvJpeg] Failed to create handle: {}", static_cast<int>(st)) << '\n';
      handle_ = nullptr;
      return;
    }
    st = nvjpegJpegStateCreate(handle_, &state_);
    if (st != NVJPEG_STATUS_SUCCESS) {
      std::cerr << std::format("[NvJpeg] Failed to create state: {}", static_cast<int>(st)) << '\n';
      nvjpegDestroy(handle_);
      handle_ = nullptr;
    }
  }

  ~NvJpegDecoder() noexcept {
    // Safety contract: every public call (decode/batch_decode/decode_to_gpu
    // when caller-synced) returns only after its CUDA stream has either
    // synchronized or been queued behind the caller's sync. So when this
    // dtor runs there is no in-flight work referencing state_/handle_.
    // If a future caller adds an unsynchronized path, sync the default
    // stream HERE before destroying handles to keep the contract.
    if (state_) nvjpegJpegStateDestroy(state_);
    if (handle_) nvjpegDestroy(handle_);
  }

  // Non-copyable, non-movable (owns GPU resources)
  NvJpegDecoder(const NvJpegDecoder &) = delete;
  NvJpegDecoder &operator=(const NvJpegDecoder &) = delete;
  NvJpegDecoder(NvJpegDecoder &&) = delete;
  NvJpegDecoder &operator=(NvJpegDecoder &&) = delete;

  // Decode JPEG bytes to BGR cv::Mat. Returns empty Mat on failure.
  [[nodiscard]] cv::Mat decode(const unsigned char *data, size_t len, cudaStream_t stream = 0) {
    if (!handle_ || len < 2)
      return {};

    // Check JPEG magic bytes
    if (data[0] != 0xFF || data[1] != 0xD8)
      return {};  // Not JPEG -- caller should fall back to cv::imdecode

    // Get image info
    int nComponents;
    nvjpegChromaSubsampling_t subsampling;
    int widths[NVJPEG_MAX_COMPONENT], heights[NVJPEG_MAX_COMPONENT];
    nvjpegStatus_t st = nvjpegGetImageInfo(handle_, data, len,
                                            &nComponents, &subsampling,
                                            widths, heights);
    if (st != NVJPEG_STATUS_SUCCESS)
      return {};

    int w = widths[0], h = heights[0];

    // Allocate output (interleaved BGR)
    nvjpegImage_t output;
    cv::Mat result(h, w, CV_8UC3);
    output.channel[0] = result.data;
    output.pitch[0] = w * 3;
    for (int i = 1; i < NVJPEG_MAX_COMPONENT; i++) {
      output.channel[i] = nullptr;
      output.pitch[i] = 0;
    }

    // Decode to CPU interleaved BGR
    st = nvjpegDecode(handle_, state_, data, len,
                      NVJPEG_OUTPUT_BGRI, &output, stream);
    if (st != NVJPEG_STATUS_SUCCESS) {
      return {};
    }

    // nvjpegDecode with CPU output is synchronous, no stream sync needed
    return result;
  }

  // Check if data is a JPEG (starts with FF D8).
  [[nodiscard]] static bool is_jpeg(const unsigned char *data, size_t len) noexcept {
    return len >= 2 && data[0] == 0xFF && data[1] == 0xD8;
  }

  // Get JPEG image dimensions without decoding.
  // Returns {width, height} or {0, 0} on failure.
  [[nodiscard]] std::pair<int, int> get_dimensions(const unsigned char *data, size_t len) {
    if (!handle_ || !is_jpeg(data, len))
      return {0, 0};
    int nComponents;
    nvjpegChromaSubsampling_t subsampling;
    int widths[NVJPEG_MAX_COMPONENT], heights[NVJPEG_MAX_COMPONENT];
    nvjpegStatus_t st = nvjpegGetImageInfo(handle_, data, len,
                                            &nComponents, &subsampling,
                                            widths, heights);
    if (st != NVJPEG_STATUS_SUCCESS)
      return {0, 0};
    return {widths[0], heights[0]};
  }

  // Decode JPEG bytes directly to a GPU buffer (device memory).
  // The caller provides a pre-allocated device buffer with the given pitch.
  // The buffer must be large enough for h * pitch bytes (pitch >= w * 3).
  // Returns true on success. The decode is async on the given stream;
  // the caller must synchronize the stream before reading the output.
  [[nodiscard]] bool decode_to_gpu(const unsigned char *data, size_t len,
                                    void *d_output, size_t pitch,
                                    int w, int h,
                                    cudaStream_t stream = 0) {
    if (!handle_ || !is_jpeg(data, len))
      return false;

    nvjpegImage_t output;
    output.channel[0] = static_cast<unsigned char *>(d_output);
    output.pitch[0] = static_cast<unsigned int>(pitch);
    for (int i = 1; i < NVJPEG_MAX_COMPONENT; i++) {
      output.channel[i] = nullptr;
      output.pitch[i] = 0;
    }

    // Decode directly to GPU memory (interleaved BGR)
    nvjpegStatus_t st = nvjpegDecode(handle_, state_, data, len,
                                      NVJPEG_OUTPUT_BGRI, &output, stream);
    return st == NVJPEG_STATUS_SUCCESS;
  }

  // Batch decode multiple JPEG images in one nvjpegDecodeBatched call.
  // Input: vector of (data, length) pairs — must all be valid JPEGs.
  // Returns: vector of cv::Mat (BGR). Failed images get empty Mat.
  // Non-JPEG images should be filtered out by the caller and decoded separately.
  [[nodiscard]] std::vector<cv::Mat> batch_decode(
      const std::vector<std::pair<const unsigned char *, size_t>> &jpeg_buffers,
      cudaStream_t stream = 0) {

    size_t n = jpeg_buffers.size();
    std::vector<cv::Mat> results(n);

    if (!handle_ || n == 0)
      return results;

    // Initialize batch decode state
    nvjpegStatus_t st = nvjpegDecodeBatchedInitialize(
        handle_, state_, static_cast<int>(n), 1 /*max_cpu_threads*/,
        NVJPEG_OUTPUT_BGRI);
    if (st != NVJPEG_STATUS_SUCCESS) {
      std::cerr << std::format("[NvJpeg] Batch init failed: {}, falling back to single decode",
                               static_cast<int>(st)) << '\n';
      // Fallback: decode one at a time
      for (size_t i = 0; i < n; ++i)
        results[i] = decode(jpeg_buffers[i].first, jpeg_buffers[i].second, stream);
      return results;
    }

    // Get image dimensions and prepare output buffers
    std::vector<nvjpegImage_t> outputs(n);
    bool all_ok = true;

    for (size_t i = 0; i < n; ++i) {
      auto [data, len] = jpeg_buffers[i];
      if (len < 2 || data[0] != 0xFF || data[1] != 0xD8) {
        all_ok = false;
        break;
      }

      int nComponents;
      nvjpegChromaSubsampling_t subsampling;
      int widths[NVJPEG_MAX_COMPONENT], heights[NVJPEG_MAX_COMPONENT];
      st = nvjpegGetImageInfo(handle_, data, len,
                              &nComponents, &subsampling, widths, heights);
      if (st != NVJPEG_STATUS_SUCCESS) {
        all_ok = false;
        break;
      }

      int w = widths[0], h = heights[0];
      results[i] = cv::Mat(h, w, CV_8UC3);
      outputs[i].channel[0] = results[i].data;
      outputs[i].pitch[0] = w * 3;
      for (int c = 1; c < NVJPEG_MAX_COMPONENT; c++) {
        outputs[i].channel[c] = nullptr;
        outputs[i].pitch[c] = 0;
      }
    }

    if (!all_ok) {
      // Fallback to single decode if any image failed header parse
      for (size_t i = 0; i < n; ++i)
        results[i] = decode(jpeg_buffers[i].first, jpeg_buffers[i].second, stream);
      return results;
    }

    // Build flat arrays for nvjpegDecodeBatched
    std::vector<const unsigned char *> data_ptrs(n);
    std::vector<size_t> lengths(n);
    for (size_t i = 0; i < n; ++i) {
      data_ptrs[i] = jpeg_buffers[i].first;
      lengths[i] = jpeg_buffers[i].second;
    }

    // Batch decode all JPEGs in one call
    st = nvjpegDecodeBatched(handle_, state_,
                              data_ptrs.data(), lengths.data(),
                              outputs.data(), stream);
    if (st != NVJPEG_STATUS_SUCCESS) {
      std::cerr << std::format("[NvJpeg] Batch decode failed: {}, falling back to single decode",
                               static_cast<int>(st)) << '\n';
      for (size_t j = 0; j < n; ++j)
        results[j] = decode(jpeg_buffers[j].first, jpeg_buffers[j].second, stream);
      return results;
    }

    // Sync stream — nvjpegDecodeBatched may use GPU for Huffman decode
    cudaStreamSynchronize(stream);

    return results;
  }

  [[nodiscard]] bool available() const noexcept { return handle_ != nullptr; }

private:
  nvjpegHandle_t handle_ = nullptr;
  nvjpegJpegState_t state_ = nullptr;
};

} // namespace turbo_ocr::decode
