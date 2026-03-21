#include "turbo_ocr/decode/fast_png_decoder.h"

// Wuffs — Google's fastest PNG decoder (Apache 2.0 / MIT)
// Single-file C library, used in Chrome
#define WUFFS_IMPLEMENTATION
#define WUFFS_CONFIG__STATIC_FUNCTIONS
#include "../../third_party/wuffs/wuffs-v0.4.c"

using namespace turbo_ocr::decode;

cv::Mat FastPngDecoder::decode(const unsigned char *data, std::size_t len) {
  // 1. Initialize Wuffs PNG decoder
  wuffs_png__decoder dec;
  wuffs_base__status status = wuffs_png__decoder__initialize(
      &dec, sizeof(dec), WUFFS_VERSION, WUFFS_INITIALIZE__DEFAULT_OPTIONS);
  if (!wuffs_base__status__is_ok(&status))
    return {};

  // 2. Set source
  wuffs_base__io_buffer src = wuffs_base__ptr_u8__reader(
      const_cast<uint8_t *>(data), len, true);

  // 3. Read image config (dimensions, pixel format)
  wuffs_base__image_config ic;
  status = wuffs_png__decoder__decode_image_config(&dec, &ic, &src);
  if (!wuffs_base__status__is_ok(&status))
    return {};

  uint32_t w = wuffs_base__pixel_config__width(&ic.pixcfg);
  uint32_t h = wuffs_base__pixel_config__height(&ic.pixcfg);

  if (w == 0 || h == 0 || w > 16384 || h > 16384)
    return {};

  // 4. Set up pixel buffer — decode directly to BGR (OpenCV CV_8UC3)
  wuffs_base__pixel_config pc;
  wuffs_base__pixel_config__set(&pc, WUFFS_BASE__PIXEL_FORMAT__BGR,
                                 WUFFS_BASE__PIXEL_SUBSAMPLING__NONE, w, h);

  // Allocate cv::Mat directly — Wuffs writes into it, zero extra copies
  cv::Mat bgr(h, w, CV_8UC3);
  wuffs_base__slice_u8 pixslice = {bgr.data, static_cast<size_t>(bgr.total() * bgr.elemSize())};

  wuffs_base__pixel_buffer pb;
  status = wuffs_base__pixel_buffer__set_from_slice(&pb, &pc, pixslice);
  if (!wuffs_base__status__is_ok(&status))
    return {};

  // 5. Thread-local work buffer (reused across decodes, avoids per-call alloc)
  uint64_t workbuf_len = wuffs_png__decoder__workbuf_len(&dec).max_incl;
  thread_local std::vector<uint8_t> workbuf;
  if (workbuf.size() < workbuf_len)
    workbuf.resize(workbuf_len);
  wuffs_base__slice_u8 work = {workbuf.data(), workbuf.size()};

  // 6. Decode frame
  wuffs_base__frame_config fc;
  status = wuffs_png__decoder__decode_frame_config(&dec, &fc, &src);
  if (!wuffs_base__status__is_ok(&status))
    return {};

  status = wuffs_png__decoder__decode_frame(&dec, &pb, &src,
      WUFFS_BASE__PIXEL_BLEND__SRC, work, nullptr);
  if (!wuffs_base__status__is_ok(&status))
    return {};

  // 7. Already decoded to BGR cv::Mat — zero copies
  return bgr;
}
