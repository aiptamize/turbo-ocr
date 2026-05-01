// Unit tests for the format-aware image-dimension sniffer.
//
// These verify that the pre-decode dim peek correctly extracts width/height
// from PNG IHDR, JPEG SOFn, and the three WebP variants — and that it
// gracefully returns nullopt on garbage / truncated / non-image inputs so
// callers fall through to the decoder's own error handling.
#include "turbo_ocr/decode/image_dims.h"

#include "catch_amalgamated.hpp"

#include <cstring>
#include <vector>

using turbo_ocr::decode::ImageDims;
using turbo_ocr::decode::peek_image_dimensions;
using turbo_ocr::decode::peek_jpeg_dims;
using turbo_ocr::decode::peek_png_dims;
using turbo_ocr::decode::peek_webp_dims;

namespace {

// Build a minimal valid PNG header just past IHDR (24 bytes total).
std::vector<unsigned char> make_png_header(uint32_t w, uint32_t h) {
  std::vector<unsigned char> bytes(24, 0);
  unsigned char sig[] = {0x89, 0x50, 0x4E, 0x47, 0x0D, 0x0A, 0x1A, 0x0A};
  std::memcpy(bytes.data(), sig, 8);
  // IHDR length (always 13)
  bytes[8] = 0; bytes[9] = 0; bytes[10] = 0; bytes[11] = 13;
  std::memcpy(bytes.data() + 12, "IHDR", 4);
  bytes[16] = (w >> 24) & 0xFF; bytes[17] = (w >> 16) & 0xFF;
  bytes[18] = (w >> 8)  & 0xFF; bytes[19] =  w        & 0xFF;
  bytes[20] = (h >> 24) & 0xFF; bytes[21] = (h >> 16) & 0xFF;
  bytes[22] = (h >> 8)  & 0xFF; bytes[23] =  h        & 0xFF;
  return bytes;
}

// Build a minimal JPEG: SOI + SOF0 segment with the requested W/H.
std::vector<unsigned char> make_jpeg_header(uint16_t w, uint16_t h) {
  std::vector<unsigned char> bytes;
  bytes.push_back(0xFF); bytes.push_back(0xD8);          // SOI
  bytes.push_back(0xFF); bytes.push_back(0xC0);          // SOF0 marker
  bytes.push_back(0x00); bytes.push_back(0x11);          // segment length = 17
  bytes.push_back(0x08);                                  // 8-bit precision
  bytes.push_back((h >> 8) & 0xFF); bytes.push_back(h & 0xFF);
  bytes.push_back((w >> 8) & 0xFF); bytes.push_back(w & 0xFF);
  bytes.push_back(0x03);                                  // 3 components
  for (int i = 0; i < 9; ++i) bytes.push_back(0);         // component info pad
  return bytes;
}

std::vector<unsigned char> make_webp_vp8x(uint32_t w, uint32_t h) {
  // RIFF .... WEBP VP8X len .. flags .. reserved .. (w-1)24LE (h-1)24LE
  std::vector<unsigned char> bytes(30, 0);
  std::memcpy(bytes.data(), "RIFF", 4);
  std::memcpy(bytes.data() + 8, "WEBP", 4);
  std::memcpy(bytes.data() + 12, "VP8X", 4);
  uint32_t wm = w - 1, hm = h - 1;
  bytes[24] = wm & 0xFF; bytes[25] = (wm >> 8) & 0xFF; bytes[26] = (wm >> 16) & 0xFF;
  bytes[27] = hm & 0xFF; bytes[28] = (hm >> 8) & 0xFF; bytes[29] = (hm >> 16) & 0xFF;
  return bytes;
}

} // namespace

TEST_CASE("PNG IHDR sniffer", "[image_dims][png]") {
  SECTION("standard 1920x1080") {
    auto bytes = make_png_header(1920, 1080);
    auto d = peek_png_dims(bytes.data(), bytes.size());
    REQUIRE(d.has_value());
    REQUIRE(d->width == 1920);
    REQUIRE(d->height == 1080);
  }
  SECTION("decompression bomb dimensions (100k x 100k)") {
    auto bytes = make_png_header(100000, 100000);
    auto d = peek_png_dims(bytes.data(), bytes.size());
    REQUIRE(d.has_value());
    REQUIRE(d->width == 100000);
    REQUIRE(d->height == 100000);
  }
  SECTION("rejects truncated buffer") {
    auto bytes = make_png_header(100, 100);
    bytes.resize(20); // mid-IHDR
    REQUIRE_FALSE(peek_png_dims(bytes.data(), bytes.size()).has_value());
  }
  SECTION("rejects bad signature") {
    auto bytes = make_png_header(100, 100);
    bytes[0] = 0x00;
    REQUIRE_FALSE(peek_png_dims(bytes.data(), bytes.size()).has_value());
  }
  SECTION("rejects 0x0") {
    auto bytes = make_png_header(0, 100);
    REQUIRE_FALSE(peek_png_dims(bytes.data(), bytes.size()).has_value());
  }
}

TEST_CASE("JPEG SOFn sniffer", "[image_dims][jpeg]") {
  SECTION("standard 800x600") {
    auto bytes = make_jpeg_header(800, 600);
    auto d = peek_jpeg_dims(bytes.data(), bytes.size());
    REQUIRE(d.has_value());
    REQUIRE(d->width == 800);
    REQUIRE(d->height == 600);
  }
  SECTION("rejects non-JPEG") {
    std::vector<unsigned char> bytes = {0x00, 0x01, 0x02, 0x03};
    REQUIRE_FALSE(peek_jpeg_dims(bytes.data(), bytes.size()).has_value());
  }
  SECTION("rejects truncated") {
    auto bytes = make_jpeg_header(800, 600);
    bytes.resize(5);
    REQUIRE_FALSE(peek_jpeg_dims(bytes.data(), bytes.size()).has_value());
  }
}

TEST_CASE("WebP VP8X sniffer", "[image_dims][webp]") {
  SECTION("standard 1024x768 VP8X") {
    auto bytes = make_webp_vp8x(1024, 768);
    auto d = peek_webp_dims(bytes.data(), bytes.size());
    REQUIRE(d.has_value());
    REQUIRE(d->width == 1024);
    REQUIRE(d->height == 768);
  }
  SECTION("rejects non-WebP RIFF") {
    auto bytes = make_webp_vp8x(100, 100);
    std::memcpy(bytes.data() + 8, "WAVE", 4);
    REQUIRE_FALSE(peek_webp_dims(bytes.data(), bytes.size()).has_value());
  }
}

TEST_CASE("dispatch picks the right format", "[image_dims][dispatch]") {
  SECTION("PNG dispatched") {
    auto bytes = make_png_header(640, 480);
    auto d = peek_image_dimensions(bytes.data(), bytes.size());
    REQUIRE(d.has_value());
    REQUIRE(d->width == 640);
  }
  SECTION("JPEG dispatched") {
    auto bytes = make_jpeg_header(640, 480);
    auto d = peek_image_dimensions(bytes.data(), bytes.size());
    REQUIRE(d.has_value());
    REQUIRE(d->height == 480);
  }
  SECTION("WebP dispatched") {
    auto bytes = make_webp_vp8x(640, 480);
    auto d = peek_image_dimensions(bytes.data(), bytes.size());
    REQUIRE(d.has_value());
    REQUIRE(d->width == 640);
  }
  SECTION("unknown format returns nullopt") {
    std::vector<unsigned char> garbage(50, 0xCC);
    REQUIRE_FALSE(peek_image_dimensions(garbage.data(), garbage.size()).has_value());
  }
  SECTION("empty input returns nullopt") {
    REQUIRE_FALSE(peek_image_dimensions(nullptr, 0).has_value());
  }
}
