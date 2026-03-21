#pragma once

#include "turbo_ocr/common/types.h"
#include <cstdio>
#include <string>
#include <vector>

namespace turbo_ocr {

// Fast JSON serializer -- builds string directly without any JSON library overhead
[[nodiscard]] inline std::string results_to_json(const std::vector<OCRResultItem> &results) {
  std::string j;
  j.reserve(results.size() * 200);
  j += "{\"results\":[";
  for (size_t i = 0; i < results.size(); ++i) {
    if (i > 0) j += ',';
    const auto &item = results[i];
    j += "{\"text\":\"";
    for (char c : item.text) {
      switch (c) {
        case '"': j += "\\\""; break;
        case '\\': j += "\\\\"; break;
        case '\n': j += "\\n"; break;
        case '\r': j += "\\r"; break;
        case '\t': j += "\\t"; break;
        default:
          if (c < 0x20) {
            char buf[7];
            snprintf(buf, sizeof(buf), "\\u%04x", static_cast<unsigned>(static_cast<unsigned char>(c)));
            j += buf;
          } else {
            j += c;
          }
      }
    }
    j += "\",\"confidence\":";
    // Fast float-to-string (avoid std::format overhead in hot path)
    char conf_str[16];
    snprintf(conf_str, sizeof(conf_str), "%.5g", item.confidence);
    j += conf_str;
    j += ",\"bounding_box\":[";
    for (int k = 0; k < 4; ++k) {
      if (k > 0) j += ',';
      j += '[';
      j += std::to_string(item.box[k][0]);
      j += ',';
      j += std::to_string(item.box[k][1]);
      j += ']';
    }
    j += "]}";
  }
  j += "]}";
  return j;
}

} // namespace turbo_ocr
