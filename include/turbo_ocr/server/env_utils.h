#pragma once

#include <cstdlib>
#include <string>
#include <string_view>

namespace turbo_ocr::server {

/// Read an environment variable with a fallback default.
[[nodiscard]] inline std::string env_or(const char *name,
                                        std::string_view def) {
  if (const char *v = std::getenv(name))
    return std::string(v);
  return std::string(def);
}

/// Check if an environment variable equals "1".
[[nodiscard]] inline bool env_enabled(const char *name) noexcept {
  const char *v = std::getenv(name);
  return v && v[0] == '1' && v[1] == '\0';
}

} // namespace turbo_ocr::server
