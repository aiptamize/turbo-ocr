#pragma once

#include <cstddef>
#include <cstdint>
#include <functional>
#include <mutex>
#include <string>
#include <vector>

#include <opencv2/core.hpp>

namespace turbo_ocr::render {

// Pool of persistent fastpdf2png v2.0 daemons.
// Each daemon = separate process with own PDFium + fork workers.
// Raw PPM output (-c -1) to /dev/shm = zero PNG overhead, ramdisk I/O.
// 16 daemons handle concurrent requests independently (no shared state).

class PdfRenderer {
public:
  explicit PdfRenderer(int pool_size = 16, int workers_per_render = 4);
  ~PdfRenderer() noexcept;

  PdfRenderer(const PdfRenderer &) = delete;
  PdfRenderer &operator=(const PdfRenderer &) = delete;
  PdfRenderer(PdfRenderer &&) = delete;
  PdfRenderer &operator=(PdfRenderer &&) = delete;

  /// Render all pages of a PDF document to cv::Mat images.
  [[nodiscard]] std::vector<cv::Mat> render(const uint8_t *data, size_t len, int dpi = 100);

  /// Callback type: (page_index, page_image) -> void.
  /// Called as soon as each page is rendered, enabling overlap with OCR.
  using PageCallback = std::function<void(int page_idx, cv::Mat img)>;

  /// Render pages and invoke callback for each page as soon as it's ready.
  /// Uses inotify to detect PPM files appearing in the output directory,
  /// enabling OCR to start on page 1 while later pages are still rendering.
  /// Returns the total number of pages.
  int render_streamed(const uint8_t *data, size_t len, int dpi,
                      PageCallback on_page);

private:
  struct Daemon {
    pid_t pid = -1;
    FILE *cmd_in = nullptr;
    FILE *result_out = nullptr;
    std::mutex mutex;
  };

  int pool_size_;
  int workers_per_render_;
  std::string binary_path_;
  std::vector<Daemon> daemons_;

  [[nodiscard]] int acquire_daemon();
  [[nodiscard]] std::string send_cmd(Daemon &d, const std::string &cmd);
  [[nodiscard]] static cv::Mat read_ppm(const std::string &path);
};

} // namespace turbo_ocr::render
