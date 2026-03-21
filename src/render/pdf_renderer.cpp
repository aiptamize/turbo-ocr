#include "turbo_ocr/render/pdf_renderer.h"
#include "turbo_ocr/common/errors.h"

#include <atomic>
#include <csignal>
#include <cstdio>
#include <cstring>
#include <filesystem>
#include <format>
#include <memory>
#include <mutex>
#include <thread>

#include <opencv2/imgproc.hpp>
#include <poll.h>
#include <sys/inotify.h>
#include <sys/wait.h>
#include <unistd.h>

using namespace turbo_ocr::render;

static std::string find_binary() {
  static constexpr const char *paths[] = {
    "/app/bin/fastpdf2png",
    "/usr/local/bin/fastpdf2png",
    "./build/fastpdf2png",
    "./bin/fastpdf2png",
  };
  for (const char *p : paths) {
    if (std::filesystem::exists(p)) return p;
  }
  throw turbo_ocr::PdfRenderError("fastpdf2png binary not found");
}

static bool try_write_file(const char *tmpl, const uint8_t *data, size_t len,
                           std::string &out) {
  char path[64];
  std::strncpy(path, tmpl, sizeof(path) - 1);
  path[sizeof(path) - 1] = '\0';
  int fd = mkstemp(path);
  if (fd < 0) return false;
  size_t written = 0;
  while (written < len) {
    auto n = ::write(fd, data + written, len - written);
    if (n <= 0) { close(fd); unlink(path); return false; }
    written += n;
  }
  close(fd);
  out = path;
  return true;
}

static std::string write_temp_pdf(const uint8_t *data, size_t len) {
  std::string path;
  if (try_write_file("/dev/shm/ocr_pdf_XXXXXX", data, len, path)) return path;
  if (try_write_file("/tmp/ocr_pdf_XXXXXX", data, len, path)) return path;
  throw turbo_ocr::PdfRenderError("Failed to create temp PDF file");
}

static std::string make_temp_dir() {
  const char *templates[] = {"/dev/shm/ocr_out_XXXXXX", "/tmp/ocr_out_XXXXXX"};
  for (auto *tmpl : templates) {
    char path[64];
    std::strncpy(path, tmpl, sizeof(path) - 1);
    path[sizeof(path) - 1] = '\0';
    if (mkdtemp(path)) return path;
  }
  throw turbo_ocr::PdfRenderError("Failed to create temp output dir");
}

// RAII guard for temp file/directory cleanup.
struct TempGuard {
  std::string path;
  bool is_dir;
  TempGuard(std::string p, bool dir) : path(std::move(p)), is_dir(dir) {}
  ~TempGuard() noexcept {
    if (path.empty()) return;
    try {
      if (is_dir) std::filesystem::remove_all(path);
      else unlink(path.c_str());
    } catch (...) {}
  }
  void release() { path.clear(); }
  TempGuard(const TempGuard &) = delete;
  TempGuard &operator=(const TempGuard &) = delete;
};

// Fast PPM/PGM reader — directly to BGR cv::Mat, no OpenCV imread overhead.
cv::Mat PdfRenderer::read_ppm(const std::string &path) {
  struct FileCloser { void operator()(FILE *fp) const noexcept { fclose(fp); } };
  std::unique_ptr<FILE, FileCloser> f(fopen(path.c_str(), "rb"));
  if (!f) return {};

  char magic[3] = {};
  if (fread(magic, 1, 2, f.get()) != 2) return {};
  bool gray = (magic[0] == 'P' && magic[1] == '5');
  bool color = (magic[0] == 'P' && magic[1] == '6');
  if (!gray && !color) return {};

  auto skip = [&]() {
    int c;
    while ((c = fgetc(f.get())) != EOF) {
      if (c == '#') { while ((c = fgetc(f.get())) != EOF && c != '\n'); }
      else if (c > ' ') { ungetc(c, f.get()); return; }
    }
  };

  int w = 0, h = 0, maxval = 0;
  skip(); if (fscanf(f.get(), "%d", &w) != 1) return {};
  skip(); if (fscanf(f.get(), "%d", &h) != 1) return {};
  skip(); if (fscanf(f.get(), "%d", &maxval) != 1) return {};
  fgetc(f.get());

  if (w <= 0 || h <= 0 || w > 16384 || h > 16384 || maxval != 255) return {};

  if (gray) {
    cv::Mat g(h, w, CV_8UC1);
    size_t expected = static_cast<size_t>(w) * h;
    if (fread(g.data, 1, expected, f.get()) != expected) return {};
    cv::Mat bgr;
    cv::cvtColor(g, bgr, cv::COLOR_GRAY2BGR);
    return bgr;
  } else {
    cv::Mat rgb(h, w, CV_8UC3);
    size_t expected = static_cast<size_t>(w) * h * 3;
    if (fread(rgb.data, 1, expected, f.get()) != expected) return {};
    cv::Mat bgr;
    cv::cvtColor(rgb, bgr, cv::COLOR_RGB2BGR);
    return bgr;
  }
}

PdfRenderer::PdfRenderer(int pool_size, int workers_per_render)
    : pool_size_(pool_size), workers_per_render_(workers_per_render),
      daemons_(pool_size) {
  binary_path_ = find_binary();

  for (int i = 0; i < pool_size_; ++i) {
    int in_pipe[2], out_pipe[2];
    if (pipe(in_pipe) < 0 || pipe(out_pipe) < 0)
      throw turbo_ocr::PdfRenderError("pipe() failed for PDF renderer daemon");

    pid_t pid = fork();
    if (pid < 0) throw turbo_ocr::PdfRenderError("fork() failed for PDF renderer daemon");

    if (pid == 0) {
      dup2(in_pipe[0], STDIN_FILENO);
      dup2(out_pipe[1], STDOUT_FILENO);
      close(in_pipe[0]); close(in_pipe[1]);
      close(out_pipe[0]); close(out_pipe[1]);
      for (int j = 0; j < i; ++j) {
        if (daemons_[j].cmd_in) fclose(daemons_[j].cmd_in);
        if (daemons_[j].result_out) fclose(daemons_[j].result_out);
      }
      execl(binary_path_.c_str(), binary_path_.c_str(), "--daemon", nullptr);
      _exit(1);
    }

    close(in_pipe[0]);
    close(out_pipe[1]);
    daemons_[i].pid = pid;
    daemons_[i].cmd_in = fdopen(in_pipe[1], "w");
    daemons_[i].result_out = fdopen(out_pipe[0], "r");
    if (!daemons_[i].cmd_in || !daemons_[i].result_out)
      throw turbo_ocr::PdfRenderError("fdopen failed for PDF renderer daemon");
  }
}

PdfRenderer::~PdfRenderer() noexcept {
  for (auto &d : daemons_) {
    if (d.cmd_in) {
      fprintf(d.cmd_in, "QUIT\n");
      fflush(d.cmd_in);
      fclose(d.cmd_in);
    }
    if (d.result_out) fclose(d.result_out);
    if (d.pid > 0) {
      // Wait briefly, then force-kill to avoid hanging on shutdown
      if (waitpid(d.pid, nullptr, WNOHANG) == 0) {
        kill(d.pid, SIGKILL);
        waitpid(d.pid, nullptr, 0);
      }
    }
  }
}

int PdfRenderer::acquire_daemon() {
  static thread_local int hint = 0;
  for (int i = 0; i < pool_size_; ++i) {
    int idx = (hint + i) % pool_size_;
    if (daemons_[idx].mutex.try_lock()) {
      hint = (idx + 1) % pool_size_;
      return idx;
    }
  }
  int idx = hint % pool_size_;
  // Lock is acquired here and released via std::unique_lock in render()
  daemons_[idx].mutex.lock();
  hint = (idx + 1) % pool_size_;
  return idx;
}

std::string PdfRenderer::send_cmd(Daemon &d, const std::string &cmd) {
  fprintf(d.cmd_in, "%s\n", cmd.c_str());
  fflush(d.cmd_in);
  char buf[4096];
  if (!fgets(buf, sizeof(buf), d.result_out))
    throw turbo_ocr::PdfRenderError("PDF renderer daemon read failed (daemon may have crashed)");
  auto len = std::strlen(buf);
  if (len > 0 && buf[len - 1] == '\n') buf[len - 1] = '\0';
  return buf;
}

std::vector<cv::Mat> PdfRenderer::render(const uint8_t *data, size_t len,
                                         int dpi) {
  TempGuard tmpfile(write_temp_pdf(data, len), false);
  TempGuard tmpdir(make_temp_dir(), true);
  std::string pattern = std::format("{}/p_%04d.ppm", tmpdir.path);

  int idx = acquire_daemon();
  // acquire_daemon() already locked the mutex; adopt it into RAII unique_lock
  std::unique_lock<std::mutex> daemon_lock(daemons_[idx].mutex, std::adopt_lock);
  std::string resp = send_cmd(daemons_[idx],
      std::format("RENDER\t{}\t{}\t{}\t{}\t-1",
                  tmpfile.path, pattern, dpi, workers_per_render_));
  daemon_lock.unlock();

  if (!resp.starts_with("OK"))
    throw turbo_ocr::PdfRenderError(std::format("PDF render failed: {}", resp));

  int num_pages = 0;
  if (resp.starts_with("OK "))
    num_pages = std::stoi(resp.substr(3));

  // Read PPM files — parallel for multi-page PDFs (each read_ppm is
  // independent: thread-safe fopen/fread, creates its own cv::Mat).
  std::vector<cv::Mat> pages(num_pages);
  if (num_pages <= 2) {
    for (int i = 0; i < num_pages; ++i)
      pages[i] = read_ppm(std::format("{}/p_{:04d}.ppm", tmpdir.path, i + 1));
  } else {
    std::vector<std::thread> readers;
    int n_readers = std::min(num_pages, 4);
    readers.reserve(n_readers);
    std::atomic<int> next{0};
    for (int t = 0; t < n_readers; ++t) {
      readers.emplace_back([&]() {
        while (true) {
          int idx = next.fetch_add(1, std::memory_order_relaxed);
          if (idx >= num_pages) break;
          pages[idx] = read_ppm(
              std::format("{}/p_{:04d}.ppm", tmpdir.path, idx + 1));
        }
      });
    }
    for (auto &th : readers) th.join();
  }

  // TempGuard destructors clean up tmpfile and tmpdir automatically
  return pages;
}

// ---------------------------------------------------------------------------
// render_streamed: overlap rendering with OCR using inotify
// ---------------------------------------------------------------------------
// The daemon's RenderMulti forks worker processes that write PPM files
// independently. By watching the output directory with inotify, we can
// detect each PPM file the moment it's closed and invoke the callback
// (typically: read PPM + run OCR) immediately — while the daemon is still
// rendering later pages.
//
// Timeline comparison (20-page PDF, pool_size=5):
//   Sequential:  [render 70ms][read 20ms][OCR 32ms] = 122ms total
//   Streamed:    [render 70ms                      ]
//                     [OCR p1][OCR p2]...[OCR p20]   = ~85ms total
//
// For single-page PDFs, the overhead is negligible (~0.1ms for inotify setup).

int PdfRenderer::render_streamed(const uint8_t *data, size_t len, int dpi,
                                 PageCallback on_page) {
  TempGuard tmpfile(write_temp_pdf(data, len), false);
  TempGuard tmpdir(make_temp_dir(), true);
  std::string pattern = std::format("{}/p_%04d.ppm", tmpdir.path);

  // Set up inotify BEFORE sending RENDER to avoid missing early pages.
  // CLOSE_WRITE fires when a worker finishes writing a PPM file.
  int inotify_fd = inotify_init1(IN_NONBLOCK | IN_CLOEXEC);
  if (inotify_fd < 0)
    throw turbo_ocr::PdfRenderError("inotify_init1 failed");

  int wd = inotify_add_watch(inotify_fd, tmpdir.path.c_str(), IN_CLOSE_WRITE);
  if (wd < 0) {
    close(inotify_fd);
    throw turbo_ocr::PdfRenderError("inotify_add_watch failed");
  }

  // Track which pages have been delivered to avoid duplicates.
  // Uses a bitset-style vector; pages delivered via inotify are marked here
  // so the safety-net scan at the end skips them. We start with a generous
  // pre-allocation (resized as needed when page indices arrive).
  std::vector<bool> delivered(256, false); // pre-alloc for typical PDFs

  // Launch render in a background thread so we can process inotify events
  // concurrently. The daemon mutex is held for the duration of RENDER.
  int idx = acquire_daemon();
  std::atomic<bool> render_done{false};
  std::string render_resp;
  std::exception_ptr render_error;

  std::thread render_thread([&]() {
    try {
      std::unique_lock<std::mutex> daemon_lock(daemons_[idx].mutex,
                                                std::adopt_lock);
      render_resp = send_cmd(daemons_[idx],
          std::format("RENDER\t{}\t{}\t{}\t{}\t-1",
                      tmpfile.path, pattern, dpi, workers_per_render_));
    } catch (...) {
      render_error = std::current_exception();
    }
    render_done.store(true, std::memory_order_release);
  });

  // Helper: parse inotify events and invoke callback for each completed PPM
  int pages_delivered = 0;
  alignas(struct inotify_event) char ev_buf[4096];

  auto process_events = [&]() {
    while (true) {
      auto nread = ::read(inotify_fd, ev_buf, sizeof(ev_buf));
      if (nread <= 0) break;
      for (char *ptr = ev_buf; ptr < ev_buf + nread; ) {
        auto *event = reinterpret_cast<struct inotify_event *>(ptr);
        ptr += sizeof(struct inotify_event) + event->len;
        if (event->len == 0 || !(event->mask & IN_CLOSE_WRITE)) continue;

        // Parse page number from "p_NNNN.ppm"
        std::string_view name(event->name);
        if (!name.starts_with("p_") || !name.ends_with(".ppm")) continue;
        auto num_part = name.substr(2, name.size() - 6);
        int page_num = 0;
        for (char c : num_part) {
          if (c < '0' || c > '9') { page_num = -1; break; }
          page_num = page_num * 10 + (c - '0');
        }
        if (page_num <= 0) continue;

        int page_idx = page_num - 1; // 0-based
        if (page_idx >= static_cast<int>(delivered.size()))
          delivered.resize(page_idx + 1, false);
        if (delivered[page_idx]) continue;
        delivered[page_idx] = true;

        std::string ppm_path = std::format("{}/{}", tmpdir.path, static_cast<const char*>(event->name));
        cv::Mat img = read_ppm(ppm_path);
        if (!img.empty()) {
          on_page(page_idx, std::move(img));
          ++pages_delivered;
        }
      }
    }
  };

  // Poll loop: process inotify events while render is in progress
  struct pollfd pfd = {inotify_fd, POLLIN, 0};
  while (!render_done.load(std::memory_order_acquire)) {
    int ret = poll(&pfd, 1, 2); // 2ms timeout — low latency, low CPU
    if (ret > 0 && (pfd.revents & POLLIN))
      process_events();
  }

  render_thread.join();

  // Drain any remaining inotify events
  process_events();

  // Clean up inotify
  inotify_rm_watch(inotify_fd, wd);
  close(inotify_fd);

  if (render_error) std::rethrow_exception(render_error);

  if (!render_resp.starts_with("OK"))
    throw turbo_ocr::PdfRenderError(
        std::format("PDF render failed: {}", render_resp));

  int num_pages = 0;
  if (render_resp.starts_with("OK "))
    num_pages = std::stoi(render_resp.substr(3));

  // Safety net: deliver any pages missed by inotify (race, coalesced events).
  if (pages_delivered < num_pages) {
    if (num_pages > static_cast<int>(delivered.size()))
      delivered.resize(num_pages, false);
    for (int i = 0; i < num_pages; ++i) {
      if (delivered[i]) continue;
      std::string ppm_path = std::format("{}/p_{:04d}.ppm", tmpdir.path, i + 1);
      if (!std::filesystem::exists(ppm_path)) continue;
      cv::Mat img = read_ppm(ppm_path);
      if (!img.empty()) {
        on_page(i, std::move(img));
        ++pages_delivered;
      }
    }
  }

  return num_pages;
}
