#pragma once

#include <atomic>
#include <chrono>
#include <condition_variable>
#include <functional>
#include <mutex>
#include <queue>
#include <thread>
#include <vector>

#include "turbo_ocr/common/errors.h"

namespace turbo_ocr::server {

/// Thread pool for offloading blocking HTTP handler work from Drogon's
/// event-loop threads.
///
/// submit() is non-blocking: it always enqueues and returns immediately.
/// Backpressure is handled downstream — the PipelineDispatcher rejects
/// with PoolExhaustedError when the GPU queue is full, and the
/// run_with_error_handling wrapper converts that to a 503 response.
///
/// Queue depth is bounded (default 8192) as a safety net against memory
/// exhaustion.  When full, submit() throws PoolExhaustedError.
class WorkPool {
public:
  explicit WorkPool(int num_threads, size_t max_depth = 8192)
      : max_depth_(max_depth) {
    workers_.reserve(num_threads);
    for (int i = 0; i < num_threads; ++i) {
      workers_.emplace_back([this] {
        while (true) {
          std::function<void()> task;
          {
            std::unique_lock lock(mutex_);
            // Predicate form is only safe against lost wakeups when the
            // notifier mutates state under the same mutex — see dtor.
            cv_.wait(lock, [this] { return stop_ || !queue_.empty(); });
            if (queue_.empty()) {
              if (stop_) return;
              continue;
            }
            task = std::move(queue_.front());
            queue_.pop();
            ++inflight_;
          }
          // RAII inflight decrement: even if `task()` escapes with an
          // exception, inflight_ MUST drop back to 0 — otherwise
          // wait_drain() hangs forever (the graceful-shutdown path
          // depends on inflight_ reaching 0). Tasks submitted via
          // submit_work() are wrapped in run_with_error_handling, which
          // catches everything; this guard is the second layer of
          // defence for any future call site that submits raw lambdas.
          struct InflightGuard {
            WorkPool *self;
            ~InflightGuard() noexcept {
              std::lock_guard lock(self->mutex_);
              --self->inflight_;
              if (self->inflight_ == 0 && self->queue_.empty())
                self->drain_cv_.notify_all();
            }
          } guard{this};
          try {
            task();
          } catch (...) {
            // Swallowed: a task escaping is a bug at the call site, but
            // we cannot let it kill this worker (the pool would slowly
            // bleed threads until wait_drain() hangs).
          }
        }
      });
    }
  }

  ~WorkPool() {
    {
      std::lock_guard lock(mutex_);
      stop_ = true;
    }
    cv_.notify_all();
    for (auto &w : workers_)
      if (w.joinable()) w.join();
  }

  WorkPool(const WorkPool &) = delete;
  WorkPool &operator=(const WorkPool &) = delete;

  void submit(std::function<void()> fn) {
    {
      std::lock_guard lock(mutex_);
      if (queue_.size() >= max_depth_)
        throw turbo_ocr::PoolExhaustedError(
            "Server at capacity (work queue full). Use persistent connections "
            "(HTTP keep-alive) instead of opening a new connection per request.");
      queue_.push(std::move(fn));
    }
    cv_.notify_one();
  }

  /// Block until queue is empty and no task is in flight, OR timeout
  /// elapses. Returns true on full drain, false on timeout. Used by the
  /// graceful-shutdown path: caller stops admitting new work first, then
  /// waits here for inflight to finish before tearing down Drogon.
  bool wait_drain(std::chrono::milliseconds timeout) {
    std::unique_lock lock(mutex_);
    return drain_cv_.wait_for(lock, timeout, [this] {
      return queue_.empty() && inflight_ == 0;
    });
  }

private:
  std::vector<std::thread> workers_;
  std::queue<std::function<void()>> queue_;
  std::mutex mutex_;
  std::condition_variable cv_;
  std::condition_variable drain_cv_;
  bool stop_{false};        // guarded by mutex_
  size_t inflight_{0};      // guarded by mutex_
  size_t max_depth_;
};

} // namespace turbo_ocr::server
