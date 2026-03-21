"""Latency benchmark: measure p50, p95, p99 latency at various concurrency levels.

Usage:
    pytest tests/benchmark/bench_latency.py -v -s
    python tests/benchmark/bench_latency.py [--server-url URL]
"""

import concurrent.futures
import statistics
import time

import pytest
import requests

from conftest import make_text_image, pil_to_png_bytes


def _get_test_image():
    from test_data.generate_test_images import generate_dense_document
    return pil_to_png_bytes(generate_dense_document())


def _percentile(data, p):
    """Compute the p-th percentile of a sorted list."""
    if not data:
        return 0
    k = (len(data) - 1) * (p / 100)
    f = int(k)
    c = f + 1
    if c >= len(data):
        return data[f]
    return data[f] + (k - f) * (data[c] - data[f])


def _measure_latencies(server_url, concurrency, total_requests, image_bytes):
    """Send requests at given concurrency, collect per-request latencies."""
    latencies = []

    def fire():
        t0 = time.perf_counter()
        r = requests.post(
            f"{server_url}/ocr/raw",
            data=image_bytes,
            headers={"Content-Type": "image/png"},
            timeout=60,
        )
        elapsed = time.perf_counter() - t0
        return elapsed * 1000, r.status_code  # ms

    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(fire) for _ in range(total_requests)]
        for f in concurrent.futures.as_completed(futures):
            ms, status = f.result()
            if status == 200:
                latencies.append(ms)

    latencies.sort()
    return latencies


class TestLatency:
    """Benchmark latency percentiles."""

    @pytest.mark.parametrize("concurrency", [1, 4, 8, 16])
    def test_latency_percentiles(self, server_url, concurrency):
        """Measure and report latency percentiles. Always passes (benchmark)."""
        image_bytes = _get_test_image()
        total = 100

        # Warmup
        for _ in range(4):
            requests.post(
                f"{server_url}/ocr/raw",
                data=image_bytes,
                headers={"Content-Type": "image/png"},
                timeout=30,
            )

        latencies = _measure_latencies(server_url, concurrency, total, image_bytes)

        if not latencies:
            pytest.fail("All requests failed")

        p50 = _percentile(latencies, 50)
        p95 = _percentile(latencies, 95)
        p99 = _percentile(latencies, 99)
        avg = statistics.mean(latencies)
        mn = min(latencies)
        mx = max(latencies)

        print(f"\n  Concurrency={concurrency} ({len(latencies)} samples):")
        print(f"    p50={p50:.1f}ms  p95={p95:.1f}ms  p99={p99:.1f}ms")
        print(f"    avg={avg:.1f}ms  min={mn:.1f}ms  max={mx:.1f}ms")

        # Sanity: p50 should be under 5 seconds (catches major issues)
        assert p50 < 5000, f"p50 latency {p50:.0f}ms is way too high"


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Latency benchmark")
    parser.add_argument("--server-url", default="http://localhost:8000")
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()

    image_bytes = _get_test_image()

    print(f"Server: {args.server_url}")
    print(f"Iterations per level: {args.iterations}")
    print()
    print(f"{'Conc':>6} {'p50':>8} {'p95':>8} {'p99':>8} {'avg':>8} {'min':>8} {'max':>8}")
    print("-" * 60)

    for c in [1, 2, 4, 8, 16, 32]:
        latencies = _measure_latencies(args.server_url, c, args.iterations, image_bytes)
        if not latencies:
            print(f"{c:>6} {'FAIL':>8}")
            continue
        p50 = _percentile(latencies, 50)
        p95 = _percentile(latencies, 95)
        p99 = _percentile(latencies, 99)
        avg = statistics.mean(latencies)
        print(f"{c:>6} {p50:>7.1f}ms {p95:>7.1f}ms {p99:>7.1f}ms "
              f"{avg:>7.1f}ms {min(latencies):>7.1f}ms {max(latencies):>7.1f}ms")


if __name__ == "__main__":
    main()
