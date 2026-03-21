"""Throughput benchmark: measure images/second at various concurrency levels.

Usage:
    pytest tests/benchmark/bench_throughput.py -v -s
    python tests/benchmark/bench_throughput.py [--server-url URL] [--iterations N]
"""

import concurrent.futures
import statistics
import time

import pytest
import requests

from conftest import make_text_image, pil_to_png_bytes

# Default test image: simulated A4-like document
_DOC_IMAGE = None


def _get_doc_image():
    global _DOC_IMAGE
    if _DOC_IMAGE is None:
        from test_data.generate_test_images import generate_dense_document
        _DOC_IMAGE = pil_to_png_bytes(generate_dense_document())
    return _DOC_IMAGE


def _measure_throughput(server_url, concurrency, total_requests, image_bytes):
    """Send total_requests at given concurrency, return img/s."""
    def fire():
        r = requests.post(
            f"{server_url}/ocr/raw",
            data=image_bytes,
            headers={"Content-Type": "image/png"},
            timeout=60,
        )
        return r.status_code

    t0 = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        futures = [pool.submit(fire) for _ in range(total_requests)]
        statuses = [f.result() for f in futures]
    elapsed = time.perf_counter() - t0

    success = sum(1 for s in statuses if s == 200)
    throughput = success / elapsed
    return throughput, elapsed, success, total_requests


class TestThroughput:
    """Benchmark throughput at various concurrency levels."""

    CONCURRENCY_LEVELS = [1, 2, 4, 8, 16, 32]
    REQUESTS_PER_LEVEL = 50  # Total requests per concurrency level

    @pytest.mark.parametrize("concurrency", CONCURRENCY_LEVELS)
    def test_throughput(self, server_url, concurrency):
        """Measure throughput at given concurrency. Always passes (benchmark)."""
        image_bytes = _get_doc_image()

        # Warmup
        for _ in range(min(concurrency, 4)):
            requests.post(
                f"{server_url}/ocr/raw",
                data=image_bytes,
                headers={"Content-Type": "image/png"},
                timeout=30,
            )

        throughput, elapsed, success, total = _measure_throughput(
            server_url, concurrency, self.REQUESTS_PER_LEVEL, image_bytes
        )

        print(f"\n  Concurrency={concurrency}: "
              f"{throughput:.1f} img/s, "
              f"{elapsed:.2f}s total, "
              f"{success}/{total} succeeded")

        # Sanity check: at least 1 img/s (catches totally broken setups)
        assert throughput > 1.0, f"Throughput {throughput:.1f} img/s is suspiciously low"


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser(description="Throughput benchmark")
    parser.add_argument("--server-url", default="http://localhost:8000")
    parser.add_argument("--iterations", type=int, default=50)
    args = parser.parse_args()

    image_bytes = _get_doc_image()

    print(f"Server: {args.server_url}")
    print(f"Iterations per level: {args.iterations}")
    print(f"Image size: {len(image_bytes):,} bytes")
    print()
    print(f"{'Concurrency':>12} {'Throughput':>12} {'Elapsed':>10} {'Success':>10}")
    print("-" * 50)

    for c in [1, 2, 4, 8, 16, 32]:
        tp, elapsed, success, total = _measure_throughput(
            args.server_url, c, args.iterations, image_bytes
        )
        print(f"{c:>12} {tp:>10.1f}/s {elapsed:>9.2f}s {success:>5}/{total}")


if __name__ == "__main__":
    main()
