"""Concurrent request handling benchmark.

Measures how the server handles bursts of concurrent requests. Verifies
all requests complete successfully and measures total wall-clock time.

Usage:
    pytest tests/benchmark/bench_concurrent.py -v -s
    python tests/benchmark/bench_concurrent.py [--server-url URL]
"""

import concurrent.futures
import time

import pytest
import requests

from conftest import make_text_image, pil_to_png_bytes


def _get_test_images(n=10):
    """Generate N distinct test images."""
    images = []
    for i in range(n):
        img = make_text_image(f"CONCURRENT{i:03d}", width=400, height=100, font_size=36)
        images.append(pil_to_png_bytes(img))
    return images


class TestConcurrentHandling:
    """Benchmark concurrent request handling."""

    @pytest.mark.parametrize("burst_size", [4, 8, 16, 32])
    def test_burst(self, server_url, burst_size):
        """Send a burst of N requests simultaneously, all should succeed."""
        images = _get_test_images(burst_size)

        def fire(img_bytes):
            t0 = time.perf_counter()
            r = requests.post(
                f"{server_url}/ocr/raw",
                data=img_bytes,
                headers={"Content-Type": "image/png"},
                timeout=60,
            )
            return time.perf_counter() - t0, r.status_code

        t0 = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=burst_size) as pool:
            futures = [pool.submit(fire, img) for img in images]
            results = [f.result() for f in futures]
        wall_time = time.perf_counter() - t0

        times = [t for t, s in results]
        statuses = [s for t, s in results]
        success = sum(1 for s in statuses if s == 200)

        print(f"\n  Burst={burst_size}: "
              f"wall={wall_time:.2f}s, "
              f"avg_per_req={sum(times)/len(times)*1000:.1f}ms, "
              f"success={success}/{burst_size}")

        assert success == burst_size, (
            f"{burst_size - success} requests failed out of {burst_size}"
        )

    def test_sustained_concurrent_load(self, server_url):
        """200 requests at concurrency=8 should all succeed."""
        img = pil_to_png_bytes(make_text_image("SUSTAINED", width=300, height=80, font_size=36))
        total = 200
        concurrency = 8

        def fire():
            r = requests.post(
                f"{server_url}/ocr/raw",
                data=img,
                headers={"Content-Type": "image/png"},
                timeout=60,
            )
            return r.status_code

        t0 = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
            futures = [pool.submit(fire) for _ in range(total)]
            statuses = [f.result() for f in futures]
        elapsed = time.perf_counter() - t0

        success = sum(1 for s in statuses if s == 200)
        throughput = success / elapsed

        print(f"\n  Sustained load: {total} requests, c={concurrency}")
        print(f"    {throughput:.1f} img/s, {elapsed:.2f}s total, {success}/{total} success")

        assert success == total, f"{total - success} requests failed"

    def test_batch_vs_individual_speed(self, server_url):
        """Compare batch endpoint speed vs individual requests.

        Batch should be faster due to reduced HTTP overhead and
        parallel dispatch within the server.
        """
        images = _get_test_images(8)

        # Individual requests (sequential)
        t0 = time.perf_counter()
        for img in images:
            requests.post(
                f"{server_url}/ocr/raw",
                data=img,
                headers={"Content-Type": "image/png"},
                timeout=30,
            )
        individual_time = time.perf_counter() - t0

        # Batch request
        import base64
        b64_images = [base64.b64encode(img).decode("ascii") for img in images]
        t0 = time.perf_counter()
        r = requests.post(
            f"{server_url}/ocr/batch",
            json={"images": b64_images},
            timeout=30,
        )
        batch_time = time.perf_counter() - t0

        print(f"\n  8 images: individual={individual_time:.2f}s, "
              f"batch={batch_time:.2f}s, "
              f"speedup={individual_time/batch_time:.1f}x")

        assert r.status_code == 200


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--server-url", default="http://localhost:8000")
    args = parser.parse_args()

    for burst in [4, 8, 16, 32, 64]:
        images = _get_test_images(burst)

        def fire(img_bytes):
            return requests.post(
                f"{args.server_url}/ocr/raw",
                data=img_bytes,
                headers={"Content-Type": "image/png"},
                timeout=60,
            ).status_code

        t0 = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=burst) as pool:
            statuses = list(pool.map(fire, images))
        elapsed = time.perf_counter() - t0
        success = sum(1 for s in statuses if s == 200)
        print(f"Burst={burst:>3}: wall={elapsed:.2f}s, "
              f"success={success}/{burst}, "
              f"throughput={success/elapsed:.1f}/s")


if __name__ == "__main__":
    main()
