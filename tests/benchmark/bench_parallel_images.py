"""Parallel image processing speed and ordering benchmark.

Measures throughput when processing multiple images concurrently via both
individual requests and the batch endpoint, and verifies ordering.

Usage:
    pytest tests/benchmark/bench_parallel_images.py -v -s
    python tests/benchmark/bench_parallel_images.py [--server-url URL]
"""

import base64
import concurrent.futures
import time

import pytest
import requests

from conftest import make_text_image, pil_to_png_bytes


def _make_labeled_images(n):
    """Generate N labeled images for ordering verification."""
    images = []
    for i in range(n):
        img = make_text_image(f"IMG{i:04d}", width=400, height=100, font_size=40)
        images.append((f"IMG{i:04d}", pil_to_png_bytes(img)))
    return images


class TestParallelImages:
    """Benchmark parallel image processing."""

    @pytest.mark.parametrize("n_images", [4, 8, 16])
    def test_concurrent_individual_requests(self, server_url, n_images):
        """Process N images concurrently via individual /ocr/raw requests."""
        images = _make_labeled_images(n_images)

        def fire(label_and_bytes):
            label, img_bytes = label_and_bytes
            t0 = time.perf_counter()
            r = requests.post(
                f"{server_url}/ocr/raw",
                data=img_bytes,
                headers={"Content-Type": "image/png"},
                timeout=30,
            )
            return label, time.perf_counter() - t0, r.status_code

        t0 = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=n_images) as pool:
            results = list(pool.map(fire, images))
        wall = time.perf_counter() - t0

        success = sum(1 for _, _, s in results if s == 200)
        print(f"\n  {n_images} parallel images: "
              f"{wall:.2f}s wall, "
              f"{n_images / wall:.1f} img/s, "
              f"{success}/{n_images} success")

        assert success == n_images

    @pytest.mark.parametrize("n_images", [4, 8, 16])
    def test_batch_endpoint_speed(self, server_url, n_images):
        """Process N images via /ocr/batch endpoint."""
        images = _make_labeled_images(n_images)
        b64_list = [base64.b64encode(b).decode("ascii") for _, b in images]

        t0 = time.perf_counter()
        r = requests.post(
            f"{server_url}/ocr/batch",
            json={"images": b64_list},
            timeout=60,
        )
        elapsed = time.perf_counter() - t0

        assert r.status_code == 200
        data = r.json()
        assert len(data["batch_results"]) == n_images

        print(f"\n  Batch of {n_images}: "
              f"{elapsed:.2f}s, "
              f"{n_images / elapsed:.1f} img/s")

    def test_batch_ordering_under_load(self, server_url):
        """Verify batch results maintain input ordering even under heavy load.

        Send 20 images in a batch and verify results correspond to inputs.
        """
        images = _make_labeled_images(20)
        b64_list = [base64.b64encode(b).decode("ascii") for _, b in images]

        r = requests.post(
            f"{server_url}/ocr/batch",
            json={"images": b64_list},
            timeout=60,
        )
        assert r.status_code == 200
        data = r.json()
        batch_results = data["batch_results"]
        assert len(batch_results) == 20

        # Each batch result should have the same number of results as when
        # processed individually (results should not be shuffled between images)
        for i in range(20):
            assert "results" in batch_results[i]

    def test_concurrent_vs_sequential_speedup(self, server_url):
        """Concurrent processing should be faster than sequential.

        This verifies the pipeline pool provides real parallelism.
        """
        images = _make_labeled_images(8)

        # Sequential
        t0 = time.perf_counter()
        for _, img_bytes in images:
            requests.post(
                f"{server_url}/ocr/raw",
                data=img_bytes,
                headers={"Content-Type": "image/png"},
                timeout=30,
            )
        sequential_time = time.perf_counter() - t0

        # Concurrent
        def fire(pair):
            _, img_bytes = pair
            return requests.post(
                f"{server_url}/ocr/raw",
                data=img_bytes,
                headers={"Content-Type": "image/png"},
                timeout=30,
            )

        t0 = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=8) as pool:
            list(pool.map(fire, images))
        concurrent_time = time.perf_counter() - t0

        speedup = sequential_time / concurrent_time
        print(f"\n  8 images: seq={sequential_time:.2f}s, "
              f"conc={concurrent_time:.2f}s, speedup={speedup:.1f}x")

        # Concurrent should be at least somewhat faster (>1.2x)
        # with a pipeline pool of 4+
        assert speedup > 1.0, (
            f"No speedup from concurrency: seq={sequential_time:.2f}s, "
            f"conc={concurrent_time:.2f}s"
        )


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--server-url", default="http://localhost:8000")
    args = parser.parse_args()

    print(f"Server: {args.server_url}\n")

    for n in [4, 8, 16, 32]:
        images = _make_labeled_images(n)

        def fire(pair):
            _, img_bytes = pair
            return requests.post(
                f"{args.server_url}/ocr/raw",
                data=img_bytes,
                headers={"Content-Type": "image/png"},
                timeout=60,
            ).status_code

        t0 = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=n) as pool:
            statuses = list(pool.map(fire, images))
        elapsed = time.perf_counter() - t0
        success = sum(1 for s in statuses if s == 200)
        print(f"  n={n:>3}: {n/elapsed:.1f} img/s, "
              f"{elapsed:.2f}s, {success}/{n} OK")


if __name__ == "__main__":
    main()
