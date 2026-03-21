"""Async latency benchmarks using real test data.

Measures p50, p95, p99 latency using aiohttp for true async I/O.
Compares against baseline: p50=9.5ms, p95=13.3ms, p99=18.2ms (RTX 5090).

Usage:
    pytest tests/benchmark/bench_async_latency.py -v -s
    python tests/benchmark/bench_async_latency.py [--server-url URL]
"""

import asyncio
import statistics
import time
from pathlib import Path

import aiohttp
import pytest

TEST_DATA_DIR = Path(__file__).resolve().parent.parent / "test_data"

# Baseline latency (RTX 5090, c=16)
BASELINE_P50_MS = 9.5
BASELINE_P95_MS = 13.3
BASELINE_P99_MS = 18.2


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


def _load_image(name, subdir="png"):
    path = TEST_DATA_DIR / subdir / name
    if not path.exists():
        return None, None
    suffix = path.suffix.lower()
    content_type = {".png": "image/png", ".jpg": "image/jpeg"}.get(suffix, "image/png")
    return path.read_bytes(), content_type


async def _measure_async_latencies(url, concurrency, total, image_bytes, content_type):
    """Send requests at given concurrency, collect per-request latencies (ms)."""
    semaphore = asyncio.Semaphore(concurrency)
    latencies = []

    async def fire(session):
        async with semaphore:
            t0 = time.perf_counter()
            async with session.post(
                f"{url}/ocr/raw",
                data=image_bytes,
                headers={"Content-Type": content_type},
                timeout=aiohttp.ClientTimeout(total=60),
            ) as resp:
                await resp.read()
                elapsed_ms = (time.perf_counter() - t0) * 1000
                if resp.status == 200:
                    latencies.append(elapsed_ms)

    connector = aiohttp.TCPConnector(limit=concurrency, force_close=False)
    async with aiohttp.ClientSession(connector=connector) as session:
        # Warmup
        for _ in range(min(concurrency, 4)):
            async with session.post(
                f"{url}/ocr/raw",
                data=image_bytes,
                headers={"Content-Type": content_type},
                timeout=aiohttp.ClientTimeout(total=30),
            ) as resp:
                await resp.read()

        tasks = [asyncio.create_task(fire(session)) for _ in range(total)]
        await asyncio.gather(*tasks)

    latencies.sort()
    return latencies


@pytest.fixture(scope="module")
def receipt_image():
    data, ct = _load_image("receipt.png")
    if data is None:
        pytest.skip("receipt.png not found")
    return data, ct


@pytest.fixture(scope="module")
def dense_text_image():
    data, ct = _load_image("dense_text.png")
    if data is None:
        pytest.skip("dense_text.png not found")
    return data, ct


class TestAsyncLatency:
    """Async latency benchmark with real test images."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("concurrency", [1, 4, 8, 16])
    async def test_receipt_latency(self, server_url, concurrency, receipt_image):
        """Measure latency percentiles for receipt.png."""
        image_bytes, content_type = receipt_image
        latencies = await _measure_async_latencies(
            server_url, concurrency, 100, image_bytes, content_type
        )

        if not latencies:
            pytest.fail("All requests failed")

        p50 = _percentile(latencies, 50)
        p95 = _percentile(latencies, 95)
        p99 = _percentile(latencies, 99)
        avg = statistics.mean(latencies)

        print(
            f"\n  [receipt.png] c={concurrency} ({len(latencies)} samples):"
            f"\n    p50={p50:.1f}ms  p95={p95:.1f}ms  p99={p99:.1f}ms  avg={avg:.1f}ms"
            f"\n    baseline: p50={BASELINE_P50_MS}ms  p95={BASELINE_P95_MS}ms  p99={BASELINE_P99_MS}ms"
        )

        # Sanity: p50 under 5 seconds
        assert p50 < 5000, f"p50={p50:.0f}ms is way too high"

    @pytest.mark.asyncio
    @pytest.mark.parametrize("concurrency", [1, 4, 8, 16])
    async def test_dense_latency(self, server_url, concurrency, dense_text_image):
        """Measure latency percentiles for dense_text.png."""
        image_bytes, content_type = dense_text_image
        latencies = await _measure_async_latencies(
            server_url, concurrency, 100, image_bytes, content_type
        )

        if not latencies:
            pytest.fail("All requests failed")

        p50 = _percentile(latencies, 50)
        p95 = _percentile(latencies, 95)
        p99 = _percentile(latencies, 99)
        avg = statistics.mean(latencies)

        print(
            f"\n  [dense_text.png] c={concurrency} ({len(latencies)} samples):"
            f"\n    p50={p50:.1f}ms  p95={p95:.1f}ms  p99={p99:.1f}ms  avg={avg:.1f}ms"
        )

        assert p50 < 5000, f"p50={p50:.0f}ms is way too high"


class TestAsyncLatencySummary:
    """Print a full latency summary table."""

    @pytest.mark.asyncio
    async def test_latency_summary(self, server_url, receipt_image, dense_text_image):
        receipt_bytes, receipt_ct = receipt_image
        dense_bytes, dense_ct = dense_text_image

        print("\n" + "=" * 80)
        print("ASYNC LATENCY BENCHMARK SUMMARY")
        print(f"Baseline (RTX 5090): p50={BASELINE_P50_MS}ms  "
              f"p95={BASELINE_P95_MS}ms  p99={BASELINE_P99_MS}ms")
        print("=" * 80)

        for label, img_bytes, ct in [
            ("receipt.png", receipt_bytes, receipt_ct),
            ("dense_text.png", dense_bytes, dense_ct),
        ]:
            print(f"\n  {label}:")
            print(f"  {'Conc':>6} {'p50':>8} {'p95':>8} {'p99':>8} {'avg':>8} {'min':>8} {'max':>8}")
            print(f"  {'-' * 54}")

            for c in [1, 4, 8, 16, 32]:
                lats = await _measure_async_latencies(
                    server_url, c, 50, img_bytes, ct
                )
                if not lats:
                    print(f"  {c:>6} {'FAIL':>8}")
                    continue
                p50 = _percentile(lats, 50)
                p95 = _percentile(lats, 95)
                p99 = _percentile(lats, 99)
                avg = statistics.mean(lats)
                print(
                    f"  {c:>6} {p50:>7.1f}ms {p95:>7.1f}ms {p99:>7.1f}ms "
                    f"{avg:>7.1f}ms {min(lats):>7.1f}ms {max(lats):>7.1f}ms"
                )

        print("=" * 80)


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------

async def _main():
    import argparse
    parser = argparse.ArgumentParser(description="Async latency benchmark")
    parser.add_argument("--server-url", default="http://localhost:8000")
    parser.add_argument("--iterations", type=int, default=100)
    args = parser.parse_args()

    receipt_bytes, receipt_ct = _load_image("receipt.png")
    dense_bytes, dense_ct = _load_image("dense_text.png")

    if receipt_bytes is None or dense_bytes is None:
        print("ERROR: test images not found in test_data/png/")
        return

    print(f"Server: {args.server_url}")
    print(f"Iterations per level: {args.iterations}")
    print(f"Baseline (RTX 5090): p50={BASELINE_P50_MS}ms  "
          f"p95={BASELINE_P95_MS}ms  p99={BASELINE_P99_MS}ms")
    print()

    for label, img_bytes, ct in [
        ("receipt.png", receipt_bytes, receipt_ct),
        ("dense_text.png", dense_bytes, dense_ct),
    ]:
        print(f"\n{label}:")
        print(f"{'Conc':>6} {'p50':>8} {'p95':>8} {'p99':>8} {'avg':>8} {'min':>8} {'max':>8}")
        print("-" * 60)

        for c in [1, 4, 8, 16, 32]:
            lats = await _measure_async_latencies(
                args.server_url, c, args.iterations, img_bytes, ct
            )
            if not lats:
                print(f"{c:>6} {'FAIL':>8}")
                continue
            p50 = _percentile(lats, 50)
            p95 = _percentile(lats, 95)
            p99 = _percentile(lats, 99)
            avg = statistics.mean(lats)
            print(
                f"{c:>6} {p50:>7.1f}ms {p95:>7.1f}ms {p99:>7.1f}ms "
                f"{avg:>7.1f}ms {min(lats):>7.1f}ms {max(lats):>7.1f}ms"
            )


if __name__ == "__main__":
    asyncio.run(_main())
