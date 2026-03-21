"""Async throughput benchmarks using real test data.

Measures actual server throughput using aiohttp for true async I/O,
eliminating Python GIL overhead from measurements. Tests at multiple
concurrency levels with both small (receipt) and large (mixed_fonts) images.

Baseline (RTX 5090, hey client): 246 A4/s, 1200+ receipts/s

Usage:
    pytest tests/benchmark/bench_async_throughput.py -v -s
    python tests/benchmark/bench_async_throughput.py [--server-url URL]
"""

import asyncio
import time
from pathlib import Path

import aiohttp
import pytest

TEST_DATA_DIR = Path(__file__).resolve().parent.parent / "test_data"

CONCURRENCY_LEVELS = [1, 4, 8, 16, 32]
REQUESTS_PER_LEVEL = 100


def _load_image(name, subdir="png"):
    """Load a test image from test_data/."""
    path = TEST_DATA_DIR / subdir / name
    if not path.exists():
        return None, None
    suffix = path.suffix.lower()
    content_type = {".png": "image/png", ".jpg": "image/jpeg"}.get(suffix, "image/png")
    return path.read_bytes(), content_type


async def _measure_async_throughput(url, concurrency, total, image_bytes, content_type):
    """Send total requests at given concurrency using aiohttp, return img/s."""
    semaphore = asyncio.Semaphore(concurrency)
    success = 0

    async def fire(session):
        nonlocal success
        async with semaphore:
            async with session.post(
                f"{url}/ocr/raw",
                data=image_bytes,
                headers={"Content-Type": content_type},
                timeout=aiohttp.ClientTimeout(total=60),
            ) as resp:
                if resp.status == 200:
                    await resp.read()
                    success += 1

    t0 = time.perf_counter()
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

        success = 0
        t0 = time.perf_counter()
        tasks = [asyncio.create_task(fire(session)) for _ in range(total)]
        await asyncio.gather(*tasks)

    elapsed = time.perf_counter() - t0
    throughput = success / elapsed if elapsed > 0 else 0
    return throughput, elapsed, success, total


@pytest.fixture(scope="module")
def receipt_image():
    data, ct = _load_image("receipt.png")
    if data is None:
        pytest.skip("receipt.png not found in test_data/png/")
    return data, ct


@pytest.fixture(scope="module")
def mixed_fonts_image():
    data, ct = _load_image("mixed_fonts.png")
    if data is None:
        pytest.skip("mixed_fonts.png not found in test_data/png/")
    return data, ct


@pytest.fixture(scope="module")
def dense_text_image():
    data, ct = _load_image("dense_text.png")
    if data is None:
        pytest.skip("dense_text.png not found in test_data/png/")
    return data, ct


class TestAsyncThroughputReceipt:
    """Async throughput benchmark with receipt.png (small/fast image)."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("concurrency", CONCURRENCY_LEVELS)
    async def test_receipt_throughput(self, server_url, concurrency, receipt_image):
        image_bytes, content_type = receipt_image
        throughput, elapsed, success, total = await _measure_async_throughput(
            server_url, concurrency, REQUESTS_PER_LEVEL, image_bytes, content_type
        )

        print(
            f"\n  [receipt.png] c={concurrency}: "
            f"{throughput:.1f} img/s, "
            f"{elapsed:.2f}s total, "
            f"{success}/{total} OK"
        )

        # Sanity: should achieve at least 5 img/s even on slow hardware
        assert throughput > 5.0, f"Throughput {throughput:.1f} img/s is too low"


class TestAsyncThroughputDense:
    """Async throughput benchmark with mixed_fonts.png (large/dense image)."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("concurrency", CONCURRENCY_LEVELS)
    async def test_dense_throughput(self, server_url, concurrency, mixed_fonts_image):
        image_bytes, content_type = mixed_fonts_image
        throughput, elapsed, success, total = await _measure_async_throughput(
            server_url, concurrency, REQUESTS_PER_LEVEL, image_bytes, content_type
        )

        print(
            f"\n  [mixed_fonts.png] c={concurrency}: "
            f"{throughput:.1f} img/s, "
            f"{elapsed:.2f}s total, "
            f"{success}/{total} OK"
        )

        assert throughput > 1.0, f"Throughput {throughput:.1f} img/s is too low"


class TestAsyncThroughputSummary:
    """Run all concurrency levels and print a summary table."""

    @pytest.mark.asyncio
    async def test_throughput_summary(self, server_url, receipt_image, dense_text_image):
        """Print a full throughput summary table."""
        receipt_bytes, receipt_ct = receipt_image
        dense_bytes, dense_ct = dense_text_image

        print("\n" + "=" * 70)
        print("ASYNC THROUGHPUT BENCHMARK SUMMARY")
        print("=" * 70)
        print(f"{'Concurrency':>12} {'Receipt (img/s)':>18} {'Dense (img/s)':>18}")
        print("-" * 50)

        for c in CONCURRENCY_LEVELS:
            r_tp, _, _, _ = await _measure_async_throughput(
                server_url, c, 50, receipt_bytes, receipt_ct
            )
            d_tp, _, _, _ = await _measure_async_throughput(
                server_url, c, 50, dense_bytes, dense_ct
            )
            print(f"{c:>12} {r_tp:>16.1f}/s {d_tp:>16.1f}/s")

        print("=" * 70)


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------

async def _main():
    import argparse
    parser = argparse.ArgumentParser(description="Async throughput benchmark")
    parser.add_argument("--server-url", default="http://localhost:8000")
    parser.add_argument("--requests", type=int, default=100)
    args = parser.parse_args()

    receipt_bytes, receipt_ct = _load_image("receipt.png")
    dense_bytes, dense_ct = _load_image("dense_text.png")

    if receipt_bytes is None:
        print("ERROR: receipt.png not found in test_data/png/")
        return
    if dense_bytes is None:
        print("ERROR: dense_text.png not found in test_data/png/")
        return

    print(f"Server: {args.server_url}")
    print(f"Requests per level: {args.requests}")
    print(f"Receipt image: {len(receipt_bytes):,} bytes")
    print(f"Dense image: {len(dense_bytes):,} bytes")
    print()
    print(f"{'Concurrency':>12} {'Receipt (img/s)':>18} {'Dense (img/s)':>18}")
    print("-" * 50)

    for c in CONCURRENCY_LEVELS:
        r_tp, _, r_ok, _ = await _measure_async_throughput(
            args.server_url, c, args.requests, receipt_bytes, receipt_ct
        )
        d_tp, _, d_ok, _ = await _measure_async_throughput(
            args.server_url, c, args.requests, dense_bytes, dense_ct
        )
        print(f"{c:>12} {r_tp:>16.1f}/s {d_tp:>16.1f}/s")


if __name__ == "__main__":
    asyncio.run(_main())
