"""Async PDF throughput benchmark using real test PDFs.

Measures pages/second for each PDF in tests/test_data/pdf/ and tests
concurrent PDF processing throughput.

Usage:
    pytest tests/benchmark/bench_pdf_throughput.py -v -s
    python tests/benchmark/bench_pdf_throughput.py [--server-url URL]
"""

import asyncio
import time
from pathlib import Path

import aiohttp
import pytest

TEST_DATA_DIR = Path(__file__).resolve().parent.parent / "test_data"
PDF_DIR = TEST_DATA_DIR / "pdf"

_PDF_FILES = sorted(PDF_DIR.glob("*.pdf")) if PDF_DIR.exists() else []


def _load_pdf(path):
    """Load PDF bytes from path."""
    return path.read_bytes()


async def _ocr_pdf_async(session, url, pdf_bytes, timeout=120):
    """Send PDF to /ocr/pdf asynchronously, return (status, json, elapsed_ms)."""
    t0 = time.perf_counter()
    async with session.post(
        f"{url}/ocr/pdf",
        data=pdf_bytes,
        timeout=aiohttp.ClientTimeout(total=timeout),
    ) as resp:
        data = await resp.json() if resp.status == 200 else None
        elapsed_ms = (time.perf_counter() - t0) * 1000
        return resp.status, data, elapsed_ms


@pytest.fixture(scope="module")
def pdf_files_data():
    """Load all PDF test files as (name, bytes) tuples."""
    if not _PDF_FILES:
        pytest.skip("No PDF test files found")
    return [(p.name, p.read_bytes()) for p in _PDF_FILES]


class TestPdfThroughput:
    """Measure pages/second for individual PDF files."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize(
        "pdf_path",
        _PDF_FILES,
        ids=[f.stem for f in _PDF_FILES],
    )
    async def test_individual_pdf_throughput(self, server_url, pdf_path):
        """Measure pages/second for a single PDF."""
        pdf_bytes = pdf_path.read_bytes()

        connector = aiohttp.TCPConnector(limit=1)
        async with aiohttp.ClientSession(connector=connector) as session:
            # Warmup
            await _ocr_pdf_async(session, server_url, pdf_bytes)

            # Measure
            status, data, elapsed_ms = await _ocr_pdf_async(session, server_url, pdf_bytes)

        assert status == 200, f"{pdf_path.name}: HTTP {status}"
        num_pages = len(data.get("pages", []))
        pages_per_sec = num_pages / (elapsed_ms / 1000) if elapsed_ms > 0 else 0
        ms_per_page = elapsed_ms / num_pages if num_pages > 0 else 0

        print(
            f"\n  {pdf_path.name}: "
            f"{num_pages} pages, "
            f"{pages_per_sec:.1f} pages/s, "
            f"{ms_per_page:.1f} ms/page, "
            f"{elapsed_ms:.0f}ms total"
        )


class TestConcurrentPdfThroughput:
    """Measure throughput when processing multiple PDFs concurrently."""

    @pytest.mark.asyncio
    @pytest.mark.parametrize("concurrency", [1, 2, 4])
    async def test_concurrent_pdf_throughput(self, server_url, concurrency, pdf_files_data):
        """Process all PDFs at given concurrency, measure total pages/second."""
        semaphore = asyncio.Semaphore(concurrency)
        total_pages = 0
        total_success = 0

        async def fire(session, name, pdf_bytes):
            nonlocal total_pages, total_success
            async with semaphore:
                status, data, _ = await _ocr_pdf_async(session, server_url, pdf_bytes)
                if status == 200 and data:
                    total_pages += len(data.get("pages", []))
                    total_success += 1

        connector = aiohttp.TCPConnector(limit=concurrency, force_close=False)
        async with aiohttp.ClientSession(connector=connector) as session:
            t0 = time.perf_counter()
            tasks = [
                asyncio.create_task(fire(session, name, pdf_bytes))
                for name, pdf_bytes in pdf_files_data
            ]
            await asyncio.gather(*tasks)
            elapsed = time.perf_counter() - t0

        pages_per_sec = total_pages / elapsed if elapsed > 0 else 0
        print(
            f"\n  c={concurrency}: "
            f"{total_success}/{len(pdf_files_data)} PDFs, "
            f"{total_pages} total pages, "
            f"{pages_per_sec:.1f} pages/s, "
            f"{elapsed:.2f}s total"
        )

        assert total_success == len(pdf_files_data), (
            f"{len(pdf_files_data) - total_success} PDFs failed at c={concurrency}"
        )

    @pytest.mark.asyncio
    async def test_repeated_pdf_throughput(self, server_url, pdf_files_data):
        """Send the same PDF 10 times concurrently for sustained throughput."""
        # Pick the smallest PDF for speed
        name, pdf_bytes = min(pdf_files_data, key=lambda x: len(x[1]))
        concurrency = 8
        repetitions = 10
        semaphore = asyncio.Semaphore(concurrency)
        total_pages = 0

        async def fire(session):
            nonlocal total_pages
            async with semaphore:
                status, data, _ = await _ocr_pdf_async(session, server_url, pdf_bytes)
                if status == 200 and data:
                    total_pages += len(data.get("pages", []))

        connector = aiohttp.TCPConnector(limit=concurrency, force_close=False)
        async with aiohttp.ClientSession(connector=connector) as session:
            t0 = time.perf_counter()
            tasks = [asyncio.create_task(fire(session)) for _ in range(repetitions)]
            await asyncio.gather(*tasks)
            elapsed = time.perf_counter() - t0

        pages_per_sec = total_pages / elapsed if elapsed > 0 else 0
        print(
            f"\n  Repeated {name} x{repetitions} (c={concurrency}): "
            f"{total_pages} pages, "
            f"{pages_per_sec:.1f} pages/s, "
            f"{elapsed:.2f}s"
        )


class TestPdfThroughputSummary:
    """Print a comprehensive PDF throughput summary."""

    @pytest.mark.asyncio
    async def test_pdf_summary(self, server_url, pdf_files_data):
        """Print a table of per-PDF throughput."""
        print("\n" + "=" * 70)
        print("PDF THROUGHPUT BENCHMARK SUMMARY")
        print("=" * 70)
        print(f"{'PDF Name':<30} {'Pages':>6} {'Pages/s':>10} {'ms/page':>10} {'Total ms':>10}")
        print("-" * 70)

        connector = aiohttp.TCPConnector(limit=1)
        async with aiohttp.ClientSession(connector=connector) as session:
            for name, pdf_bytes in pdf_files_data:
                # Warmup
                await _ocr_pdf_async(session, server_url, pdf_bytes)
                # Measure (average of 3 runs)
                times = []
                page_counts = []
                for _ in range(3):
                    status, data, elapsed_ms = await _ocr_pdf_async(
                        session, server_url, pdf_bytes
                    )
                    if status == 200 and data:
                        times.append(elapsed_ms)
                        page_counts.append(len(data.get("pages", [])))

                if times:
                    avg_ms = sum(times) / len(times)
                    num_pages = page_counts[0]
                    pages_per_sec = num_pages / (avg_ms / 1000) if avg_ms > 0 else 0
                    ms_per_page = avg_ms / num_pages if num_pages > 0 else 0
                    print(
                        f"  {name:<28} {num_pages:>6} "
                        f"{pages_per_sec:>9.1f}/s {ms_per_page:>9.1f}ms {avg_ms:>9.0f}ms"
                    )
                else:
                    print(f"  {name:<28} {'FAIL':>6}")

        print("=" * 70)


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------

async def _main():
    import argparse
    parser = argparse.ArgumentParser(description="PDF throughput benchmark")
    parser.add_argument("--server-url", default="http://localhost:8000")
    args = parser.parse_args()

    pdf_files = sorted(PDF_DIR.glob("*.pdf"))
    if not pdf_files:
        print("No PDF test files found in test_data/pdf/")
        return

    print(f"Server: {args.server_url}")
    print(f"PDFs: {len(pdf_files)}")
    print()
    print(f"{'PDF Name':<30} {'Pages':>6} {'Pages/s':>10} {'ms/page':>10} {'Total ms':>10}")
    print("-" * 70)

    connector = aiohttp.TCPConnector(limit=1)
    async with aiohttp.ClientSession(connector=connector) as session:
        for pdf_path in pdf_files:
            pdf_bytes = pdf_path.read_bytes()
            status, data, elapsed_ms = await _ocr_pdf_async(
                session, args.server_url, pdf_bytes
            )
            if status == 200 and data:
                num_pages = len(data.get("pages", []))
                pages_per_sec = num_pages / (elapsed_ms / 1000) if elapsed_ms > 0 else 0
                ms_per_page = elapsed_ms / num_pages if num_pages > 0 else 0
                print(
                    f"  {pdf_path.name:<28} {num_pages:>6} "
                    f"{pages_per_sec:>9.1f}/s {ms_per_page:>9.1f}ms {elapsed_ms:>9.0f}ms"
                )
            else:
                print(f"  {pdf_path.name:<28} ERROR {status}")

    # Concurrent test
    print(f"\nConcurrent PDF processing:")
    for c in [1, 2, 4]:
        pdf_data = [(p.name, p.read_bytes()) for p in pdf_files]
        semaphore = asyncio.Semaphore(c)
        total_pages = 0

        async def fire(session, name, pdf_bytes):
            nonlocal total_pages
            async with semaphore:
                status, data, _ = await _ocr_pdf_async(
                    session, args.server_url, pdf_bytes
                )
                if status == 200 and data:
                    total_pages += len(data.get("pages", []))

        connector = aiohttp.TCPConnector(limit=c, force_close=False)
        async with aiohttp.ClientSession(connector=connector) as session:
            total_pages = 0
            t0 = time.perf_counter()
            tasks = [
                asyncio.create_task(fire(session, n, b))
                for n, b in pdf_data
            ]
            await asyncio.gather(*tasks)
            elapsed = time.perf_counter() - t0

        pages_per_sec = total_pages / elapsed if elapsed > 0 else 0
        print(f"  c={c}: {total_pages} pages, {pages_per_sec:.1f} pages/s, {elapsed:.2f}s")


if __name__ == "__main__":
    asyncio.run(_main())
