"""Parallel PDF processing speed and ordering benchmark.

Tests PDF processing throughput and verifies page ordering under load.

Usage:
    pytest tests/benchmark/bench_parallel_pdf.py -v -s
    python tests/benchmark/bench_parallel_pdf.py [--server-url URL]
"""

import concurrent.futures
import io
import time

import pytest
import requests

try:
    from reportlab.lib.pagesizes import A4
    from reportlab.pdfgen import canvas
    _HAS_REPORTLAB = True
except ImportError:
    _HAS_REPORTLAB = False


def _make_pdf(pages, text_prefix="PAGE"):
    """Generate a PDF with N pages, each with unique text."""
    buf = io.BytesIO()
    c = canvas.Canvas(buf, pagesize=A4)
    for i in range(pages):
        c.setFont("Helvetica", 36)
        c.drawString(100, 700, f"{text_prefix}{i + 1}")
        c.drawString(100, 650, f"Content for page number {i + 1}")
        c.showPage()
    c.save()
    return buf.getvalue()


@pytest.mark.skipif(not _HAS_REPORTLAB, reason="reportlab not installed")
class TestParallelPdf:
    """Benchmark parallel PDF processing."""

    def test_single_page_latency(self, server_url):
        """Baseline latency for a single-page PDF."""
        pdf = _make_pdf(1)
        t0 = time.perf_counter()
        r = requests.post(f"{server_url}/ocr/pdf", data=pdf, timeout=30)
        elapsed = (time.perf_counter() - t0) * 1000
        assert r.status_code == 200
        print(f"\n  Single page PDF: {elapsed:.1f}ms")

    @pytest.mark.parametrize("num_pages", [5, 10, 20])
    def test_multi_page_throughput(self, server_url, num_pages):
        """Measure pages/second for multi-page PDFs."""
        pdf = _make_pdf(num_pages)
        t0 = time.perf_counter()
        r = requests.post(f"{server_url}/ocr/pdf", data=pdf, timeout=120)
        elapsed = time.perf_counter() - t0
        assert r.status_code == 200
        data = r.json()
        assert len(data["pages"]) == num_pages

        pages_per_sec = num_pages / elapsed
        per_page_ms = elapsed / num_pages * 1000
        print(f"\n  {num_pages}-page PDF: "
              f"{pages_per_sec:.1f} pages/s, "
              f"{per_page_ms:.1f}ms/page, "
              f"{elapsed:.2f}s total")

    def test_concurrent_pdfs(self, server_url):
        """Send multiple PDFs concurrently."""
        pdfs = [_make_pdf(5, f"DOC{i}") for i in range(4)]

        def fire(pdf_bytes):
            return requests.post(f"{server_url}/ocr/pdf", data=pdf_bytes, timeout=60)

        t0 = time.perf_counter()
        with concurrent.futures.ThreadPoolExecutor(max_workers=4) as pool:
            responses = list(pool.map(fire, pdfs))
        elapsed = time.perf_counter() - t0

        total_pages = sum(len(r.json()["pages"]) for r in responses if r.status_code == 200)
        print(f"\n  4 concurrent 5-page PDFs: "
              f"{total_pages / elapsed:.1f} pages/s, "
              f"{elapsed:.2f}s total")

        for r in responses:
            assert r.status_code == 200

    def test_page_ordering_preserved(self, server_url):
        """Verify page ordering is preserved for multi-page PDF under load."""
        pdf = _make_pdf(10, "PAGE")
        r = requests.post(f"{server_url}/ocr/pdf", data=pdf, timeout=60)
        assert r.status_code == 200
        data = r.json()
        pages = data["pages"]
        assert len(pages) == 10
        for i, page in enumerate(pages):
            assert page["page"] == i + 1, (
                f"Page ordering broken: expected page {i+1}, got {page['page']}"
            )


# ---------------------------------------------------------------------------
# Standalone runner
# ---------------------------------------------------------------------------

def main():
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--server-url", default="http://localhost:8000")
    args = parser.parse_args()

    if not _HAS_REPORTLAB:
        print("Install reportlab: pip install reportlab")
        return

    print(f"Server: {args.server_url}\n")

    for pages in [1, 5, 10, 20, 50]:
        pdf = _make_pdf(pages)
        t0 = time.perf_counter()
        r = requests.post(f"{args.server_url}/ocr/pdf", data=pdf, timeout=120)
        elapsed = time.perf_counter() - t0
        if r.status_code == 200:
            print(f"  {pages:>3} pages: {pages/elapsed:.1f} pg/s, "
                  f"{elapsed/pages*1000:.1f}ms/pg, {elapsed:.2f}s total")
        else:
            print(f"  {pages:>3} pages: ERROR {r.status_code}")


if __name__ == "__main__":
    main()
