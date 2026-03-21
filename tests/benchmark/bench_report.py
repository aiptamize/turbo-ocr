#!/usr/bin/env python3
"""Generate a markdown benchmark report.

Runs all benchmarks and produces a formatted report with tables.

Usage:
    python tests/benchmark/bench_report.py [--server-url URL] [--output report.md]
"""

import argparse
import base64
import concurrent.futures
import datetime
import io
import statistics
import sys
import time
from pathlib import Path

import requests

# Add parent directory to path for imports
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from conftest import make_text_image, pil_to_png_bytes


def _get_doc_image():
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "test_data"))
    from generate_test_images import generate_dense_document
    return pil_to_png_bytes(generate_dense_document())


def _percentile(data, p):
    if not data:
        return 0
    data_sorted = sorted(data)
    k = (len(data_sorted) - 1) * (p / 100)
    f = int(k)
    c = f + 1
    if c >= len(data_sorted):
        return data_sorted[f]
    return data_sorted[f] + (k - f) * (data_sorted[c] - data_sorted[f])


def measure_throughput(server_url, image_bytes, concurrency, total):
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
        statuses = list(pool.map(lambda _: fire(), range(total)))
    elapsed = time.perf_counter() - t0
    success = sum(1 for s in statuses if s == 200)
    return success / elapsed, success, total, elapsed


def measure_latencies(server_url, image_bytes, concurrency, total):
    latencies = []

    def fire():
        t0 = time.perf_counter()
        r = requests.post(
            f"{server_url}/ocr/raw",
            data=image_bytes,
            headers={"Content-Type": "image/png"},
            timeout=60,
        )
        return (time.perf_counter() - t0) * 1000, r.status_code

    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        results = list(pool.map(lambda _: fire(), range(total)))

    for ms, status in results:
        if status == 200:
            latencies.append(ms)
    return sorted(latencies)


def main():
    parser = argparse.ArgumentParser(description="Generate benchmark report")
    parser.add_argument("--server-url", default="http://localhost:8000")
    parser.add_argument("--output", default="benchmark_report.md")
    parser.add_argument("--iterations", type=int, default=100,
                        help="Requests per concurrency level")
    args = parser.parse_args()

    url = args.server_url

    # Check server health
    try:
        r = requests.get(f"{url}/health", timeout=5)
        if r.status_code != 200:
            print(f"Server health check failed: {r.status_code}")
            sys.exit(1)
    except requests.ConnectionError:
        print(f"Server not reachable at {url}")
        sys.exit(1)

    print(f"Generating benchmark report against {url}")
    print(f"Iterations per level: {args.iterations}")
    print()

    image_bytes = _get_doc_image()
    image_size_kb = len(image_bytes) / 1024

    lines = []
    lines.append(f"# Turbo OCR Benchmark Report")
    lines.append(f"")
    lines.append(f"**Date:** {datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append(f"**Server:** {url}")
    lines.append(f"**Test image:** Dense document ({image_size_kb:.0f} KB PNG)")
    lines.append(f"**Iterations per level:** {args.iterations}")
    lines.append(f"")

    # --- Warmup ---
    print("Warming up...")
    for _ in range(8):
        requests.post(
            f"{url}/ocr/raw",
            data=image_bytes,
            headers={"Content-Type": "image/png"},
            timeout=30,
        )

    # --- Throughput ---
    lines.append("## Throughput")
    lines.append("")
    lines.append("| Concurrency | Throughput (img/s) | Success | Time (s) |")
    lines.append("|:-----------:|:------------------:|:-------:|:--------:|")

    print("Measuring throughput...")
    for c in [1, 2, 4, 8, 16, 32]:
        tp, success, total, elapsed = measure_throughput(url, image_bytes, c, args.iterations)
        lines.append(f"| {c} | {tp:.1f} | {success}/{total} | {elapsed:.2f} |")
        print(f"  c={c}: {tp:.1f} img/s")

    lines.append("")

    # --- Latency ---
    lines.append("## Latency Percentiles")
    lines.append("")
    lines.append("| Concurrency | p50 (ms) | p95 (ms) | p99 (ms) | avg (ms) | min (ms) | max (ms) |")
    lines.append("|:-----------:|:--------:|:--------:|:--------:|:--------:|:--------:|:--------:|")

    print("Measuring latency...")
    for c in [1, 4, 8, 16]:
        lats = measure_latencies(url, image_bytes, c, args.iterations)
        if lats:
            p50 = _percentile(lats, 50)
            p95 = _percentile(lats, 95)
            p99 = _percentile(lats, 99)
            avg = statistics.mean(lats)
            lines.append(f"| {c} | {p50:.1f} | {p95:.1f} | {p99:.1f} | "
                         f"{avg:.1f} | {min(lats):.1f} | {max(lats):.1f} |")
            print(f"  c={c}: p50={p50:.1f}ms p95={p95:.1f}ms p99={p99:.1f}ms")

    lines.append("")

    # --- Batch throughput ---
    lines.append("## Batch Processing")
    lines.append("")
    lines.append("| Batch Size | Time (s) | Throughput (img/s) |")
    lines.append("|:----------:|:--------:|:------------------:|")

    print("Measuring batch throughput...")
    for n in [4, 8, 16]:
        images = []
        for i in range(n):
            img = make_text_image(f"BATCH{i}", width=400, height=100, font_size=36)
            images.append(base64.b64encode(pil_to_png_bytes(img)).decode("ascii"))

        t0 = time.perf_counter()
        r = requests.post(f"{url}/ocr/batch", json={"images": images}, timeout=60)
        elapsed = time.perf_counter() - t0
        if r.status_code == 200:
            tp = n / elapsed
            lines.append(f"| {n} | {elapsed:.2f} | {tp:.1f} |")
            print(f"  batch={n}: {tp:.1f} img/s")

    lines.append("")
    lines.append("---")
    lines.append("*Generated by bench_report.py*")

    report = "\n".join(lines) + "\n"

    output_path = Path(args.output)
    output_path.write_text(report)
    print(f"\nReport written to {output_path}")


if __name__ == "__main__":
    main()
