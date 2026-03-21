#!/usr/bin/env python3
"""Comprehensive benchmark suite for turbo_ocr server."""

import time
import requests
import statistics
import os
import sys
import json
import base64
import concurrent.futures
import threading
from pathlib import Path
from datetime import datetime

SERVER_RAW = "http://localhost:8000/ocr/raw"
SERVER_BATCH = "http://localhost:8000/ocr/batch"
SERVER_PDF = "http://localhost:8000/ocr/pdf"

BASE = Path("/home/nataell/code/epAiland/paddle-highspeed-cpp/tests/test_data")
PNG_DIR = BASE / "png"
JPEG_DIR = BASE / "jpeg"
PDF_DIR = BASE / "pdf"

# ── helpers ──────────────────────────────────────────────────────────────────

def content_type_for(path):
    s = str(path).lower()
    if s.endswith(".png"):
        return "image/png"
    elif s.endswith((".jpg", ".jpeg")):
        return "image/jpeg"
    elif s.endswith(".pdf"):
        return "application/pdf"
    return "application/octet-stream"


def send_raw(data, ct):
    return requests.post(SERVER_RAW, data=data, headers={"Content-Type": ct})


def fmt(v, unit="ms"):
    return f"{v:.2f} {unit}"


def percentile(sorted_list, p):
    idx = int(len(sorted_list) * p / 100)
    idx = min(idx, len(sorted_list) - 1)
    return sorted_list[idx]


# ── 1. Single-image latency ─────────────────────────────────────────────────

def bench_single(filepath, n_warmup=5, n_measure=50):
    ct = content_type_for(filepath)
    data = filepath.read_bytes()
    for _ in range(n_warmup):
        send_raw(data, ct)
    times = []
    for _ in range(n_measure):
        t0 = time.perf_counter()
        r = send_raw(data, ct)
        t1 = time.perf_counter()
        assert r.status_code == 200, f"status {r.status_code}"
        times.append((t1 - t0) * 1000)
    times.sort()
    return {
        "avg": statistics.mean(times),
        "p50": percentile(times, 50),
        "p95": percentile(times, 95),
        "p99": percentile(times, 99),
        "ips": 1000.0 / statistics.mean(times),
        "regions": len(r.json().get("results", [])),
    }


def run_single_latency():
    print("\n=== 1. Single-image latency ===")
    results = {}
    for d, label in [(PNG_DIR, "PNG"), (JPEG_DIR, "JPEG")]:
        for f in sorted(d.iterdir()):
            if f.is_dir():
                continue
            name = f"{label}/{f.name}"
            print(f"  {name} ...", end=" ", flush=True)
            r = bench_single(f)
            results[name] = r
            print(f"avg={fmt(r['avg'])}  p50={fmt(r['p50'])}  p95={fmt(r['p95'])}  "
                  f"p99={fmt(r['p99'])}  ips={r['ips']:.1f}  regions={r['regions']}")
    return results


# ── 2. Concurrent throughput ─────────────────────────────────────────────────

def bench_concurrent(filepath, concurrency, n_total=200):
    ct = content_type_for(filepath)
    data = filepath.read_bytes()
    # warmup
    for _ in range(5):
        send_raw(data, ct)

    def send(_):
        r = requests.post(SERVER_RAW, data=data, headers={"Content-Type": ct})
        return r.status_code == 200

    t0 = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as pool:
        results = list(pool.map(send, range(n_total)))
    t1 = time.perf_counter()
    elapsed = t1 - t0
    success = sum(results)
    return {
        "concurrency": concurrency,
        "total": n_total,
        "success": success,
        "elapsed_s": elapsed,
        "ips": n_total / elapsed,
    }


def run_concurrent_throughput():
    print("\n=== 2. Concurrent throughput ===")
    results = {}
    for fname, label in [("receipt.png", "receipt (sparse)"), ("mixed_fonts.png", "mixed_fonts (dense)")]:
        fp = PNG_DIR / fname
        results[label] = []
        print(f"\n  {label}:")
        for c in [1, 2, 4, 8, 16, 32]:
            print(f"    concurrency={c} ...", end=" ", flush=True)
            r = bench_concurrent(fp, c)
            results[label].append(r)
            print(f"ips={r['ips']:.1f}  elapsed={r['elapsed_s']:.2f}s  ok={r['success']}/{r['total']}")
    return results


# ── 3. JPEG vs PNG format comparison ────────────────────────────────────────

def run_format_comparison():
    print("\n=== 3. JPEG vs PNG format comparison ===")
    # Find similar images across formats by rough name matching
    png_files = {f.stem: f for f in sorted(PNG_DIR.iterdir()) if f.is_file()}
    jpeg_files = {f.stem: f for f in sorted(JPEG_DIR.iterdir()) if f.is_file()}

    # Just benchmark all PNGs and all JPEGs and compare aggregates
    results = {"png": {}, "jpeg": {}}
    for f in sorted(PNG_DIR.iterdir()):
        if f.is_dir():
            continue
        print(f"  PNG/{f.name} ...", end=" ", flush=True)
        r = bench_single(f, n_warmup=3, n_measure=30)
        results["png"][f.name] = r
        print(f"avg={fmt(r['avg'])}  size={f.stat().st_size/1024:.1f}KB")

    for f in sorted(JPEG_DIR.iterdir()):
        if f.is_dir():
            continue
        print(f"  JPEG/{f.name} ...", end=" ", flush=True)
        r = bench_single(f, n_warmup=3, n_measure=30)
        results["jpeg"][f.name] = r
        print(f"avg={fmt(r['avg'])}  size={f.stat().st_size/1024:.1f}KB")

    return results


# ── 4. PDF throughput ────────────────────────────────────────────────────────

def run_pdf_benchmark():
    print("\n=== 4. PDF throughput ===")
    results = {}
    for f in sorted(PDF_DIR.iterdir()):
        if f.is_dir():
            continue
        data = f.read_bytes()
        print(f"  {f.name} ...", end=" ", flush=True)
        # warmup
        try:
            r = requests.post(SERVER_PDF, data=data, headers={"Content-Type": "application/pdf"})
            if r.status_code != 200:
                print(f"SKIP (status {r.status_code}: {r.text[:100]})")
                results[f.name] = {"error": f"status {r.status_code}"}
                continue
        except Exception as e:
            print(f"SKIP ({e})")
            results[f.name] = {"error": str(e)}
            continue

        # Measure 10 runs
        times = []
        pages = 0
        for _ in range(10):
            t0 = time.perf_counter()
            r = requests.post(SERVER_PDF, data=data, headers={"Content-Type": "application/pdf"})
            t1 = time.perf_counter()
            times.append((t1 - t0) * 1000)
            if r.status_code == 200:
                resp = r.json()
                pages = len(resp.get("pages", []))
        times.sort()
        avg = statistics.mean(times)
        results[f.name] = {
            "avg_ms": avg,
            "p50": percentile(times, 50),
            "p95": percentile(times, 95),
            "pages": pages,
            "pages_per_sec": pages * 1000.0 / avg if avg > 0 else 0,
            "size_kb": f.stat().st_size / 1024,
        }
        print(f"avg={fmt(avg)}  pages={pages}  pages/s={results[f.name]['pages_per_sec']:.1f}  "
              f"size={results[f.name]['size_kb']:.1f}KB")
    return results


# ── 5. Batch endpoint ────────────────────────────────────────────────────────

def run_batch_benchmark():
    print("\n=== 5. Batch endpoint ===")
    # Load some images for batching
    images = []
    for f in sorted(PNG_DIR.iterdir()):
        if f.is_dir():
            continue
        images.append(base64.b64encode(f.read_bytes()).decode())
        if len(images) >= 20:
            break

    results = {}
    for batch_size in [5, 10, 20]:
        batch_images = images[:batch_size]
        payload = json.dumps({"images": batch_images})

        # warmup
        requests.post(SERVER_BATCH, data=payload, headers={"Content-Type": "application/json"})

        # measure batch
        times_batch = []
        for _ in range(10):
            t0 = time.perf_counter()
            r = requests.post(SERVER_BATCH, data=payload, headers={"Content-Type": "application/json"})
            t1 = time.perf_counter()
            assert r.status_code == 200, f"batch status {r.status_code}: {r.text[:200]}"
            times_batch.append((t1 - t0) * 1000)
        times_batch.sort()

        # measure individual (sequential)
        raw_data_list = [(PNG_DIR / f.name, f.read_bytes())
                         for f in sorted(PNG_DIR.iterdir()) if f.is_file()][:batch_size]
        times_individual = []
        for _ in range(10):
            t0 = time.perf_counter()
            for fpath, _ in raw_data_list:
                data = fpath.read_bytes()
                r2 = send_raw(data, content_type_for(fpath))
                assert r2.status_code == 200
            t1 = time.perf_counter()
            times_individual.append((t1 - t0) * 1000)
        times_individual.sort()

        batch_avg = statistics.mean(times_batch)
        indiv_avg = statistics.mean(times_individual)
        speedup = indiv_avg / batch_avg if batch_avg > 0 else 0

        results[batch_size] = {
            "batch_avg_ms": batch_avg,
            "batch_p50": percentile(times_batch, 50),
            "individual_avg_ms": indiv_avg,
            "individual_p50": percentile(times_individual, 50),
            "speedup": speedup,
        }
        print(f"  batch={batch_size}:  batch_avg={fmt(batch_avg)}  indiv_avg={fmt(indiv_avg)}  "
              f"speedup={speedup:.2f}x")
    return results


# ── 6. Parallel request isolation ────────────────────────────────────────────

def run_isolation_test():
    print("\n=== 6. Parallel request isolation ===")
    # Load all PNG images and their expected region counts
    test_cases = []
    for f in sorted(PNG_DIR.iterdir()):
        if f.is_dir():
            continue
        expected_file = PNG_DIR / "expected" / f"{f.stem}.json"
        if expected_file.exists():
            expected = json.loads(expected_file.read_text())
            expected_count = len(expected.get("results", []))
        else:
            expected_count = None
        test_cases.append((f, f.read_bytes(), expected_count))

    results = []
    # Send all 10 simultaneously
    def send_and_check(args):
        fpath, data, expected_count = args
        ct = content_type_for(fpath)
        r = requests.post(SERVER_RAW, data=data, headers={"Content-Type": ct})
        actual_count = len(r.json().get("results", []))
        return {
            "file": fpath.name,
            "status": r.status_code,
            "actual_regions": actual_count,
            "expected_regions": expected_count,
            "match": expected_count is None or actual_count == expected_count,
        }

    with concurrent.futures.ThreadPoolExecutor(max_workers=len(test_cases)) as pool:
        results = list(pool.map(send_and_check, test_cases))

    all_ok = all(r["match"] for r in results)
    for r in results:
        status = "OK" if r["match"] else "MISMATCH"
        exp = r["expected_regions"] if r["expected_regions"] is not None else "?"
        print(f"  {r['file']}: actual={r['actual_regions']} expected={exp} [{status}]")
    print(f"  Isolation test: {'PASS' if all_ok else 'FAIL'}")
    return {"results": results, "pass": all_ok}


# ── 7. Mixed format stress test ─────────────────────────────────────────────

def run_stress_test(duration_s=30):
    print(f"\n=== 7. Mixed format stress test ({duration_s}s) ===")
    # Prepare payloads
    payloads = []
    for f in sorted(PNG_DIR.iterdir()):
        if f.is_dir():
            continue
        payloads.append((f.read_bytes(), content_type_for(f), "png", f.name))
    for f in sorted(JPEG_DIR.iterdir()):
        if f.is_dir():
            continue
        payloads.append((f.read_bytes(), content_type_for(f), "jpeg", f.name))
    # Add PDFs
    for f in sorted(PDF_DIR.iterdir()):
        if f.is_dir():
            continue
        payloads.append((f.read_bytes(), content_type_for(f), "pdf", f.name))

    counters = {"total": 0, "success": 0, "errors": 0, "png": 0, "jpeg": 0, "pdf": 0}
    lock = threading.Lock()
    stop = threading.Event()
    error_details = []

    def worker():
        import random
        while not stop.is_set():
            data, ct, fmt_type, name = random.choice(payloads)
            try:
                if fmt_type == "pdf":
                    r = requests.post(SERVER_PDF, data=data, headers={"Content-Type": ct}, timeout=30)
                else:
                    r = requests.post(SERVER_RAW, data=data, headers={"Content-Type": ct}, timeout=30)
                with lock:
                    counters["total"] += 1
                    counters[fmt_type] += 1
                    if r.status_code == 200:
                        counters["success"] += 1
                    else:
                        counters["errors"] += 1
                        error_details.append(f"{name}: {r.status_code}")
            except Exception as e:
                with lock:
                    counters["total"] += 1
                    counters["errors"] += 1
                    error_details.append(f"{name}: {e}")

    threads = []
    for _ in range(20):
        t = threading.Thread(target=worker, daemon=True)
        t.start()
        threads.append(t)

    t0 = time.perf_counter()
    time.sleep(duration_s)
    stop.set()
    for t in threads:
        t.join(timeout=5)
    elapsed = time.perf_counter() - t0

    error_rate = counters["errors"] / counters["total"] * 100 if counters["total"] > 0 else 0
    rps = counters["total"] / elapsed

    print(f"  Total requests: {counters['total']}")
    print(f"  Success: {counters['success']}")
    print(f"  Errors: {counters['errors']} ({error_rate:.1f}%)")
    print(f"  PNG: {counters['png']}  JPEG: {counters['jpeg']}  PDF: {counters['pdf']}")
    print(f"  Requests/sec: {rps:.1f}")
    if error_details[:5]:
        print(f"  Sample errors: {error_details[:5]}")

    return {
        "duration_s": elapsed,
        "total": counters["total"],
        "success": counters["success"],
        "errors": counters["errors"],
        "error_rate_pct": error_rate,
        "rps": rps,
        "by_format": {k: counters[k] for k in ["png", "jpeg", "pdf"]},
        "sample_errors": error_details[:10],
    }


# ── Report generation ────────────────────────────────────────────────────────

def generate_report(single, concurrent_res, format_cmp, pdf_res, batch_res, isolation, stress):
    lines = []
    a = lines.append

    a(f"# Benchmark Results")
    a(f"")
    a(f"**Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    a(f"**Server:** http://localhost:8000")
    a(f"")

    # Baseline comparison
    a("## Baseline Comparison")
    a("")
    a("| Metric | Baseline (git log) | Measured |")
    a("|--------|-------------------|----------|")

    # Find receipt for sparse comparison
    receipt_key = None
    for k in single:
        if "receipt" in k.lower():
            receipt_key = k
            break

    # Find a dense A4-like image
    dense_key = None
    for k in single:
        if "dense_text" in k.lower() or "mixed_fonts" in k.lower() or "business_letter" in k.lower():
            dense_key = k
            break

    # Get concurrent results for receipt at c=8
    receipt_ips_c8 = "N/A"
    if "receipt (sparse)" in concurrent_res:
        for r in concurrent_res["receipt (sparse)"]:
            if r["concurrency"] == 8:
                receipt_ips_c8 = f"{r['ips']:.0f} img/s"

    # Get concurrent results for dense at c=16
    dense_ips_c16 = "N/A"
    if "mixed_fonts (dense)" in concurrent_res:
        for r in concurrent_res["mixed_fonts (dense)"]:
            if r["concurrency"] == 16:
                dense_ips_c16 = f"{r['ips']:.0f} img/s"

    a(f"| A4 dense @ c=16 | ~246 img/s | {dense_ips_c16} |")
    a(f"| Sparse receipt @ c=8 | ~1200+ img/s | {receipt_ips_c8} |")

    if receipt_key:
        r = single[receipt_key]
        a(f"| Latency p50 (receipt) | 9.5 ms | {r['p50']:.1f} ms |")
        a(f"| Latency p95 (receipt) | 13.3 ms | {r['p95']:.1f} ms |")
        a(f"| Latency p99 (receipt) | 18.2 ms | {r['p99']:.1f} ms |")
    a("")

    # 1. Single-image latency
    a("## 1. Single-Image Latency")
    a("")
    a("| Image | Avg (ms) | p50 (ms) | p95 (ms) | p99 (ms) | img/s | Regions |")
    a("|-------|----------|----------|----------|----------|-------|---------|")
    for name, r in sorted(single.items()):
        a(f"| {name} | {r['avg']:.2f} | {r['p50']:.2f} | {r['p95']:.2f} | {r['p99']:.2f} | {r['ips']:.1f} | {r['regions']} |")
    a("")

    # 2. Concurrent throughput
    a("## 2. Concurrent Throughput")
    a("")
    for label, runs in concurrent_res.items():
        a(f"### {label}")
        a("")
        a("| Concurrency | img/s | Elapsed (s) | Success/Total |")
        a("|-------------|-------|-------------|---------------|")
        for r in runs:
            a(f"| {r['concurrency']} | {r['ips']:.1f} | {r['elapsed_s']:.2f} | {r['success']}/{r['total']} |")
        a("")

    # 3. Format comparison
    a("## 3. JPEG vs PNG Format Comparison")
    a("")
    a("### PNG")
    a("")
    a("| Image | Avg (ms) | img/s | Regions |")
    a("|-------|----------|-------|---------|")
    png_avgs = []
    for name, r in sorted(format_cmp["png"].items()):
        a(f"| {name} | {r['avg']:.2f} | {r['ips']:.1f} | {r['regions']} |")
        png_avgs.append(r['avg'])
    a("")
    a("### JPEG")
    a("")
    a("| Image | Avg (ms) | img/s | Regions |")
    a("|-------|----------|-------|---------|")
    jpeg_avgs = []
    for name, r in sorted(format_cmp["jpeg"].items()):
        a(f"| {name} | {r['avg']:.2f} | {r['ips']:.1f} | {r['regions']} |")
        jpeg_avgs.append(r['avg'])
    a("")
    if png_avgs and jpeg_avgs:
        a(f"**PNG average:** {statistics.mean(png_avgs):.2f} ms | "
          f"**JPEG average:** {statistics.mean(jpeg_avgs):.2f} ms")
        a("")

    # 4. PDF
    a("## 4. PDF Throughput")
    a("")
    a("| PDF | Pages | Avg (ms) | p50 (ms) | p95 (ms) | Pages/s | Size (KB) |")
    a("|-----|-------|----------|----------|----------|---------|-----------|")
    for name, r in sorted(pdf_res.items()):
        if "error" in r:
            a(f"| {name} | - | ERROR | - | - | - | - |")
        else:
            a(f"| {name} | {r['pages']} | {r['avg_ms']:.1f} | {r['p50']:.1f} | {r['p95']:.1f} | {r['pages_per_sec']:.1f} | {r['size_kb']:.1f} |")
    a("")

    # 5. Batch
    a("## 5. Batch Endpoint")
    a("")
    a("| Batch Size | Batch Avg (ms) | Individual Avg (ms) | Speedup |")
    a("|------------|----------------|---------------------|---------|")
    for bs, r in sorted(batch_res.items()):
        a(f"| {bs} | {r['batch_avg_ms']:.1f} | {r['individual_avg_ms']:.1f} | {r['speedup']:.2f}x |")
    a("")

    # 6. Isolation
    a("## 6. Parallel Request Isolation")
    a("")
    a(f"**Result:** {'PASS' if isolation['pass'] else 'FAIL'}")
    a("")
    a("| Image | Actual Regions | Expected Regions | Match |")
    a("|-------|----------------|------------------|-------|")
    for r in isolation["results"]:
        exp = r["expected_regions"] if r["expected_regions"] is not None else "?"
        a(f"| {r['file']} | {r['actual_regions']} | {exp} | {'Yes' if r['match'] else 'No'} |")
    a("")

    # 7. Stress
    a("## 7. Mixed Format Stress Test")
    a("")
    a(f"- **Duration:** {stress['duration_s']:.1f}s")
    a(f"- **Total requests:** {stress['total']}")
    a(f"- **Success:** {stress['success']}")
    a(f"- **Errors:** {stress['errors']} ({stress['error_rate_pct']:.1f}%)")
    a(f"- **Requests/sec:** {stress['rps']:.1f}")
    a(f"- **By format:** PNG={stress['by_format']['png']}, JPEG={stress['by_format']['jpeg']}, PDF={stress['by_format']['pdf']}")
    if stress["sample_errors"]:
        a(f"- **Sample errors:** {', '.join(stress['sample_errors'][:5])}")
    a("")

    return "\n".join(lines)


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    # Quick health check
    try:
        r = requests.get("http://localhost:8000/health", timeout=5)
        assert r.status_code == 200
        print("Server health: OK")
    except Exception as e:
        print(f"Server not reachable: {e}")
        sys.exit(1)

    single = run_single_latency()
    concurrent_res = run_concurrent_throughput()
    format_cmp = run_format_comparison()
    pdf_res = run_pdf_benchmark()
    batch_res = run_batch_benchmark()
    isolation = run_isolation_test()
    stress = run_stress_test(duration_s=30)

    report = generate_report(single, concurrent_res, format_cmp, pdf_res, batch_res, isolation, stress)

    out_path = Path("/home/nataell/code/epAiland/paddle-highspeed-cpp/tests/benchmark/BENCHMARK_RESULTS.md")
    out_path.write_text(report)
    print(f"\nReport written to {out_path}")

    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY vs BASELINE")
    print("=" * 60)
    print(report.split("## 1.")[0])


if __name__ == "__main__":
    main()
