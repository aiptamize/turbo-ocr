# Hey Benchmark Results

Measured with `hey` (Go HTTP client) -- zero Python overhead.
Date: 2026-03-30
Endpoint: `POST /ocr/raw` (raw bytes, no base64/JSON)

## System

- GPU: RTX 5090
- Server: Turbo OCR C++ (Crow HTTP)
- Pipeline pool: 5

---

## Receipt (small, sparse ~10 detections)

| Concurrency | Requests/sec | Notes |
|------------|-------------|-------|
| 1          | 101         | Single-threaded baseline |
| 4          | 470         | |
| 8          | 583         | |
| 16         | 564         | |
| 32         | 600         | Saturated |

**Peak: ~600 img/s at c=32**

## A4 Document (business_letter)

| Concurrency | Requests/sec |
|------------|-------------|
| 16         | 227         |

## Dense Document (mixed_fonts, 413 regions)

| Concurrency | Requests/sec |
|------------|-------------|
| 1          | 17.6        |
| 4          | 18.8        |
| 8          | 20.8        |
| 16         | 21.9        |

Heavy recognition workload -- GPU-bound on rec inference.

## JPEG (restaurant menu)

| Concurrency | Requests/sec |
|------------|-------------|
| 1          | 19.4        |
| 4          | 112         |
| 8          | 161         |
| 16         | 158         |

## Format Comparison (c=8, n=500)

| Format | Requests/sec |
|--------|-------------|
| PNG (receipt) | 238 |
| JPEG (restaurant menu) | 88 |

## All Test Images (c=8, n=100 each)

### PNG
| Image | Requests/sec |
|-------|-------------|
| business_letter | 97 |
| dense_text | 65 |
| form_fields | 81 |
| mixed_fonts | 19 |
| multi_language | 115 |
| receipt | 496 |
| rotated_text | 494 |
| small_text | 430 |
| street_sign | 133 |
| table | 243 |

### JPEG
| Image | Requests/sec |
|-------|-------------|
| 01_restaurant_menu | 304 |
| 02_road_sign | 132 |
| 03_book_page | 359 |
| 04_product_label | 324 |
| 05_handwritten_note | 450 |
| 06_whiteboard | 136 |
| 07_low_quality | 140 |
| 08_document_scan | 264 |
| 09_complex_background | 141 |
| 10_newspaper | 288 |

---

## Latency (sequential, c=1)

### Receipt (sparse)
| Percentile | Latency |
|-----------|---------|
| p50 | 6.6ms |
| p75 | 8.4ms |
| p90 | 11.4ms |
| p95 | 43.8ms |
| p99 | 196.3ms |

### A4 Document (business_letter)
| Percentile | Latency |
|-----------|---------|
| p50 | 11.9ms |
| p75 | 12.2ms |
| p90 | 14.5ms |
| p95 | 15.5ms |
| p99 | 20.0ms |

---

## Comparison vs Baseline

Baseline (RTX 5090, pool=5, previously measured):
- A4 documents: 246 img/s (c=16)
- Sparse receipts: 1200+ img/s (c=8)
- p50=9.5ms, p95=13.3ms, p99=18.2ms

### Current results:
- A4 documents: **227 img/s** (c=16) -- **8% regression** vs 246 baseline
- Sparse receipts: **583 img/s** (c=8) -- **~52% regression** vs 1200+ baseline
- Receipt latency: p50=6.6ms (better), p95=43.8ms (worse), p99=196.3ms (worse)
- A4 latency: p50=11.9ms, p95=15.5ms, p99=20.0ms (close to baseline)

### Analysis

1. **A4 throughput** is close to baseline (227 vs 246, ~8% lower). This is within
   normal variance depending on GPU thermals and background load, but worth investigating.

2. **Receipt throughput** shows significant regression (583 vs 1200+ img/s). The receipt
   image has very few text regions (~10 detections), so the bottleneck shifts from
   GPU inference to request handling overhead. Possible causes:
   - Server concurrency handling changes
   - Pipeline pool contention
   - Image decode path changes
   - HTTP framework overhead

3. **Tail latency** (p95/p99) for receipts is notably worse, with occasional spikes
   to 200ms+ suggesting contention or GC-like pauses.

4. **A4 latency** is reasonable: p50=11.9ms is close to baseline p50=9.5ms (the
   baseline was measured on receipts, not A4 docs, so not directly comparable).

### Recommendation

The receipt throughput regression is the main concern. Profile the server under
high concurrency with small images to identify the bottleneck (likely not GPU-bound
for sparse images). Check pipeline pool acquire/release overhead and HTTP parsing.

---

## Notes

- Some requests at high concurrency (c=16, c=32) show as dropped (496/500 or 480/500).
  These are `hey` client-side connection timeouts, not server errors.
- All 200 responses returned valid OCR results.
- gRPC benchmark tools (`grpc_bench`, `grpc_burst`) are not currently compiled in the build.
