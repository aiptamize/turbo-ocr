# Benchmark Results

**Date:** 2026-03-31 02:42:28
**Server:** http://localhost:8000

## Baseline Comparison

| Metric | Baseline (git log) | Measured |
|--------|-------------------|----------|
| A4 dense @ c=16 | ~246 img/s | 23 img/s |
| Sparse receipt @ c=8 | ~1200+ img/s | 407 img/s |
| Latency p50 (receipt) | 9.5 ms | 11.9 ms |
| Latency p95 (receipt) | 13.3 ms | 18.1 ms |
| Latency p99 (receipt) | 18.2 ms | 22.1 ms |

## 1. Single-Image Latency

| Image | Avg (ms) | p50 (ms) | p95 (ms) | p99 (ms) | img/s | Regions |
|-------|----------|----------|----------|----------|-------|---------|
| JPEG/01_restaurant_menu.jpg | 14.44 | 13.87 | 17.37 | 18.15 | 69.2 | 22 |
| JPEG/02_road_sign.jpg | 10.21 | 8.74 | 16.89 | 25.00 | 97.9 | 5 |
| JPEG/03_book_page.jpg | 14.02 | 13.65 | 20.62 | 26.68 | 71.3 | 12 |
| JPEG/04_product_label.jpg | 14.24 | 14.19 | 19.72 | 23.00 | 70.2 | 53 |
| JPEG/05_handwritten_note.jpg | 13.18 | 13.01 | 18.33 | 18.81 | 75.9 | 14 |
| JPEG/06_whiteboard.jpg | 8.86 | 8.05 | 13.58 | 15.65 | 112.8 | 10 |
| JPEG/07_low_quality.jpg | 10.07 | 9.73 | 17.01 | 22.52 | 99.3 | 8 |
| JPEG/08_document_scan.jpg | 12.61 | 11.99 | 17.19 | 21.15 | 79.3 | 33 |
| JPEG/09_complex_background.jpg | 10.03 | 9.17 | 15.96 | 20.15 | 99.7 | 3 |
| JPEG/10_newspaper.jpg | 12.17 | 11.26 | 15.48 | 18.35 | 82.1 | 31 |
| PNG/business_letter.png | 15.22 | 14.68 | 20.79 | 25.39 | 65.7 | 141 |
| PNG/dense_text.png | 18.07 | 17.27 | 23.38 | 23.52 | 55.4 | 26 |
| PNG/form_fields.png | 16.74 | 15.89 | 22.03 | 23.80 | 59.7 | 99 |
| PNG/mixed_fonts.png | 42.97 | 42.21 | 47.53 | 48.55 | 23.3 | 413 |
| PNG/multi_language.png | 9.08 | 7.85 | 15.76 | 18.36 | 110.1 | 8 |
| PNG/receipt.png | 11.83 | 11.88 | 18.11 | 22.14 | 84.5 | 35 |
| PNG/rotated_text.png | 10.49 | 9.72 | 16.98 | 19.32 | 95.3 | 41 |
| PNG/small_text.png | 10.14 | 9.08 | 16.41 | 16.93 | 98.6 | 42 |
| PNG/street_sign.png | 10.13 | 9.67 | 15.36 | 18.67 | 98.7 | 1 |
| PNG/table.png | 16.61 | 15.99 | 20.53 | 21.45 | 60.2 | 152 |

## 2. Concurrent Throughput

### receipt (sparse)

| Concurrency | img/s | Elapsed (s) | Success/Total |
|-------------|-------|-------------|---------------|
| 1 | 117.8 | 1.70 | 200/200 |
| 2 | 244.7 | 0.82 | 200/200 |
| 4 | 395.9 | 0.51 | 200/200 |
| 8 | 406.7 | 0.49 | 200/200 |
| 16 | 413.1 | 0.48 | 200/200 |
| 32 | 550.7 | 0.36 | 200/200 |

### mixed_fonts (dense)

| Concurrency | img/s | Elapsed (s) | Success/Total |
|-------------|-------|-------------|---------------|
| 1 | 23.3 | 8.59 | 200/200 |
| 2 | 33.4 | 5.99 | 200/200 |
| 4 | 24.6 | 8.13 | 200/200 |
| 8 | 23.3 | 8.59 | 200/200 |
| 16 | 23.3 | 8.58 | 200/200 |
| 32 | 23.2 | 8.63 | 200/200 |

## 3. JPEG vs PNG Format Comparison

### PNG

| Image | Avg (ms) | img/s | Regions |
|-------|----------|-------|---------|
| business_letter.png | 14.05 | 71.2 | 141 |
| dense_text.png | 17.60 | 56.8 | 26 |
| form_fields.png | 16.34 | 61.2 | 99 |
| mixed_fonts.png | 42.64 | 23.5 | 413 |
| multi_language.png | 8.05 | 124.2 | 8 |
| receipt.png | 8.10 | 123.5 | 35 |
| rotated_text.png | 10.89 | 91.8 | 41 |
| small_text.png | 12.18 | 82.1 | 42 |
| street_sign.png | 12.83 | 78.0 | 1 |
| table.png | 16.43 | 60.9 | 152 |

### JPEG

| Image | Avg (ms) | img/s | Regions |
|-------|----------|-------|---------|
| 01_restaurant_menu.jpg | 14.41 | 69.4 | 22 |
| 02_road_sign.jpg | 11.94 | 83.8 | 5 |
| 03_book_page.jpg | 11.03 | 90.7 | 12 |
| 04_product_label.jpg | 11.83 | 84.5 | 53 |
| 05_handwritten_note.jpg | 13.30 | 75.2 | 14 |
| 06_whiteboard.jpg | 11.87 | 84.3 | 10 |
| 07_low_quality.jpg | 8.80 | 113.6 | 8 |
| 08_document_scan.jpg | 11.42 | 87.5 | 33 |
| 09_complex_background.jpg | 8.40 | 119.0 | 3 |
| 10_newspaper.jpg | 14.77 | 67.7 | 31 |

**PNG average:** 15.91 ms | **JPEG average:** 11.78 ms

## 4. PDF Throughput

| PDF | Pages | Avg (ms) | p50 (ms) | p95 (ms) | Pages/s | Size (KB) |
|-----|-------|----------|----------|----------|---------|-----------|
| academic_paper.pdf | 15 | 297.7 | 298.2 | 303.5 | 50.4 | 620.2 |
| formulas.pdf | 20 | 214.6 | 213.9 | 236.3 | 93.2 | 229.1 |
| headers_footers.pdf | 2 | 33.1 | 34.5 | 35.8 | 60.4 | 119.7 |
| mixed_text_images.pdf | 12 | 264.5 | 267.3 | 272.7 | 45.4 | 800.2 |
| multi_column.pdf | 27 | 407.9 | 408.0 | 423.8 | 66.2 | 709.5 |
| scanned_document.pdf | 8 | 74.3 | 73.7 | 84.1 | 107.7 | 18.8 |
| simple_letter.pdf | 2 | 46.3 | 45.1 | 53.3 | 43.2 | 215.1 |
| single_page_form.pdf | 6 | 163.5 | 162.7 | 167.5 | 36.7 | 137.5 |
| small_font.pdf | 2 | 33.5 | 33.2 | 39.5 | 59.7 | 97.6 |
| tables_document.pdf | 2 | 30.9 | 30.8 | 33.9 | 64.7 | 78.4 |

## 5. Batch Endpoint

| Batch Size | Batch Avg (ms) | Individual Avg (ms) | Speedup |
|------------|----------------|---------------------|---------|
| 5 | 73.3 | 99.4 | 1.36x |
| 10 | 105.1 | 156.5 | 1.49x |
| 20 | 105.5 | 161.5 | 1.53x |

## 6. Parallel Request Isolation

**Result:** PASS

| Image | Actual Regions | Expected Regions | Match |
|-------|----------------|------------------|-------|
| business_letter.png | 141 | 141 | Yes |
| dense_text.png | 26 | 26 | Yes |
| form_fields.png | 99 | 99 | Yes |
| mixed_fonts.png | 413 | 413 | Yes |
| multi_language.png | 8 | 8 | Yes |
| receipt.png | 35 | 35 | Yes |
| rotated_text.png | 41 | 41 | Yes |
| small_text.png | 42 | 42 | Yes |
| street_sign.png | 1 | 1 | Yes |
| table.png | 152 | 152 | Yes |

## 7. Mixed Format Stress Test

- **Duration:** 30.9s
- **Total requests:** 937
- **Success:** 937
- **Errors:** 0 (0.0%)
- **Requests/sec:** 30.3
- **By format:** PNG=302, JPEG=314, PDF=321
