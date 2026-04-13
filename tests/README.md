# Turbo OCR Test Suite

Comprehensive test and benchmark suite for the Turbo OCR server.

## Setup

```bash
pip install -r tests/requirements.txt
```

The OCR server must be running before tests can execute. Default: `http://localhost:8000` (HTTP) and `localhost:50051` (gRPC).

## Running Tests

### All tests

```bash
python tests/run_all.py
```

### Suites

Default run (`python tests/run_all.py`) covers fast-to-slow correctness:
`unit -> integration -> regression -> accuracy`. Stress and benchmark are
opt-in.

```bash
python tests/run_all.py                          # default correctness
python tests/run_all.py --suite unit
python tests/run_all.py --suite integration
python tests/run_all.py --suite accuracy         # ground-truth F1/CER
python tests/run_all.py --suite stress           # 60s soak (opt-in)
python tests/run_all.py --suite benchmark        # perf matrix (opt-in)
python tests/run_all.py --suite all              # everything
```

### Layout-enabled runs

```bash
ENABLE_LAYOUT=1 python tests/run_all.py --suite integration --suite accuracy
```

### Benchmark orchestrator

Single canonical entry point writes `tests/benchmark/LATEST.md` and
`LATEST.json` (gitignored):

```bash
python tests/benchmark/bench_matrix.py --quick             # ~2 min smoke
python tests/benchmark/bench_matrix.py                     # full 3-phase
python tests/benchmark/bench_matrix.py --phases stress --duration 120
```

### Direct pytest

```bash
pytest tests/unit/ -v
pytest tests/integration/ -v
pytest tests/regression/ -v
pytest tests/benchmark/ -v -s
```

### Custom server

```bash
python tests/run_all.py --server-url http://myhost:8000 --grpc-target myhost:50051
```

### Run specific test

```bash
pytest tests/integration/test_ocr_endpoint.py::TestOcrEndpoint::test_detects_known_text -v
```

## Test Suites

### unit/
Tests for individual functions and behaviors accessible through the API:
- **test_base64.py** -- Base64 decode handling (padding, whitespace, invalid chars)
- **test_json_response.py** -- JSON response format validation (schema, Content-Type, escaping)
- **test_box_sorting.py** -- Box sorting logic (reading order, Y-quantization)

### integration/
End-to-end tests against the running server:
- **test_ocr_endpoint.py** -- `/ocr` endpoint (base64 JSON, various formats/sizes)
- **test_ocr_raw_endpoint.py** -- `/ocr/raw` endpoint (raw bytes, format detection)
- **test_ocr_batch_endpoint.py** -- `/ocr/batch` endpoint (parallel processing, ordering)
- **test_grpc_endpoint.py** -- gRPC `OCRService.Recognize` and `RecognizeBatch`
- **test_error_handling.py** -- Error cases (bad input, empty body, corrupt images)
- **test_pdf_endpoint.py** -- `/ocr/pdf` endpoint (requires `reportlab`)

### regression/
Tests that catch regressions in accuracy and ordering:
- **test_accuracy_regression.py** -- Character recall must stay above threshold
- **test_ordering.py** -- Reading order (top-to-bottom, left-to-right, grid)
- **test_parallel_ordering.py** -- Concurrent requests return correct results (no cross-contamination)

### benchmark/
Performance benchmarks (use `-s` flag to see output):
- **bench_throughput.py** -- Images/second at concurrency 1-32
- **bench_latency.py** -- p50/p95/p99 latency percentiles
- **bench_concurrent.py** -- Burst handling and batch vs individual comparison
- **bench_parallel_pdf.py** -- PDF pages/second (requires `reportlab`)
- **bench_parallel_images.py** -- Parallel image processing speedup
- **bench_report.py** -- Generate a markdown report with all results

### Generate benchmark report

```bash
python tests/benchmark/bench_report.py --server-url http://localhost:8000 --output report.md
```

## Test Image Generation

Test images are generated programmatically using Pillow. No external test data files needed.

```bash
# Generate images to disk (optional, for inspection)
python tests/test_data/generate_test_images.py
```

## Optional Dependencies

- `reportlab` -- Required for PDF endpoint tests (`pip install reportlab`)
- `grpcio-tools` -- Required for gRPC tests (auto-compiles proto stubs)
