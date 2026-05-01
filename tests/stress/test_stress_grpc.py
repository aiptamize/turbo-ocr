"""60s soak on gRPC Recognize endpoint.

Imported here so the stress suite covers the gRPC surface too. Uses the
grpc_target fixture from tests/conftest.py.
"""

import concurrent.futures
import os
import time

import pytest

pytestmark = pytest.mark.stress


def test_stress_grpc_recognize(grpc_target):
    try:
        import grpc
        from tests._grpc_generated import ocr_pb2, ocr_pb2_grpc  # type: ignore
    except Exception as e:
        pytest.skip(f"gRPC stack unavailable: {e}")

    from conftest import IMAGES_DIR  # type: ignore
    image_path = IMAGES_DIR / "png" / "business_letter.png"
    if not image_path.exists():
        pytest.skip("no business_letter.png fixture")
    image_bytes = image_path.read_bytes()

    duration = float(os.environ.get("STRESS_DURATION", "60"))
    concurrency = 32

    channel = grpc.insecure_channel(grpc_target)
    stub = ocr_pb2_grpc.OCRServiceStub(channel)

    errors = 0
    ok = 0
    latencies: list[float] = []
    stop_at = time.monotonic() + duration

    def worker():
        nonlocal errors, ok
        local_lat = []
        local_ok = 0
        local_err = 0
        while time.monotonic() < stop_at:
            t0 = time.perf_counter()
            try:
                stub.Recognize(ocr_pb2.OCRRequest(image=image_bytes))
                local_lat.append((time.perf_counter() - t0) * 1000)
                local_ok += 1
            except Exception:
                local_err += 1
        return local_lat, local_ok, local_err

    t0 = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=concurrency) as ex:
        futures = [ex.submit(worker) for _ in range(concurrency)]
        for f in futures:
            lat, k, e = f.result()
            latencies.extend(lat)
            ok += k
            errors += e
    dt = time.perf_counter() - t0

    latencies.sort()
    total = ok + errors
    assert errors == 0, f"grpc stress: {errors}/{total} errors"
    rps = ok / dt
    p99 = latencies[min(int(len(latencies) * 0.99), len(latencies) - 1)] if latencies else 0
    assert rps >= 100.0, f"grpc stress: {rps:.1f} reqs/s below 100"
    assert p99 <= 500.0, f"grpc stress: p99 {p99:.1f}ms above 500"
