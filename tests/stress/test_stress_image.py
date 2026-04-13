"""60s soak on /ocr/raw. Opt-in via -m stress."""

import pytest

from _helpers import assert_healthy, run_stress

pytestmark = pytest.mark.stress


def test_stress_ocr_raw(server_url):
    r = run_stress(server_url, "/ocr/raw")
    assert_healthy(r)
    assert r["reqs_per_s"] >= 100.0, (
        f"ocr_raw: {r['reqs_per_s']:.1f} reqs/s below 100 floor"
    )
    assert r["p99_ms"] <= 500.0, (
        f"ocr_raw: p99 {r['p99_ms']:.1f}ms above 500 ceiling"
    )
