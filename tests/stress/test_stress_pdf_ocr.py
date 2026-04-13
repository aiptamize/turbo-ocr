"""60s soak on /ocr/pdf?mode=ocr at c=32. SOTA PDF throughput target."""

import pytest

from _helpers import assert_healthy, run_stress

pytestmark = pytest.mark.stress


def test_stress_pdf_ocr(server_url):
    r = run_stress(server_url, "/ocr/pdf?mode=ocr")
    assert_healthy(r)
    assert r["units_per_s"] >= 200.0, (
        f"pdf_ocr: {r['units_per_s']:.1f} pages/s below 200 floor"
    )
    assert r["p99_ms"] <= 2500.0, (
        f"pdf_ocr: p99 {r['p99_ms']:.1f}ms above 2500 ceiling"
    )
