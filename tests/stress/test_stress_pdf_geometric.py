"""60s soak on /ocr/pdf?mode=geometric at c=8.

Concurrency is capped at 8 because PDFium serializes behind a global
mutex (see feature/pdfExtractionModes notes). Throughput ceiling ~135 p/s.
"""

import pytest

from _helpers import assert_healthy, run_stress

pytestmark = pytest.mark.stress


def test_stress_pdf_geometric(server_url):
    r = run_stress(server_url, "/ocr/pdf?mode=geometric")
    assert_healthy(r)
    assert r["units_per_s"] >= 80.0, (
        f"pdf_geometric: {r['units_per_s']:.1f} pages/s below 80 floor"
    )


def test_stress_pdf_auto_verified(server_url):
    r = run_stress(server_url, "/ocr/pdf?mode=auto_verified")
    assert_healthy(r)
    assert r["units_per_s"] >= 50.0, (
        f"pdf_auto_verified: {r['units_per_s']:.1f} pages/s below 50 floor"
    )
