"""60s soak on /ocr/batch."""

import pytest

from _helpers import assert_healthy, run_stress

pytestmark = pytest.mark.stress


def test_stress_ocr_batch(server_url):
    r = run_stress(server_url, "/ocr/batch")
    assert_healthy(r)
    assert r["units_per_s"] >= 50.0, (
        f"ocr_batch: {r['units_per_s']:.1f} imgs/s below floor"
    )
