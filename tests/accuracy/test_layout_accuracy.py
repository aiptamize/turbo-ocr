"""Layout detection accuracy smoke test.

Runs only when ENABLE_LAYOUT=1. Asserts the server returns non-empty layout
boxes with sane coordinates on multi-column / table fixtures. No absolute
ground truth — PP-DocLayoutV3 has its own published numbers; this suite
just guards against regression-to-empty.
"""

import os
from pathlib import Path

import pytest
import requests

from conftest import PDF_DIR

pytestmark = [
    pytest.mark.accuracy,
    pytest.mark.layout,
    pytest.mark.skipif(
        os.environ.get("ENABLE_LAYOUT") != "1",
        reason="server not started with ENABLE_LAYOUT=1",
    ),
]

FIXTURES = [
    "academic_paper.pdf",
    "multi_column.pdf",
    "tables_document.pdf",
]


def _post_pdf(server_url, path):
    return requests.post(
        f"{server_url}/ocr/pdf?mode=ocr&layout=1",
        data=path.read_bytes(),
        headers={"Content-Type": "application/pdf"},
        timeout=120,
    )


@pytest.mark.parametrize("fixture", FIXTURES)
def test_layout_nonempty_on_structured_pdfs(server_url, fixture):
    p = PDF_DIR / fixture
    if not p.exists():
        pytest.skip(f"{fixture} not in fixtures")
    r = _post_pdf(server_url, p)
    assert r.status_code == 200
    layouts_per_page = [len(page.get("layout", [])) for page in r.json()["pages"]]
    assert sum(layouts_per_page) > 0, (
        f"{fixture}: layout returned 0 boxes across {len(layouts_per_page)} pages"
    )
    assert max(layouts_per_page) >= 2, (
        f"{fixture}: expected at least one page with >=2 layout blocks"
    )
