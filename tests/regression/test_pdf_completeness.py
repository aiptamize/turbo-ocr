"""Regression: PDF endpoint must return every page fully decoded.

Guards a class of bugs where the streaming-render scratch tmpdir gets
unlinked before the dispatcher's OCR workers actually open the PPM
files — the symptom is pages silently coming back with width=0,
height=0 and an empty results list. This was caused by confining
StreamHandle's lifetime to a helper while the page futures lived in
the caller, so the helper's return ran ~StreamHandle and removed the
tmpdir out from under the still-pending GPU decodes.

The fixture chosen here (multi_column.pdf, 27 pages, dense text)
makes the GPU pool busy enough relative to the renderer that any
lifetime regression of that shape will deterministically drop the
tail pages.
"""

from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import pytest
import requests

PDF_FIXTURES_DIR = Path(__file__).resolve().parent.parent / "fixtures" / "pdf"
MULTI_COLUMN = PDF_FIXTURES_DIR / "multi_column.pdf"


def _post_pdf(server_url: str, path: Path) -> dict:
    with path.open("rb") as f:
        r = requests.post(
            f"{server_url}/ocr/pdf?mode=ocr",
            files={"file": (path.name, f)},
            timeout=180,
        )
    r.raise_for_status()
    return r.json()


def _assert_all_pages_decoded(data: dict, name: str, run: int) -> None:
    pages = data.get("pages") or []
    assert pages, f"{name} run {run}: response had no pages"
    bad = [
        i for i, p in enumerate(pages)
        if p.get("width", 0) == 0 or p.get("height", 0) == 0
    ]
    assert not bad, (
        f"{name} run {run}: {len(bad)}/{len(pages)} pages came back empty "
        f"(width=0 or height=0). Empty page indices: {bad}. "
        f"This usually means the StreamHandle's tmpdir was unlinked before "
        f"the GPU pool decoded the PPMs — see src/routes/pdf_routes.cpp "
        f"`run_streamed_render_gpu` and the comment on its return value."
    )


@pytest.mark.skipif(
    not MULTI_COLUMN.exists(),
    reason=f"fixture missing: {MULTI_COLUMN}",
)
class TestPdfStreamingCompleteness:
    """Every page of a real multi-page PDF must decode end-to-end."""

    def test_serial_no_empty_pages(self, server_url):
        """5 sequential requests, every page must be non-empty."""
        for run in range(5):
            data = _post_pdf(server_url, MULTI_COLUMN)
            _assert_all_pages_decoded(data, MULTI_COLUMN.name, run)

    def test_concurrent_no_empty_pages(self, server_url):
        """4 parallel requests × 3 iterations.

        Concurrency stresses the dispatcher pool — when the pool is
        saturated, decode_ppm runs further from the renderer's
        StreamHandle ctor, which is exactly when a lifetime bug
        manifests as empty tail pages.
        """
        for iteration in range(3):
            with ThreadPoolExecutor(max_workers=4) as pool:
                futures = [
                    pool.submit(_post_pdf, server_url, MULTI_COLUMN)
                    for _ in range(4)
                ]
                for j, fut in enumerate(futures):
                    data = fut.result()
                    _assert_all_pages_decoded(
                        data,
                        MULTI_COLUMN.name,
                        run=iteration * 4 + j,
                    )
