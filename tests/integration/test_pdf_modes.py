"""Integration tests for /ocr/pdf mode matrix.

Covers the 4 extraction modes (ocr, geometric, auto, auto_verified) on
small real-world PDF fixtures. Accuracy floors live in tests/accuracy/.
"""

import pytest
import requests

MODES = ["ocr", "geometric", "auto", "auto_verified"]


def _post_pdf(server_url, pdf_path, mode, timeout=60):
    return requests.post(
        f"{server_url}/ocr/pdf?mode={mode}",
        data=pdf_path.read_bytes(),
        headers={"Content-Type": "application/pdf"},
        timeout=timeout,
    )


@pytest.fixture(scope="module")
def small_pdfs(fixtures_dir):
    d = fixtures_dir / "pdf"
    fixtures = {
        "simple_letter": d / "simple_letter.pdf",
        "scanned": d / "scanned_document.pdf",
        "single_page_form": d / "single_page_form.pdf",
    }
    missing = [k for k, v in fixtures.items() if not v.exists()]
    if missing:
        pytest.skip(f"missing fixtures: {missing}")
    return fixtures


@pytest.mark.parametrize("mode", MODES)
class TestPdfModeMatrix:
    def test_returns_200_and_pages(self, server_url, small_pdfs, mode):
        r = _post_pdf(server_url, small_pdfs["simple_letter"], mode)
        assert r.status_code == 200, r.text
        data = r.json()
        assert "pages" in data
        assert len(data["pages"]) >= 1

    def test_page_schema(self, server_url, small_pdfs, mode):
        r = _post_pdf(server_url, small_pdfs["single_page_form"], mode)
        assert r.status_code == 200
        for page in r.json()["pages"]:
            assert "page" in page or "page_index" in page
            assert "results" in page
            for item in page["results"]:
                assert "text" in item
                assert "confidence" in item
                assert "bounding_box" in item
                assert len(item["bounding_box"]) == 4

    def test_results_nonempty_for_digital_pdf(self, server_url, small_pdfs, mode):
        r = _post_pdf(server_url, small_pdfs["simple_letter"], mode)
        data = r.json()
        total = sum(len(p["results"]) for p in data["pages"])
        assert total > 0, f"mode={mode} returned zero results on digital PDF"


class TestPdfModeSemantics:
    def test_geometric_on_image_only_pdf_returns_empty(self, server_url, fixtures_dir):
        """geometric reads the text layer only; an image-only PDF has none."""
        image_only = fixtures_dir / "edge_cases" / "no_text_layer.pdf"
        if not image_only.exists():
            pytest.skip("edge_cases/no_text_layer.pdf not generated")
        r = _post_pdf(server_url, image_only, "geometric")
        assert r.status_code == 200
        total = sum(len(p["results"]) for p in r.json()["pages"])
        assert total == 0, (
            f"geometric should return 0 results on image-only PDF, got {total}"
        )

    def test_auto_falls_back_to_ocr_on_image_only(self, server_url, fixtures_dir):
        image_only = fixtures_dir / "edge_cases" / "no_text_layer.pdf"
        if not image_only.exists():
            pytest.skip()
        r = _post_pdf(server_url, image_only, "auto")
        assert r.status_code == 200
        total = sum(len(p["results"]) for p in r.json()["pages"])
        assert total > 0, "auto should fall back to OCR on image-only PDF"

    def test_ocr_mode_works_on_digital(self, server_url, small_pdfs):
        r = _post_pdf(server_url, small_pdfs["simple_letter"], "ocr")
        assert r.status_code == 200
        total = sum(len(p["results"]) for p in r.json()["pages"])
        assert total > 0

    def test_auto_verified_not_worse_than_ocr(self, server_url, small_pdfs):
        r_ocr = _post_pdf(server_url, small_pdfs["simple_letter"], "ocr")
        r_av = _post_pdf(server_url, small_pdfs["simple_letter"], "auto_verified")
        total_ocr = sum(len(p["results"]) for p in r_ocr.json()["pages"])
        total_av = sum(len(p["results"]) for p in r_av.json()["pages"])
        assert total_av >= total_ocr * 0.8, (
            f"auto_verified ({total_av}) should not be much worse than ocr ({total_ocr})"
        )
