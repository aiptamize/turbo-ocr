"""Ground-truth F1 accuracy tests for /ocr/pdf across all 4 modes."""

from pathlib import Path

import pytest
import requests

import sys
sys.path.insert(0, str(Path(__file__).parent))

from conftest import PDF_DIR
from _scoring import load_floors, load_json, score_pdf, tokenize, word_f1

pytestmark = pytest.mark.accuracy

FLOORS_PATH = Path(__file__).parent / "floors.json"
EXPECTED_DIR = PDF_DIR / "expected"
MODES = ["ocr", "geometric", "auto", "auto_verified"]

_PDFS = sorted(PDF_DIR.glob("*.pdf"))


def _floor(stem: str, mode: str) -> float:
    floors = load_floors(FLOORS_PATH).get("pdf", {})
    per = floors.get("per_fixture", {}).get(stem)
    if per is not None and mode in per:
        return float(per[mode])
    return float(floors.get("default_f1", {}).get(mode, 0.5))


def _expected_for(pdf_path: Path):
    exp = EXPECTED_DIR / f"{pdf_path.stem}.json"
    return load_json(exp) if exp.exists() else None


def _post_pdf(server_url, pdf_path, mode):
    r = requests.post(
        f"{server_url}/ocr/pdf?mode={mode}",
        data=pdf_path.read_bytes(),
        headers={"Content-Type": "application/pdf"},
        timeout=120,
    )
    r.raise_for_status()
    return r.json()


@pytest.mark.parametrize("mode", MODES)
@pytest.mark.parametrize("pdf", _PDFS, ids=[p.stem for p in _PDFS])
class TestPdfAccuracy:
    def test_f1_above_floor(self, server_url, pdf, mode):
        expected = _expected_for(pdf)
        if expected is None:
            pytest.skip(f"no expected/{pdf.stem}.json")
        actual = _post_pdf(server_url, pdf, mode)
        s = score_pdf(expected, actual)
        floor = _floor(pdf.stem, mode)
        assert s["f1"] >= floor, (
            f"{pdf.stem} mode={mode}: F1={s['f1']:.3f} < floor {floor:.3f} "
            f"(per_page={['%.2f' % f for f in s['per_page_f1'][:5]]})"
        )


class TestPdfCrossModeInvariants:
    def _digital_pdfs(self):
        return [p for p in _PDFS if p.stem != "scanned_document"]

    def test_geometric_near_perfect_on_digital(self, server_url):
        """Geometric reads the text layer; F1 vs expected should be very high."""
        pdfs = self._digital_pdfs()
        if not pdfs:
            pytest.skip()
        for pdf in pdfs[:3]:
            expected = _expected_for(pdf)
            if expected is None:
                continue
            actual = _post_pdf(server_url, pdf, "geometric")
            s = score_pdf(expected, actual)
            assert s["f1"] >= 0.60, (
                f"{pdf.stem} geometric F1 {s['f1']:.3f} < 0.60 "
                f"(geometric should match text layer closely)"
            )

    def test_ocr_and_geometric_agree_on_digital(self, server_url):
        pdfs = self._digital_pdfs()
        if not pdfs:
            pytest.skip()
        pdf = pdfs[0]
        ocr = _post_pdf(server_url, pdf, "ocr")
        geo = _post_pdf(server_url, pdf, "geometric")
        ocr_tok = tokenize(" ".join(
            r["text"] for p in ocr["pages"] for r in p["results"]
        ))
        geo_tok = tokenize(" ".join(
            r["text"] for p in geo["pages"] for r in p["results"]
        ))
        s = word_f1(ocr_tok, geo_tok)
        assert s["f1"] >= 0.60, (
            f"{pdf.stem}: ocr vs geometric agreement F1={s['f1']:.3f} < 0.60"
        )

    def test_auto_verified_not_much_worse_than_ocr(self, server_url):
        pdfs = self._digital_pdfs()
        if not pdfs:
            pytest.skip()
        for pdf in pdfs[:3]:
            expected = _expected_for(pdf)
            if expected is None:
                continue
            ocr_score = score_pdf(expected, _post_pdf(server_url, pdf, "ocr"))
            av_score = score_pdf(expected, _post_pdf(server_url, pdf, "auto_verified"))
            assert av_score["f1"] >= ocr_score["f1"] - 0.10, (
                f"{pdf.stem}: auto_verified F1={av_score['f1']:.3f} "
                f"much worse than ocr F1={ocr_score['f1']:.3f}"
            )
