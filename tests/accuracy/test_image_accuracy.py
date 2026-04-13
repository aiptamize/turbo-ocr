"""Ground-truth F1 accuracy tests for image endpoints."""

import json
from pathlib import Path

import pytest
import requests

import sys
sys.path.insert(0, str(Path(__file__).parent))

from conftest import IMAGES_DIR
from _scoring import load_floors, load_json, score_image, tokenize, word_f1

pytestmark = pytest.mark.accuracy

FLOORS_PATH = Path(__file__).parent / "floors.json"
EXPECTED_DIR = IMAGES_DIR / "expected"

_PNG = sorted((IMAGES_DIR / "png").glob("*.png"))
_JPEG = sorted((IMAGES_DIR / "jpeg").glob("*.jpg"))
_ALL_IMAGES = _PNG + _JPEG


def _floor(stem: str) -> float:
    floors = load_floors(FLOORS_PATH).get("image", {})
    per = floors.get("per_fixture", {}).get(stem)
    if per is not None:
        return float(per)
    return float(floors.get("default_f1", 0.5))


def _expected_for(img_path: Path):
    exp_path = EXPECTED_DIR / f"{img_path.stem}.json"
    if not exp_path.exists():
        return None
    return load_json(exp_path)


def _post_raw(server_url, img_path):
    suffix = img_path.suffix.lower()
    ct = {".png": "image/png", ".jpg": "image/jpeg", ".jpeg": "image/jpeg"}[suffix]
    r = requests.post(
        f"{server_url}/ocr/raw",
        data=img_path.read_bytes(),
        headers={"Content-Type": ct},
        timeout=30,
    )
    r.raise_for_status()
    return r.json()


def _post_base64(server_url, img_path):
    import base64
    b64 = base64.b64encode(img_path.read_bytes()).decode()
    r = requests.post(f"{server_url}/ocr", json={"image": b64}, timeout=30)
    r.raise_for_status()
    return r.json()


class TestImageAccuracyRaw:
    @pytest.mark.parametrize("img", _ALL_IMAGES, ids=[p.stem for p in _ALL_IMAGES])
    def test_f1_above_floor_ocr_raw(self, server_url, img):
        expected = _expected_for(img)
        if expected is None:
            pytest.skip(f"no expected/{img.stem}.json")
        actual = _post_raw(server_url, img)
        s = score_image(expected, actual)
        floor = _floor(img.stem)
        assert s["f1"] >= floor, (
            f"{img.stem}: F1={s['f1']:.3f} < floor {floor:.3f} "
            f"(exp_regions={s['n_expected']}, act_regions={s['n_actual']}, cer={s['cer']:.3f})"
        )


class TestImageAccuracyJson:
    @pytest.mark.parametrize("img", _ALL_IMAGES, ids=[p.stem for p in _ALL_IMAGES])
    def test_f1_above_floor_ocr_json(self, server_url, img):
        expected = _expected_for(img)
        if expected is None:
            pytest.skip()
        actual = _post_base64(server_url, img)
        s = score_image(expected, actual)
        floor = _floor(img.stem)
        assert s["f1"] >= floor, (
            f"{img.stem}: F1={s['f1']:.3f} < floor {floor:.3f}"
        )


class TestBatchMatchesIndividual:
    """Critical correctness: /ocr/batch must score the same as individual /ocr calls."""

    def test_batch_f1_matches_individual(self, server_url):
        import base64
        imgs = _ALL_IMAGES[:5]
        if not imgs:
            pytest.skip()

        individual = [_post_base64(server_url, p) for p in imgs]
        b64s = [base64.b64encode(p.read_bytes()).decode() for p in imgs]
        r = requests.post(
            f"{server_url}/ocr/batch", json={"images": b64s}, timeout=60
        )
        r.raise_for_status()
        batch = r.json()
        assert "batch_results" in batch
        assert len(batch["batch_results"]) == len(imgs)

        for img, ind, bat in zip(imgs, individual, batch["batch_results"]):
            ind_tok = tokenize(" ".join(i["text"] for i in ind["results"]))
            bat_tok = tokenize(" ".join(i["text"] for i in bat["results"]))
            s = word_f1(ind_tok, bat_tok)
            assert s["f1"] >= 0.95, (
                f"{img.stem}: batch F1 vs individual {s['f1']:.3f} < 0.95"
            )
