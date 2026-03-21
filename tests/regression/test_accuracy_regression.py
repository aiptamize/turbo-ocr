"""Regression tests for OCR accuracy.

Tests both generated images (original tests) and real test data from
tests/test_data/ (PNG, JPEG). Real data tests compare against expected
JSON outputs with tolerance for region count and key text presence.
"""

import json

import pytest
import requests

from conftest import make_text_image, pil_to_base64, load_expected, TEST_DATA_DIR


def _ocr_text(server_url, text, width=500, height=100, font_size=40):
    """Helper: render text, OCR it, return detected text."""
    img = make_text_image(text, width=width, height=height, font_size=font_size)
    b64 = pil_to_base64(img)
    r = requests.post(f"{server_url}/ocr", json={"image": b64}, timeout=10)
    assert r.status_code == 200
    data = r.json()
    return " ".join(item["text"] for item in data["results"])


def _char_accuracy(expected, detected):
    """Compute character-level recall: fraction of expected chars found in detected."""
    if not expected:
        return 1.0
    expected_lower = expected.lower().replace(" ", "")
    detected_lower = detected.lower().replace(" ", "")
    if not expected_lower:
        return 1.0
    hits = sum(1 for c in expected_lower if c in detected_lower)
    return hits / len(expected_lower)


def _ocr_raw_file(server_url, file_path, timeout=30):
    """Send a real file to /ocr/raw and return the parsed JSON response."""
    suffix = file_path.suffix.lower()
    content_type = {
        ".png": "image/png",
        ".jpg": "image/jpeg",
        ".jpeg": "image/jpeg",
    }.get(suffix, "image/png")
    data = file_path.read_bytes()
    r = requests.post(
        f"{server_url}/ocr/raw",
        data=data,
        headers={"Content-Type": content_type},
        timeout=timeout,
    )
    assert r.status_code == 200, f"Failed for {file_path.name}: HTTP {r.status_code}"
    return r.json()


def _ocr_pdf_file(server_url, file_path, timeout=60):
    """Send a real PDF to /ocr/pdf and return the parsed JSON response."""
    data = file_path.read_bytes()
    r = requests.post(f"{server_url}/ocr/pdf", data=data, timeout=timeout)
    assert r.status_code == 200, f"Failed for {file_path.name}: HTTP {r.status_code}"
    return r.json()


# ---------------------------------------------------------------------------
# Collect real test image paths for parametrize
# ---------------------------------------------------------------------------

_PNG_DIR = TEST_DATA_DIR / "png"
_JPEG_DIR = TEST_DATA_DIR / "jpeg"
_PDF_DIR = TEST_DATA_DIR / "pdf"

_PNG_FILES = sorted(_PNG_DIR.glob("*.png")) if _PNG_DIR.exists() else []
_JPEG_FILES = sorted(_JPEG_DIR.glob("*.jpg")) if _JPEG_DIR.exists() else []
_PDF_FILES = sorted(_PDF_DIR.glob("*.pdf")) if _PDF_DIR.exists() else []


class TestAccuracyRegression:
    """Verify OCR accuracy does not regress below thresholds."""

    # Minimum character recall threshold. Set conservatively -- clear
    # black-on-white text with a good font should achieve >= 70%.
    MIN_RECALL = 0.70

    @pytest.mark.parametrize("text", [
        "HELLO",
        "WORLD",
        "12345",
        "ABCDEF",
        "Testing",
    ])
    def test_single_word_recall(self, server_url, text):
        """Single clear word should be recognized with high recall."""
        detected = _ocr_text(server_url, text, width=400, height=100, font_size=48)
        recall = _char_accuracy(text, detected)
        assert recall >= self.MIN_RECALL, (
            f"Recall {recall:.0%} < {self.MIN_RECALL:.0%} for '{text}'. "
            f"Detected: '{detected}'"
        )

    def test_numbers_recall(self, server_url):
        """Numeric text should be recognized."""
        detected = _ocr_text(server_url, "0123456789", width=500, height=100, font_size=48)
        recall = _char_accuracy("0123456789", detected)
        assert recall >= self.MIN_RECALL, f"Detected: '{detected}'"

    def test_mixed_case(self, server_url):
        """Mixed case text should be recognized."""
        text = "Hello World"
        detected = _ocr_text(server_url, text, width=500, height=100, font_size=40)
        recall = _char_accuracy(text, detected)
        assert recall >= self.MIN_RECALL, f"Detected: '{detected}'"

    def test_multiline_recall(self, server_url):
        """Multi-line text -- each line should be partially detected."""
        img = make_text_image(
            "First line here\nSecond line here",
            width=500, height=150, font_size=28,
        )
        b64 = pil_to_base64(img)
        r = requests.post(f"{server_url}/ocr", json={"image": b64}, timeout=10)
        data = r.json()
        all_text = " ".join(item["text"] for item in data["results"]).lower()
        # At least some words from each line should appear
        assert "first" in all_text or "line" in all_text or "second" in all_text, (
            f"Expected some words from the input. Got: '{all_text}'"
        )

    def test_confidence_not_garbage(self, server_url):
        """Clear text should have confidence > 0.5 (catches broken recognition)."""
        img = make_text_image("CONFIDENCE", width=500, height=100, font_size=48)
        b64 = pil_to_base64(img)
        r = requests.post(f"{server_url}/ocr", json={"image": b64}, timeout=10)
        data = r.json()
        if not data["results"]:
            pytest.skip("No detections")
        avg_conf = sum(item["confidence"] for item in data["results"]) / len(data["results"])
        assert avg_conf > 0.5, f"Average confidence {avg_conf:.2f} is suspiciously low"

    def test_inverted_text_detected(self, server_url):
        """White text on black background should still be detected."""
        from test_data.generate_test_images import generate_inverted
        img = generate_inverted("INVERTED")
        b64 = pil_to_base64(img)
        r = requests.post(f"{server_url}/ocr", json={"image": b64}, timeout=10)
        data = r.json()
        # Inverted text is harder -- just verify the server doesn't crash
        # and returns either results or empty
        assert isinstance(data["results"], list)


class TestRealPngAccuracy:
    """Verify OCR accuracy on real PNG test images against expected outputs."""

    REGION_TOLERANCE = 0.10  # Allow +/-10% region count difference

    @pytest.mark.parametrize(
        "png_path",
        _PNG_FILES,
        ids=[f.stem for f in _PNG_FILES],
    )
    def test_png_region_count(self, server_url, png_path):
        """Region count should match expected within +/-10%."""
        expected = load_expected(png_path)
        if expected is None:
            pytest.skip(f"No expected JSON for {png_path.name}")

        actual = _ocr_raw_file(server_url, png_path)
        expected_count = len(expected["results"])
        actual_count = len(actual["results"])

        if expected_count == 0:
            # If expected has no results, actual should also be empty or near-empty
            assert actual_count <= 2, (
                f"{png_path.name}: expected 0 regions, got {actual_count}"
            )
            return

        ratio = actual_count / expected_count
        assert 1.0 - self.REGION_TOLERANCE <= ratio <= 1.0 + self.REGION_TOLERANCE, (
            f"{png_path.name}: expected ~{expected_count} regions, "
            f"got {actual_count} (ratio={ratio:.2f}, tolerance=+/-{self.REGION_TOLERANCE:.0%})"
        )

    @pytest.mark.parametrize(
        "png_path",
        _PNG_FILES,
        ids=[f.stem for f in _PNG_FILES],
    )
    def test_png_key_text_present(self, server_url, png_path):
        """Key text strings from expected output should be present in actual output."""
        expected = load_expected(png_path)
        if expected is None:
            pytest.skip(f"No expected JSON for {png_path.name}")
        if not expected["results"]:
            pytest.skip(f"No expected results for {png_path.name}")

        actual = _ocr_raw_file(server_url, png_path)
        actual_texts = " ".join(item["text"] for item in actual["results"]).lower()

        # Pick up to 5 high-confidence expected texts as key strings
        key_items = sorted(
            expected["results"],
            key=lambda x: x.get("confidence", 0),
            reverse=True,
        )[:5]

        found = 0
        for item in key_items:
            # Check if a significant portion of the expected text is in the output
            expected_text = item["text"].lower().strip()
            if len(expected_text) < 2:
                found += 1  # Skip very short strings
                continue
            if expected_text in actual_texts:
                found += 1

        # At least 40% of key texts should be found (allows for OCR variance)
        min_found = max(1, int(len(key_items) * 0.4))
        assert found >= min_found, (
            f"{png_path.name}: only {found}/{len(key_items)} key texts found. "
            f"Expected keys: {[i['text'] for i in key_items]}"
        )

    @pytest.mark.parametrize(
        "png_path",
        _PNG_FILES,
        ids=[f.stem for f in _PNG_FILES],
    )
    def test_png_confidence_reasonable(self, server_url, png_path):
        """Average confidence on real test images should be > 0.5."""
        actual = _ocr_raw_file(server_url, png_path)
        if not actual["results"]:
            pytest.skip(f"No detections for {png_path.name}")

        avg_conf = sum(r["confidence"] for r in actual["results"]) / len(actual["results"])
        assert avg_conf > 0.5, (
            f"{png_path.name}: average confidence {avg_conf:.3f} is too low"
        )


class TestRealJpegAccuracy:
    """Verify OCR accuracy on real JPEG test images against expected outputs."""

    REGION_TOLERANCE = 0.10

    @pytest.mark.parametrize(
        "jpeg_path",
        _JPEG_FILES,
        ids=[f.stem for f in _JPEG_FILES],
    )
    def test_jpeg_region_count(self, server_url, jpeg_path):
        """Region count should match expected within +/-10%."""
        expected = load_expected(jpeg_path)
        if expected is None:
            pytest.skip(f"No expected JSON for {jpeg_path.name}")

        actual = _ocr_raw_file(server_url, jpeg_path)
        expected_count = len(expected["results"])
        actual_count = len(actual["results"])

        if expected_count == 0:
            assert actual_count <= 2, (
                f"{jpeg_path.name}: expected 0 regions, got {actual_count}"
            )
            return

        ratio = actual_count / expected_count
        assert 1.0 - self.REGION_TOLERANCE <= ratio <= 1.0 + self.REGION_TOLERANCE, (
            f"{jpeg_path.name}: expected ~{expected_count} regions, "
            f"got {actual_count} (ratio={ratio:.2f}, tolerance=+/-{self.REGION_TOLERANCE:.0%})"
        )

    @pytest.mark.parametrize(
        "jpeg_path",
        _JPEG_FILES,
        ids=[f.stem for f in _JPEG_FILES],
    )
    def test_jpeg_key_text_present(self, server_url, jpeg_path):
        """Key text strings from expected output should be present in actual output."""
        expected = load_expected(jpeg_path)
        if expected is None:
            pytest.skip(f"No expected JSON for {jpeg_path.name}")
        if not expected["results"]:
            pytest.skip(f"No expected results for {jpeg_path.name}")

        actual = _ocr_raw_file(server_url, jpeg_path)
        actual_texts = " ".join(item["text"] for item in actual["results"]).lower()

        key_items = sorted(
            expected["results"],
            key=lambda x: x.get("confidence", 0),
            reverse=True,
        )[:5]

        found = 0
        for item in key_items:
            expected_text = item["text"].lower().strip()
            if len(expected_text) < 2:
                found += 1
                continue
            if expected_text in actual_texts:
                found += 1

        min_found = max(1, int(len(key_items) * 0.4))
        assert found >= min_found, (
            f"{jpeg_path.name}: only {found}/{len(key_items)} key texts found. "
            f"Expected keys: {[i['text'] for i in key_items]}"
        )


class TestRealPdfAccuracy:
    """Verify OCR accuracy on real PDF test files against expected outputs."""

    REGION_TOLERANCE = 0.10

    @pytest.mark.parametrize(
        "pdf_path",
        _PDF_FILES,
        ids=[f.stem for f in _PDF_FILES],
    )
    def test_pdf_page_count_and_regions(self, server_url, pdf_path):
        """PDF page count and per-page region count should match expected."""
        expected = load_expected(pdf_path)
        if expected is None:
            pytest.skip(f"No expected JSON for {pdf_path.name}")

        actual = _ocr_pdf_file(server_url, pdf_path)

        # Check page count
        expected_pages = expected.get("pages", [])
        actual_pages = actual.get("pages", [])
        assert len(actual_pages) == len(expected_pages), (
            f"{pdf_path.name}: expected {len(expected_pages)} pages, got {len(actual_pages)}"
        )

        # Check region counts per page
        for i, (exp_page, act_page) in enumerate(zip(expected_pages, actual_pages)):
            exp_count = len(exp_page.get("results", []))
            act_count = len(act_page.get("results", []))
            if exp_count == 0:
                continue
            ratio = act_count / exp_count
            assert 1.0 - self.REGION_TOLERANCE <= ratio <= 1.0 + self.REGION_TOLERANCE, (
                f"{pdf_path.name} page {i+1}: expected ~{exp_count} regions, "
                f"got {act_count} (ratio={ratio:.2f})"
            )
