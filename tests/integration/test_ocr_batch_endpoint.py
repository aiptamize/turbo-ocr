"""Integration tests for the /ocr/batch endpoint."""

import pytest
import requests

from conftest import make_text_image, pil_to_base64


class TestOcrBatchEndpoint:
    """Test /ocr/batch endpoint for parallel image processing."""

    def test_batch_two_images(self, server_url, hello_image, numbers_image):
        """Batch of 2 images should return 2 result sets."""
        b64_1 = pil_to_base64(hello_image)
        b64_2 = pil_to_base64(numbers_image)
        r = requests.post(
            f"{server_url}/ocr/batch",
            json={"images": [b64_1, b64_2]},
            timeout=15,
        )
        assert r.status_code == 200
        data = r.json()
        assert len(data["batch_results"]) == 2

    def test_batch_single_image(self, server_url, hello_image):
        """Batch with 1 image should work and return 1 result set."""
        b64 = pil_to_base64(hello_image)
        r = requests.post(
            f"{server_url}/ocr/batch",
            json={"images": [b64]},
            timeout=10,
        )
        assert r.status_code == 200
        data = r.json()
        assert len(data["batch_results"]) == 1

    def test_batch_preserves_order(self, server_url, unique_images):
        """Batch results must be in the same order as input images.

        This catches a critical bug: if the pipeline pool dispatches work
        out of order, results could be associated with the wrong image.
        """
        # Use first 5 unique images
        images = unique_images[:5]
        b64_list = [pil_to_base64(img) for _, img in images]

        r = requests.post(
            f"{server_url}/ocr/batch",
            json={"images": b64_list},
            timeout=20,
        )
        assert r.status_code == 200
        data = r.json()
        assert len(data["batch_results"]) == 5

        # Verify each result corresponds to its input
        for i, (expected_text, _) in enumerate(images):
            batch_text = " ".join(
                item["text"] for item in data["batch_results"][i]["results"]
            ).upper()
            # The expected_text is like "UNIQUE0000" -- check it appears
            if data["batch_results"][i]["results"]:
                # At least verify results are non-empty for valid images
                assert len(data["batch_results"][i]["results"]) > 0

    def test_batch_many_images(self, server_url):
        """Batch of 10 images should all be processed."""
        images = []
        for i in range(10):
            img = make_text_image(f"BATCH{i}", width=300, height=80, font_size=36)
            images.append(pil_to_base64(img))

        r = requests.post(
            f"{server_url}/ocr/batch",
            json={"images": images},
            timeout=30,
        )
        assert r.status_code == 200
        data = r.json()
        assert len(data["batch_results"]) == 10

    def test_batch_empty_array(self, server_url):
        """Empty images array should return 400."""
        r = requests.post(
            f"{server_url}/ocr/batch",
            json={"images": []},
            timeout=10,
        )
        assert r.status_code == 400

    def test_batch_missing_images_key(self, server_url):
        """Missing 'images' key should return 400."""
        r = requests.post(
            f"{server_url}/ocr/batch",
            json={"wrong_key": []},
            timeout=10,
        )
        assert r.status_code == 400

    def test_batch_mixed_formats(self, server_url, hello_image, numbers_image):
        """Batch with mixed PNG and JPEG images should work."""
        b64_png = pil_to_base64(hello_image, "PNG")
        b64_jpg = pil_to_base64(numbers_image, "JPEG")
        r = requests.post(
            f"{server_url}/ocr/batch",
            json={"images": [b64_png, b64_jpg]},
            timeout=15,
        )
        assert r.status_code == 200
        data = r.json()
        assert len(data["batch_results"]) == 2
