#!/usr/bin/env python3
"""Generate test images programmatically for the OCR test suite.

Run standalone to generate images to disk, or import the functions.
"""

import os
import sys
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont


def _get_font(size=28):
    font_paths = [
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/TTF/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf",
        "/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf",
        "/usr/share/fonts/liberation-sans/LiberationSans-Regular.ttf",
    ]
    for p in font_paths:
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def generate_single_word(word="HELLO", width=400, height=100, font_size=40):
    """Image with a single large word."""
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    draw.text((20, 20), word, fill="black", font=_get_font(font_size))
    return img


def generate_paragraph(lines=None, width=600, height=300, font_size=22):
    """Image with multiple lines of text."""
    if lines is None:
        lines = [
            "The quick brown fox jumps",
            "over the lazy dog near the",
            "river bank on a sunny day.",
        ]
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    font = _get_font(font_size)
    y = 20
    for line in lines:
        draw.text((20, y), line, fill="black", font=font)
        y += font_size + 10
    return img


def generate_numbers(width=400, height=100, font_size=40):
    """Image with digits."""
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    draw.text((20, 20), "0123456789", fill="black", font=_get_font(font_size))
    return img


def generate_special_characters(width=500, height=100, font_size=30):
    """Image with special characters the OCR might encounter."""
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    draw.text((20, 20), "@#$%& test-2024 (v1.0)", fill="black", font=_get_font(font_size))
    return img


def generate_ordering_grid(rows=3, cols=3, cell_w=200, cell_h=120, font_size=36):
    """Image with numbered blocks in a grid for ordering verification.

    Numbers are placed at known grid positions so the expected reading
    order (top-to-bottom, left-to-right) is deterministic.
    """
    width = cols * cell_w
    height = rows * cell_h
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    font = _get_font(font_size)
    num = 1
    for r in range(rows):
        for c in range(cols):
            x = c * cell_w + 30
            y = r * cell_h + 20
            draw.text((x, y), str(num), fill="black", font=font)
            num += 1
    return img


def generate_dense_document(width=800, height=1100, font_size=18, line_count=40):
    """Simulated dense A4-like document with many lines of text."""
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    font = _get_font(font_size)
    y = 30
    for i in range(line_count):
        draw.text((30, y), f"Line {i+1}: Sample document text for OCR testing purposes", fill="black", font=font)
        y += font_size + 8
    return img


def generate_low_contrast(text="LOW CONTRAST", width=400, height=100, font_size=30):
    """Image with low contrast text (light gray on white)."""
    img = Image.new("RGB", (width, height), (240, 240, 240))
    draw = ImageDraw.Draw(img)
    draw.text((20, 20), text, fill=(180, 180, 180), font=_get_font(font_size))
    return img


def generate_inverted(text="INVERTED", width=400, height=100, font_size=36):
    """White text on black background."""
    img = Image.new("RGB", (width, height), "black")
    draw = ImageDraw.Draw(img)
    draw.text((20, 20), text, fill="white", font=_get_font(font_size))
    return img


def generate_unique_image(identifier, width=400, height=100, font_size=36):
    """Image with a unique identifier string for concurrency testing."""
    img = Image.new("RGB", (width, height), "white")
    draw = ImageDraw.Draw(img)
    draw.text((20, 20), f"ID{identifier:06d}", fill="black", font=_get_font(font_size))
    return img


# ---------------------------------------------------------------------------
# CLI: generate all test images to a directory
# ---------------------------------------------------------------------------

def main():
    out_dir = Path(__file__).parent / "generated"
    out_dir.mkdir(exist_ok=True)

    images = {
        "single_word.png": generate_single_word(),
        "paragraph.png": generate_paragraph(),
        "numbers.png": generate_numbers(),
        "special_chars.png": generate_special_characters(),
        "ordering_3x3.png": generate_ordering_grid(),
        "dense_doc.png": generate_dense_document(),
        "low_contrast.png": generate_low_contrast(),
        "inverted.png": generate_inverted(),
    }
    for i in range(5):
        images[f"unique_{i}.png"] = generate_unique_image(i)

    for name, img in images.items():
        path = out_dir / name
        img.save(path, "PNG")
        print(f"  Generated {path}")

    print(f"\nGenerated {len(images)} test images in {out_dir}")


if __name__ == "__main__":
    main()
