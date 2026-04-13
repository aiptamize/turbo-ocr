"""Generate edge-case fixtures for the integration + stress test suites.

Run once; commit the outputs. Idempotent — skips files that already exist
unless --force is passed.

Produces under tests/fixtures/edge_cases/:
    zero_byte.bin
    one_pixel.png
    4k_text.png
    corrupt_truncated.jpg
    garbage_random.bin
    single_pixel_pixels.bgr
    whitespace_only.pdf
    pdf_500_pages.pdf
    rotated_90.pdf
    password_protected.pdf
    no_text_layer.pdf
    cropbox_offset.pdf
"""

import argparse
import os
import random
import struct
from pathlib import Path

from PIL import Image, ImageDraw, ImageFont
from reportlab.lib.pagesizes import A4, letter
from reportlab.pdfgen import canvas
from pypdf import PdfReader, PdfWriter
from pypdf.generic import RectangleObject

HERE = Path(__file__).resolve().parent
OUT = HERE / "edge_cases"


def _font(size=20):
    for p in (
        "/usr/share/fonts/truetype/dejavu/DejaVuSans.ttf",
        "/usr/share/fonts/TTF/DejaVuSans.ttf",
        "/usr/share/fonts/dejavu-sans-fonts/DejaVuSans.ttf",
    ):
        if os.path.exists(p):
            return ImageFont.truetype(p, size)
    return ImageFont.load_default()


def _skip_if_exists(path, force):
    if path.exists() and not force:
        print(f"  skip {path.name} (exists)")
        return True
    return False


def gen_zero_byte(force):
    p = OUT / "zero_byte.bin"
    if _skip_if_exists(p, force): return
    p.write_bytes(b"")
    print(f"  wrote {p.name} (0 bytes)")


def gen_one_pixel(force):
    p = OUT / "one_pixel.png"
    if _skip_if_exists(p, force): return
    Image.new("RGB", (1, 1), (255, 255, 255)).save(p, format="PNG")
    print(f"  wrote {p.name}")


def gen_4k_text(force):
    p = OUT / "4k_text.png"
    if _skip_if_exists(p, force): return
    img = Image.new("RGB", (3840, 2160), "white")
    draw = ImageDraw.Draw(img)
    font = _font(32)
    lines = [
        "The quick brown fox jumps over the lazy dog 0123456789",
        "Lorem ipsum dolor sit amet consectetur adipiscing elit",
        "EDGE CASE 4K RESOLUTION BENCHMARK FIXTURE",
    ]
    y = 50
    for repeat in range(40):
        for line in lines:
            draw.text((50, y), line, fill="black", font=font)
            y += 50
            if y > 2100:
                break
        if y > 2100:
            break
    img.save(p, format="PNG", optimize=False)
    print(f"  wrote {p.name}")


def gen_corrupt_jpeg(force):
    p = OUT / "corrupt_truncated.jpg"
    if _skip_if_exists(p, force): return
    img = Image.new("RGB", (400, 200), "white")
    d = ImageDraw.Draw(img)
    d.text((20, 80), "will be truncated", fill="black", font=_font(24))
    tmp = OUT / "_full.jpg"
    img.save(tmp, format="JPEG", quality=90)
    data = tmp.read_bytes()
    p.write_bytes(data[:200])
    tmp.unlink()
    print(f"  wrote {p.name} (200 bytes of {len(data)})")


def gen_garbage_random(force):
    p = OUT / "garbage_random.bin"
    if _skip_if_exists(p, force): return
    rng = random.Random(42)
    p.write_bytes(bytes(rng.randrange(256) for _ in range(8192)))
    print(f"  wrote {p.name} (8192 bytes)")


def gen_single_pixel_bgr(force):
    p = OUT / "single_pixel_pixels.bgr"
    if _skip_if_exists(p, force): return
    p.write_bytes(b"\x00\x00\x00")
    print(f"  wrote {p.name} (3 bytes)")


def gen_whitespace_pdf(force):
    p = OUT / "whitespace_only.pdf"
    if _skip_if_exists(p, force): return
    c = canvas.Canvas(str(p), pagesize=A4)
    c.showPage()
    c.save()
    print(f"  wrote {p.name}")


def gen_500_page_pdf(force):
    p = OUT / "pdf_500_pages.pdf"
    if _skip_if_exists(p, force): return
    c = canvas.Canvas(str(p), pagesize=letter)
    for i in range(1, 501):
        c.setFont("Helvetica", 14)
        c.drawString(100, 750, f"Page {i} of 500")
        c.drawString(100, 720, "Edge case: very long document")
        c.drawString(100, 690, f"Content repeats for page number {i}")
        c.showPage()
    c.save()
    print(f"  wrote {p.name}")


def gen_rotated_pdf(force):
    p = OUT / "rotated_90.pdf"
    if _skip_if_exists(p, force): return
    tmp = OUT / "_upright.pdf"
    c = canvas.Canvas(str(tmp), pagesize=A4)
    c.setFont("Helvetica", 18)
    c.drawString(100, 750, "Rotated page edge case")
    c.drawString(100, 720, "This page has /Rotate 90 metadata")
    c.drawString(100, 690, "Text should still be detected")
    c.showPage()
    c.save()

    reader = PdfReader(str(tmp))
    writer = PdfWriter()
    for page in reader.pages:
        page.rotate(90)
        writer.add_page(page)
    with open(p, "wb") as fh:
        writer.write(fh)
    tmp.unlink()
    print(f"  wrote {p.name}")


def gen_password_pdf(force):
    p = OUT / "password_protected.pdf"
    if _skip_if_exists(p, force): return
    tmp = OUT / "_plain.pdf"
    c = canvas.Canvas(str(tmp), pagesize=A4)
    c.setFont("Helvetica", 18)
    c.drawString(100, 750, "This PDF is password protected")
    c.showPage()
    c.save()

    reader = PdfReader(str(tmp))
    writer = PdfWriter()
    for page in reader.pages:
        writer.add_page(page)
    writer.encrypt(user_password="test", owner_password="test")
    with open(p, "wb") as fh:
        writer.write(fh)
    tmp.unlink()
    print(f"  wrote {p.name}")


def gen_no_text_layer_pdf(force):
    """Image-only PDF (rasterized) — pure image on page, no text objects."""
    p = OUT / "no_text_layer.pdf"
    if _skip_if_exists(p, force): return
    img_path = OUT / "_image.png"
    img = Image.new("RGB", (800, 400), "white")
    d = ImageDraw.Draw(img)
    d.text((50, 150), "IMAGE ONLY NO TEXT LAYER", fill="black", font=_font(36))
    img.save(img_path, format="PNG")

    c = canvas.Canvas(str(p), pagesize=A4)
    c.drawImage(str(img_path), 50, 400, width=500, height=250)
    c.showPage()
    c.save()
    img_path.unlink()
    print(f"  wrote {p.name}")


def gen_cropbox_offset_pdf(force):
    """PDF with non-zero CropBox origin — exercises coordinate-translation code."""
    p = OUT / "cropbox_offset.pdf"
    if _skip_if_exists(p, force): return
    tmp = OUT / "_plain2.pdf"
    c = canvas.Canvas(str(tmp), pagesize=(700, 900))
    c.setFont("Helvetica", 18)
    c.drawString(200, 500, "Cropbox offset test")
    c.drawString(200, 470, "Origin is not at (0, 0)")
    c.showPage()
    c.save()

    reader = PdfReader(str(tmp))
    writer = PdfWriter()
    for page in reader.pages:
        page.cropbox = RectangleObject((100, 100, 600, 800))
        writer.add_page(page)
    with open(p, "wb") as fh:
        writer.write(fh)
    tmp.unlink()
    print(f"  wrote {p.name}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--force", action="store_true", help="overwrite existing")
    args = parser.parse_args()

    OUT.mkdir(parents=True, exist_ok=True)
    print(f"Generating edge-case fixtures in {OUT}/")
    gen_zero_byte(args.force)
    gen_one_pixel(args.force)
    gen_4k_text(args.force)
    gen_corrupt_jpeg(args.force)
    gen_garbage_random(args.force)
    gen_single_pixel_bgr(args.force)
    gen_whitespace_pdf(args.force)
    gen_500_page_pdf(args.force)
    gen_rotated_pdf(args.force)
    gen_password_pdf(args.force)
    gen_no_text_layer_pdf(args.force)
    gen_cropbox_offset_pdf(args.force)
    print("done")


if __name__ == "__main__":
    main()
