#!/usr/bin/env python3
"""
Download a subset of MJSynth text recognition dataset from HuggingFace
and convert to PaddleOCR format (image_path\tlabel).

Downloads 30,000 samples (27,000 train + 3,000 val) which is sufficient
for Quantization-Aware Training (QAT) fine-tuning.

Usage:
    cd /home/nataell/code/epAiland/paddle-highspeed-cpp/quantization
    python scripts/download_dataset.py
"""

import os
import sys
from pathlib import Path

# Paths
QUANT_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = QUANT_DIR / "data"
TRAIN_DIR = DATA_DIR / "train"
VAL_DIR = DATA_DIR / "val"
TRAIN_LIST = DATA_DIR / "train_list.txt"
VAL_LIST = DATA_DIR / "val_list.txt"

# How many samples to download
TOTAL_SAMPLES = 30000
TRAIN_RATIO = 0.9
TRAIN_COUNT = int(TOTAL_SAMPLES * TRAIN_RATIO)  # 27000
VAL_COUNT = TOTAL_SAMPLES - TRAIN_COUNT          # 3000

# Load the character set from the model's keys.txt to filter samples
KEYS_PATH = QUANT_DIR.parent / "models" / "v5_latin_rec" / "keys.txt"


def load_valid_chars():
    """Load the character dictionary used by the PP-OCRv5 latin rec model."""
    valid_chars = set()
    if KEYS_PATH.exists():
        with open(KEYS_PATH, "r", encoding="utf-8") as f:
            for line in f:
                ch = line.strip("\n")
                if ch:
                    valid_chars.add(ch)
        print(f"Loaded {len(valid_chars)} valid characters from {KEYS_PATH}")
    else:
        # Fallback: basic Latin chars + digits + punctuation
        import string
        valid_chars = set(string.printable) - {"\t", "\n", "\r", "\x0b", "\x0c"}
        print(f"WARNING: keys.txt not found at {KEYS_PATH}, using basic ASCII fallback")
    return valid_chars


def label_is_valid(label, valid_chars, max_len=25):
    """Check if all characters in label are in the model's dictionary."""
    if not label or len(label) > max_len:
        return False
    return all(c in valid_chars for c in label)


def main():
    try:
        from datasets import load_dataset
    except ImportError:
        print("ERROR: 'datasets' package not installed. Run: pip install datasets Pillow")
        sys.exit(1)

    # Create output directories
    TRAIN_DIR.mkdir(parents=True, exist_ok=True)
    VAL_DIR.mkdir(parents=True, exist_ok=True)

    valid_chars = load_valid_chars()

    print(f"Downloading MJSynth subset from HuggingFace (streaming mode)...")
    print(f"Target: {TRAIN_COUNT} train + {VAL_COUNT} val samples")

    # Use streaming to avoid downloading the full 12GB dataset
    dataset = load_dataset(
        "priyank-m/MJSynth_text_recognition",
        split="train",
        streaming=True,
    )

    train_count = 0
    val_count = 0
    skipped = 0
    total_processed = 0

    train_labels = []
    val_labels = []

    for example in dataset:
        total_processed += 1
        label = example["label"]

        # Filter: only keep samples whose characters are in our dictionary
        if not label_is_valid(label, valid_chars):
            skipped += 1
            continue

        image = example["image"]

        # Decide train vs val
        if train_count < TRAIN_COUNT:
            idx = train_count
            out_dir = TRAIN_DIR
            img_name = f"{idx:06d}.jpg"
            rel_path = img_name
            train_labels.append(f"{rel_path}\t{label}")
            train_count += 1
        elif val_count < VAL_COUNT:
            idx = val_count
            out_dir = VAL_DIR
            img_name = f"{idx:06d}.jpg"
            rel_path = img_name
            val_labels.append(f"{rel_path}\t{label}")
            val_count += 1
        else:
            break

        # Save image
        img_path = out_dir / img_name
        if image.mode != "RGB":
            image = image.convert("RGB")
        image.save(str(img_path), "JPEG", quality=95)

        if (train_count + val_count) % 1000 == 0:
            print(f"  Progress: {train_count} train + {val_count} val "
                  f"({skipped} skipped, {total_processed} processed)")

    # Write label files
    with open(TRAIN_LIST, "w", encoding="utf-8") as f:
        f.write("\n".join(train_labels) + "\n")

    with open(VAL_LIST, "w", encoding="utf-8") as f:
        f.write("\n".join(val_labels) + "\n")

    print(f"\nDone!")
    print(f"  Train: {train_count} images in {TRAIN_DIR}")
    print(f"  Val:   {val_count} images in {VAL_DIR}")
    print(f"  Skipped: {skipped} (chars not in dictionary)")
    print(f"  Total processed from stream: {total_processed}")
    print(f"  Train labels: {TRAIN_LIST}")
    print(f"  Val labels:   {VAL_LIST}")


if __name__ == "__main__":
    main()
