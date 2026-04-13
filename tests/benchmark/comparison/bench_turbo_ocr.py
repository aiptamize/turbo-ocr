#!/usr/bin/env python3
"""Benchmark Turbo-OCR (C++/TRT) via HTTP on FUNSD — Counter-based bag-of-words F1."""
import concurrent.futures
import io
import json
import os
import re
import time
from collections import Counter

import numpy as np
import requests
from datasets import load_dataset

URL = os.environ.get("TURBO_OCR_URL", "http://localhost:8000")
N = 50
CONCURRENCY = 16
THROUGHPUT_ITERS = 200

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    """Split text into lowercase alphanumeric tokens (matches _scoring.py)."""
    return _TOKEN_RE.findall(text.lower()) if text else []


def word_f1(gt_tokens: list[str], pred_tokens: list[str]) -> dict:
    """Counter-based bag-of-words F1 — duplicates counted correctly."""
    if not gt_tokens and not pred_tokens:
        return {"recall": 1.0, "precision": 1.0, "f1": 1.0}
    if not gt_tokens or not pred_tokens:
        return {"recall": 0.0, "precision": 0.0, "f1": 0.0}
    gt_bag = Counter(gt_tokens)
    pred_bag = Counter(pred_tokens)
    tp = sum((gt_bag & pred_bag).values())
    r = tp / sum(gt_bag.values())
    p = tp / sum(pred_bag.values())
    f1 = 2 * r * p / (r + p) if (r + p) > 0 else 0.0
    return {"recall": r, "precision": p, "f1": f1}


def main():
    print("Loading FUNSD…")
    ds = load_dataset("nielsr/funsd", split="test").select(range(N))

    images_png, ground_truths = [], []
    for sample in ds:
        buf = io.BytesIO()
        sample["image"].convert("RGB").save(buf, format="PNG")
        images_png.append(buf.getvalue())
        ground_truths.append(tokenize(" ".join(sample["words"])))

    def run_one(i: int) -> list[str]:
        r = requests.post(f"{URL}/ocr/raw", data=images_png[i],
                          headers={"Content-Type": "image/png"}, timeout=30)
        return [item["text"] for item in r.json().get("results", [])]

    print("Warmup…")
    run_one(0); run_one(0)

    print(f"Accuracy + latency on {N} images…")
    accs, lats = [], []
    for i in range(N):
        t0 = time.perf_counter()
        preds = run_one(i)
        lat = (time.perf_counter() - t0) * 1000
        lats.append(lat)
        pred_tokens = tokenize(" ".join(preds))
        accs.append(word_f1(ground_truths[i], pred_tokens))
        if (i + 1) % 10 == 0:
            mf1 = np.mean([a["f1"] for a in accs])
            ml = np.mean(lats)
            print(f"  [{i+1}/{N}] F1={mf1:.1%} lat={ml:.0f}ms")

    print(f"Concurrent throughput (c={CONCURRENCY}, {THROUGHPUT_ITERS} requests)…")
    t0 = time.perf_counter()
    with concurrent.futures.ThreadPoolExecutor(max_workers=CONCURRENCY) as pool:
        list(pool.map(lambda k: run_one(k % N), range(THROUGHPUT_ITERS)))
    tp_total = time.perf_counter() - t0
    throughput = THROUGHPUT_ITERS / tp_total

    result = {
        "name": "Turbo-OCR (C++/TRT)",
        "accuracy": accs,
        "latencies_ms": lats,
        "throughput_img_per_sec": throughput,
        "errors": 0,
        "total_images": N,
    }
    f1 = np.mean([a["f1"] for a in accs])
    p = np.mean([a["precision"] for a in accs])
    r = np.mean([a["recall"] for a in accs])
    print(f"\nTurbo-OCR: F1={f1:.1%} P={p:.1%} R={r:.1%} | {throughput:.2f} img/s | p50={np.median(lats):.0f}ms")

    with open("vlm_result_turbo_ocr.json", "w") as f:
        json.dump(result, f, indent=2, default=str)
    print("Saved to vlm_result_turbo_ocr.json")


if __name__ == "__main__":
    main()
