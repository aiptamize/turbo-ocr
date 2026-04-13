"""Shared scoring primitives for ground-truth accuracy tests.

Ground truth is captured-from-OCR baseline, not hand-labelled. See README.md.
"""

from __future__ import annotations

import json
import re
from collections import Counter
from pathlib import Path
from typing import Any

_TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text: str) -> list[str]:
    if not text:
        return []
    return _TOKEN_RE.findall(text.lower())


def word_f1(expected_tokens: list[str], actual_tokens: list[str]) -> dict[str, float]:
    if not expected_tokens and not actual_tokens:
        return {"f1": 1.0, "precision": 1.0, "recall": 1.0}
    if not expected_tokens or not actual_tokens:
        return {"f1": 0.0, "precision": 0.0, "recall": 0.0}
    ec = Counter(expected_tokens)
    ac = Counter(actual_tokens)
    match = sum((ec & ac).values())
    precision = match / max(sum(ac.values()), 1)
    recall = match / max(sum(ec.values()), 1)
    f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) else 0.0
    return {"f1": f1, "precision": precision, "recall": recall}


def levenshtein(a: str, b: str) -> int:
    if a == b:
        return 0
    if len(a) < len(b):
        a, b = b, a
    prev = list(range(len(b) + 1))
    for i, ca in enumerate(a, 1):
        curr = [i] + [0] * len(b)
        for j, cb in enumerate(b, 1):
            curr[j] = min(
                prev[j] + 1,
                curr[j - 1] + 1,
                prev[j - 1] + (0 if ca == cb else 1),
            )
        prev = curr
    return prev[-1]


def cer(expected: str, actual: str) -> float:
    if not expected:
        return 0.0 if not actual else 1.0
    return levenshtein(expected, actual) / max(len(expected), 1)


def _join_results(results: list[dict[str, Any]]) -> str:
    return " ".join(r.get("text", "") for r in results)


def score_image(expected_json: dict, actual_json: dict) -> dict[str, float]:
    exp_text = _join_results(expected_json.get("results", []))
    act_text = _join_results(actual_json.get("results", []))
    tok = word_f1(tokenize(exp_text), tokenize(act_text))
    region_exp = len(expected_json.get("results", []))
    region_act = len(actual_json.get("results", []))
    region_ratio = region_act / region_exp if region_exp else 0.0
    return {
        **tok,
        "cer": cer(exp_text.lower(), act_text.lower()),
        "n_expected": region_exp,
        "n_actual": region_act,
        "region_ratio": region_ratio,
    }


def score_pdf(expected_json: dict, actual_json: dict) -> dict[str, float]:
    # Normalize to 1-based page keys. GT uses {"page": 1, ...}; server
    # may use "page" (1-based) or "page_index" (0-based).
    def _page_key(p, i):
        if "page" in p:
            return p["page"]
        if "page_index" in p:
            return p["page_index"] + 1
        return i + 1

    exp_pages = {_page_key(p, i): p for i, p in enumerate(expected_json.get("pages", []))}
    act_pages = {_page_key(p, i): p for i, p in enumerate(actual_json.get("pages", []))}

    total_exp_tokens = []
    total_act_tokens = []
    per_page = []
    for pg, exp_page in exp_pages.items():
        act_page = act_pages.get(pg, {"results": []})
        exp_t = tokenize(_join_results(exp_page.get("results", [])))
        act_t = tokenize(_join_results(act_page.get("results", [])))
        total_exp_tokens.extend(exp_t)
        total_act_tokens.extend(act_t)
        per_page.append(word_f1(exp_t, act_t))

    agg = word_f1(total_exp_tokens, total_act_tokens)
    agg["n_pages"] = len(exp_pages)
    agg["per_page_f1"] = [p["f1"] for p in per_page]
    return agg


def iou_box(quad_a, quad_b) -> float:
    """IoU of two quadrilaterals by reducing to axis-aligned bboxes."""
    def _bbox(q):
        xs = [p[0] for p in q]
        ys = [p[1] for p in q]
        return min(xs), min(ys), max(xs), max(ys)
    ax0, ay0, ax1, ay1 = _bbox(quad_a)
    bx0, by0, bx1, by1 = _bbox(quad_b)
    ix0, iy0 = max(ax0, bx0), max(ay0, by0)
    ix1, iy1 = min(ax1, bx1), min(ay1, by1)
    iw, ih = max(0, ix1 - ix0), max(0, iy1 - iy0)
    inter = iw * ih
    area_a = max(0, ax1 - ax0) * max(0, ay1 - ay0)
    area_b = max(0, bx1 - bx0) * max(0, by1 - by0)
    union = area_a + area_b - inter
    return inter / union if union > 0 else 0.0


def load_json(path: Path) -> dict:
    return json.loads(Path(path).read_text())


def load_floors(path: Path) -> dict:
    if not Path(path).exists():
        return {}
    return json.loads(Path(path).read_text())
