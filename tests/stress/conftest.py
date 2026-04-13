"""Stress suite fixtures. Adds the benchmark dir to sys.path so we can
import the harness primitives.
"""

import sys
from pathlib import Path

HERE = Path(__file__).resolve().parent
BENCH_DIR = HERE.parent / "benchmark"
for d in (HERE, BENCH_DIR):
    if str(d) not in sys.path:
        sys.path.insert(0, str(d))
