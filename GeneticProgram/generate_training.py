#!/usr/bin/env python3
"""Generate training data for f(x,y) = x * y / 7
Writes 50 samples to training_data.csv in the same directory.

Assumptions:
- x and y are integers in range [0, 100]
- outputs are computed as float x*y/7
- file is overwritten each run
"""

import csv
import random
from pathlib import Path

OUT = Path(__file__).resolve().parent / "training_data.csv"

random.seed(42)  # deterministic outputs; change/remove for different data
N = 50

with OUT.open("w", newline="") as f:
    writer = csv.writer(f)
    writer.writerow(("x", "y", "out"))
    for _ in range(N):
        x = random.randint(0, 100)
        y = random.randint(0, 100)
        out = x * y / 7
        writer.writerow((x, y, out))

print(f"Wrote {N} samples to {OUT}")
