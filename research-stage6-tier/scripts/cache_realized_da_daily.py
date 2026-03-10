#!/usr/bin/env python
"""Cache daily realized DA net sums for stage6 partial-month features."""
from __future__ import annotations

import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ml.realized_da_daily import fetch_and_cache_daily_month


def generate_months(start: str, end: str) -> list[str]:
    sy, sm = map(int, start.split("-"))
    ey, em = map(int, end.split("-"))
    out = []
    y, m = sy, sm
    while (y, m) <= (ey, em):
        out.append(f"{y:04d}-{m:02d}")
        m += 1
        if m > 12:
            y += 1
            m = 1
    return out


def main() -> None:
    os.environ["RAY_ADDRESS"] = "ray://10.8.0.36:10001"
    from pbase.config.ray import init_ray
    import pmodel
    init_ray(extra_modules=[pmodel])

    for month in generate_months("2019-06", "2025-12"):
        out = fetch_and_cache_daily_month(month)
        print(f"cached {month}: {out}")


if __name__ == "__main__":
    main()
