#!/usr/bin/env python
"""Verify the stage6 partial-month rebuild against full-month collapse.

This verification checks feature reconstruction only. It does not imply the
overall dataset or model metrics should match the old leaky runs.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from ml.lookback_history import verify_month_collapse


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
    parser = argparse.ArgumentParser()
    parser.add_argument("--start", default="2020-06")
    parser.add_argument("--end", default="2025-12")
    parser.add_argument("--look-back-days", type=int, default=31)
    args = parser.parse_args()

    failures = 0
    for month in generate_months(args.start, args.end):
        check = verify_month_collapse(month, look_back_days=args.look_back_days)
        status = "OK" if check.matches else "MISMATCH"
        print(
            f"{status} {month}: partial={check.partial_count} full={check.full_count} "
            f"missing={check.missing_in_partial} extra={check.extra_in_partial}"
        )
        if not check.matches:
            failures += 1

    if failures:
        raise SystemExit(f"{failures} month(s) failed the 31-day collapse check")


if __name__ == "__main__":
    main()
