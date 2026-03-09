#!/usr/bin/env python
"""Generate comprehensive comparison tables across all (ptype, class_type, version) slices."""
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
METRICS = ["VC@20", "VC@50", "VC@100", "Recall@20", "Recall@50", "NDCG", "Spearman"]
SLICES = [("f0", "onpeak"), ("f0", "offpeak"), ("f1", "onpeak"), ("f1", "offpeak")]
VERSIONS = [("v0", "Formula"), ("v1", "Opt. Blend"), ("v2", "Full ML")]


def vid_map(ptype, vid):
    if ptype == "f0" and vid == "v2":
        return "v10e-lag1"
    return vid


def load_agg(base_dir, ptype, ctype, vid):
    p = base_dir / ptype / ctype / vid / "metrics.json"
    if not p.exists():
        return None, None
    d = json.load(open(p))
    return d.get("aggregate", {}).get("mean", {}), d.get("n_months", "?")


def print_section(title, base_dir):
    print(f"\n{'=' * 120}")
    print(title)
    print(f"{'=' * 120}")

    for ptype, ctype in SLICES:
        print(f"\n--- {ptype}/{ctype} ---")
        hdr = f"{'Version':<16} {'N':>3}  " + "  ".join(f"{m:>10}" for m in METRICS)
        print(hdr)
        print("-" * len(hdr))

        for vid, label in VERSIONS:
            actual_vid = vid_map(ptype, vid)
            agg, n = load_agg(base_dir, ptype, ctype, actual_vid)
            if agg is None:
                print(f"{label:<16}  --   (not found: {actual_vid})")
                continue
            vals = "  ".join(f"{agg.get(m, 0):>10.4f}" for m in METRICS)
            print(f"{label:<16} {n:>3}  {vals}")

        v0_agg, _ = load_agg(base_dir, ptype, ctype, vid_map(ptype, "v0"))
        v2_agg, _ = load_agg(base_dir, ptype, ctype, vid_map(ptype, "v2"))
        if v0_agg and v2_agg:
            deltas = "  ".join(
                f"{100 * (v2_agg.get(m, 0) / max(v0_agg.get(m, 1), 1e-9) - 1):>+9.1f}%"
                for m in METRICS
            )
            print(f"{'ML vs Formula':<16}      {deltas}")


def print_summary(base_dir, label):
    print(f"\n{'=' * 120}")
    print(f"SUMMARY: ML vs Formula — {label} VC@20")
    print(f"{'=' * 120}")
    print(f"{'Slice':<20} {'v0 VC@20':>10} {'ML VC@20':>10} {'Delta':>10} {'%Change':>10}")
    print("-" * 60)
    for ptype, ctype in SLICES:
        v0, _ = load_agg(base_dir, ptype, ctype, vid_map(ptype, "v0"))
        v2, _ = load_agg(base_dir, ptype, ctype, vid_map(ptype, "v2"))
        if v0 and v2:
            v0_vc = v0.get("VC@20", 0)
            v2_vc = v2.get("VC@20", 0)
            delta = v2_vc - v0_vc
            pct = 100 * (v2_vc / v0_vc - 1) if v0_vc else 0
            print(f"{ptype}/{ctype:<15} {v0_vc:>10.4f} {v2_vc:>10.4f} {delta:>+10.4f} {pct:>+9.1f}%")
        else:
            print(f"{ptype}/{ctype:<15}   (missing data)")


def main():
    reg = ROOT / "registry"
    ho = ROOT / "holdout"

    print_section("DEV COMPARISON (all period_type x class_type x version)", reg)
    print_section("HOLDOUT COMPARISON (all period_type x class_type x version)", ho)
    print_summary(reg, "Dev")
    print_summary(ho, "Holdout")


if __name__ == "__main__":
    main()
