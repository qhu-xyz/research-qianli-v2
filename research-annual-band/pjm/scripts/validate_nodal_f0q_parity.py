"""Validate nodal_f0_q parity: live recompute vs research nodal_f0 * 3.

This is the go/no-go gate for MISO R1 annual port.
Pass/fail thresholds (from miso-annual-port-plan.md):
  - median absolute difference ≤ 1.0 quarterly $
  - 95th percentile absolute difference ≤ 50.0 quarterly $

Computed on PY2020+ overlap set (research coverage >99%).

Usage:
    cd /home/xyz/workspace/pmodel && source .venv/bin/activate
    python /home/xyz/workspace/research-qianli-v2/research-annual-band/pjm/scripts/validate_nodal_f0q_parity.py
"""

from __future__ import annotations

import os
import sys
import time

import numpy as np
import pandas as pd
import polars as pl

os.environ["RAY_ADDRESS"] = "ray://10.8.0.36:10001"

RESEARCH_BASELINES = "/opt/temp/qianli/annual_research/crossproduct_work"
MIN_PY = 2020  # PY2020+ for >99% coverage
PASS_MEDIAN_ABS = 1.0  # quarterly $
PASS_P95_ABS = 50.0  # quarterly $

QUARTER_DELIVERY = {
    "aq1": lambda py: [f"{py}-06", f"{py}-07", f"{py}-08"],
    "aq2": lambda py: [f"{py}-09", f"{py}-10", f"{py}-11"],
    "aq3": lambda py: [f"{py}-12", f"{py + 1}-01", f"{py + 1}-02"],
    "aq4": lambda py: [f"{py + 1}-03", f"{py + 1}-04", f"{py + 1}-05"],
}


def main():
    from pbase.config.ray import init_ray
    import pmodel

    init_ray(extra_modules=[pmodel])

    from pbase.data.m2m.calculator import MisoCalculator

    calc = MisoCalculator()

    all_results = []

    for qt in ["aq1", "aq2", "aq3", "aq4"]:
        # Load research nodal_f0
        f = f"{RESEARCH_BASELINES}/{qt}_all_baselines.parquet"
        research = pl.read_parquet(f).filter(
            pl.col("planning_year") >= MIN_PY,
            pl.col("nodal_f0").is_not_null(),
        ).select(["source_id", "sink_id", "class_type", "planning_year", "nodal_f0"])
        research = research.with_columns((pl.col("nodal_f0") * 3).alias("research_f0q"))

        pys = sorted(research["planning_year"].unique().to_list())
        print(f"\n{qt}: {len(research):,} research paths, PYs={pys}")

        for py in pys:
            months = QUARTER_DELIVERY[qt](int(py))
            print(f"  PY{py}: loading f0 MCPs for {months}...", end=" ", flush=True)

            # Recompute nodal_f0_q from MisoCalculator
            node_sums: dict[tuple[str, str], float] = {}
            node_counts: dict[tuple[str, str], int] = {}
            t0 = time.time()

            for m in months:
                try:
                    mcp_df, _ = calc.get_mcp_df(market_month=m, fillna=True)
                except Exception as e:
                    print(f"SKIP ({e})")
                    continue
                if mcp_df is None or mcp_df.empty:
                    print(f"SKIP (empty for {m})")
                    continue
                latest = mcp_df[0]
                for (pnode_id, class_type), val in latest.items():
                    key = (str(pnode_id), str(class_type))
                    if pd.notna(val):
                        node_sums[key] = node_sums.get(key, 0.0) + float(val)
                        node_counts[key] = node_counts.get(key, 0) + 1

            dt = time.time() - t0

            # Only keep nodes with all 3 months
            recomputed = {
                k: v for k, v in node_sums.items() if node_counts.get(k, 0) == 3
            }
            print(f"{len(recomputed):,} nodes ({dt:.1f}s)")

            if not recomputed:
                print(f"    WARNING: no complete 3-month nodes for PY{py} {qt}")
                continue

            # Build recomputed DataFrame
            recomp_rows = []
            for (pnode_id, cls), f0q in recomputed.items():
                recomp_rows.append({"pnode_id": pnode_id, "class_type": cls, "recomp_f0q": f0q})
            recomp_df = pl.DataFrame(recomp_rows)

            # Join research paths with recomputed source/sink f0
            py_research = research.filter(pl.col("planning_year") == py)

            # Source join
            src = recomp_df.rename({"pnode_id": "source_id", "recomp_f0q": "src_recomp"})
            snk = recomp_df.rename({"pnode_id": "sink_id", "recomp_f0q": "snk_recomp"})

            merged = (
                py_research
                .with_columns(pl.col("source_id").cast(pl.Utf8), pl.col("sink_id").cast(pl.Utf8))
                .join(src.with_columns(pl.col("source_id").cast(pl.Utf8)), on=["source_id", "class_type"], how="inner")
                .join(snk.with_columns(pl.col("sink_id").cast(pl.Utf8)), on=["sink_id", "class_type"], how="inner")
            )
            merged = merged.with_columns(
                (pl.col("snk_recomp") - pl.col("src_recomp")).alias("recomp_f0q")
            )

            if len(merged) == 0:
                print(f"    WARNING: no overlap for PY{py} {qt}")
                continue

            diff = (merged["recomp_f0q"] - merged["research_f0q"]).to_numpy()
            abs_diff = np.abs(diff)
            med = float(np.median(abs_diff))
            p95 = float(np.percentile(abs_diff, 95))
            mean_d = float(np.mean(diff))
            coverage = len(merged) / len(py_research) * 100

            pass_med = "PASS" if med <= PASS_MEDIAN_ABS else "FAIL"
            pass_p95 = "PASS" if p95 <= PASS_P95_ABS else "FAIL"

            print(f"    overlap={len(merged):,}/{len(py_research):,} ({coverage:.1f}%)")
            print(f"    median|diff|={med:.4f} [{pass_med}]  p95|diff|={p95:.4f} [{pass_p95}]  mean(diff)={mean_d:.4f}")

            all_results.append({
                "quarter": qt, "py": py, "n_overlap": len(merged),
                "n_research": len(py_research), "coverage_pct": coverage,
                "median_abs_diff": med, "p95_abs_diff": p95, "mean_diff": mean_d,
                "pass_median": med <= PASS_MEDIAN_ABS, "pass_p95": p95 <= PASS_P95_ABS,
            })

    # Summary
    print("\n" + "=" * 80)
    print("PARITY GATE SUMMARY")
    print("=" * 80)
    if not all_results:
        print("NO RESULTS — cannot determine parity")
        sys.exit(1)

    results_df = pd.DataFrame(all_results)
    all_pass = results_df["pass_median"].all() and results_df["pass_p95"].all()

    print(f"\nOverall: {'PASS' if all_pass else 'FAIL'}")
    print(f"  Median |diff| threshold: {PASS_MEDIAN_ABS} quarterly $")
    print(f"  P95 |diff| threshold: {PASS_P95_ABS} quarterly $")
    print(f"\n{results_df.to_string(index=False)}")

    if not all_pass:
        failed = results_df[~(results_df["pass_median"] & results_df["pass_p95"])]
        print(f"\nFAILED cells:\n{failed.to_string(index=False)}")
        print("\nnodal_f0_q recompute does NOT match research. MISO R1 weights are NOT valid.")
        sys.exit(1)
    else:
        print("\nnodal_f0_q recompute matches research within thresholds.")
        print("MISO R1 frozen weights are valid for aq1-aq3.")

        # Compute P1/P99 winsor thresholds from the full research set
        print("\n--- Winsor thresholds (P1/P99 from research training data) ---")
        for qt in ["aq1", "aq2", "aq3", "aq4"]:
            f = f"{RESEARCH_BASELINES}/{qt}_all_baselines.parquet"
            r = pl.read_parquet(f).filter(pl.col("nodal_f0").is_not_null())
            f0q = (r["nodal_f0"] * 3).to_numpy()
            p1, p99 = float(np.percentile(f0q, 1)), float(np.percentile(f0q, 99))
            print(f"  {qt} nodal_f0_q: P1={p1:.1f}, P99={p99:.1f}")

        # Also compute for 1_rev
        rev = pl.read_parquet(
            "/home/xyz/workspace/research-qianli-v2/research-annual-band/miso/data/r1_1rev_option_b.parquet"
        )
        rev_vals = rev["1_rev"].to_numpy()
        p1r, p99r = float(np.percentile(rev_vals, 1)), float(np.percentile(rev_vals, 99))
        print(f"  1_rev (all quarters): P1={p1r:.1f}, P99={p99r:.1f}")


if __name__ == "__main__":
    main()
