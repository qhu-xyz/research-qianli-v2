"""Quick test: 5-bin experiments (with/without bidir correction) across all rounds.

Imports run_experiment from run_v7_bands.py. Prints comparison table only — no files written.

Usage:
    cd /home/xyz/workspace/pmodel && source .venv/bin/activate
    python /home/xyz/workspace/research-qianli-v2/research-annual/scripts/test_5bins.py
"""

from __future__ import annotations

import gc
import json
import resource
import sys
from pathlib import Path

import polars as pl

# Import everything from v7
sys.path.insert(0, str(Path(__file__).resolve().parent))
from run_v7_bands import (
    ROOT,
    R1_DATA_DIR,
    R2R3_DATA_PATH,
    QUARTERS,
    R1_PYS,
    R2R3_PYS,
    MCP_COL,
    PY_COL,
    CLASS_COL,
    CLASSES,
    SIGN_SEGS,
    COVERAGE_LEVELS,
    COVERAGE_LABELS,
    BG1_TOLERANCE,
    add_sign_seg,
    run_experiment,
    mem_mb,
)


def run_5bin_test():
    experiments = [
        {"name": "asym_5b",       "n_bins": 5, "correction": False},
        {"name": "asym_5b_bidir", "n_bins": 5, "correction": True},
    ]

    all_round_results = {}

    # ─── R1 ──────────────────────────────────────────────────────────
    print(f"\n{'#'*80}")
    print(f"  R1 — 5-bin test (temporal CV)")
    print(f"{'#'*80}")

    r1_results = {e["name"]: {} for e in experiments}
    for quarter in QUARTERS:
        parquet_path = R1_DATA_DIR / f"{quarter}_all_baselines.parquet"
        df = (
            pl.scan_parquet(parquet_path)
            .filter(
                (pl.col(PY_COL) >= 2019)
                & pl.col("nodal_f0").is_not_null()
                & pl.col(MCP_COL).is_not_null()
            )
            .collect()
        )
        df = add_sign_seg(df, "nodal_f0")
        available_pys = sorted(df[PY_COL].unique().to_list())
        pys_to_use = [py for py in R1_PYS if py in available_pys]
        print(f"\n  R1 {quarter.upper()}: {df.height:,} rows, PYs={pys_to_use}")

        for exp in experiments:
            result = run_experiment(
                df, quarter, pys_to_use, exp["n_bins"], "nodal_f0",
                exp["name"], cv_mode="temporal",
                min_train_pys=1, apply_correction=exp.get("correction", False),
            )
            p95_cov = result["aggregate"]["coverage"]["overall"].get("p95", {}).get("actual", 0)
            p95_w = result["aggregate"]["widths"]["overall"].get("p95", {}).get("mean_width", 0)
            print(f"    {exp['name']}: P95 cov={p95_cov:.2f}%, width={p95_w:.1f}")
            r1_results[exp["name"]][quarter] = result

        del df
        gc.collect()

    all_round_results["r1"] = r1_results

    # ─── R2 ──────────────────────────────────────────────────────────
    print(f"\n{'#'*80}")
    print(f"  R2 — 5-bin test (temporal CV)")
    print(f"{'#'*80}")

    r2_results = {e["name"]: {} for e in experiments}
    for quarter in QUARTERS:
        df = (
            pl.scan_parquet(R2R3_DATA_PATH)
            .filter(
                (pl.col("round") == 2)
                & (pl.col("period_type") == quarter)
                & (pl.col(PY_COL) >= 2019)
                & pl.col("mtm_1st_mean").is_not_null()
                & pl.col(MCP_COL).is_not_null()
            )
            .select(["mtm_1st_mean", MCP_COL, PY_COL, "period_type", CLASS_COL, "source_id", "sink_id"])
            .collect()
        )
        df = add_sign_seg(df, "mtm_1st_mean")
        available_pys = sorted(df[PY_COL].unique().to_list())
        pys_to_use = [py for py in R2R3_PYS if py in available_pys]
        print(f"\n  R2 {quarter.upper()}: {df.height:,} rows, PYs={pys_to_use}")

        for exp in experiments:
            result = run_experiment(
                df, quarter, pys_to_use, exp["n_bins"], "mtm_1st_mean",
                exp["name"], cv_mode="temporal",
                min_train_pys=1, apply_correction=exp.get("correction", False),
            )
            p95_cov = result["aggregate"]["coverage"]["overall"].get("p95", {}).get("actual", 0)
            p95_w = result["aggregate"]["widths"]["overall"].get("p95", {}).get("mean_width", 0)
            print(f"    {exp['name']}: P95 cov={p95_cov:.2f}%, width={p95_w:.1f}")
            r2_results[exp["name"]][quarter] = result

        del df
        gc.collect()

    all_round_results["r2"] = r2_results

    # ─── R3 ──────────────────────────────────────────────────────────
    print(f"\n{'#'*80}")
    print(f"  R3 — 5-bin test (temporal CV)")
    print(f"{'#'*80}")

    r3_results = {e["name"]: {} for e in experiments}
    for quarter in QUARTERS:
        df = (
            pl.scan_parquet(R2R3_DATA_PATH)
            .filter(
                (pl.col("round") == 3)
                & (pl.col("period_type") == quarter)
                & (pl.col(PY_COL) >= 2019)
                & pl.col("mtm_1st_mean").is_not_null()
                & pl.col(MCP_COL).is_not_null()
            )
            .select(["mtm_1st_mean", MCP_COL, PY_COL, "period_type", CLASS_COL, "source_id", "sink_id"])
            .collect()
        )
        df = add_sign_seg(df, "mtm_1st_mean")
        available_pys = sorted(df[PY_COL].unique().to_list())
        pys_to_use = [py for py in R2R3_PYS if py in available_pys]
        print(f"\n  R3 {quarter.upper()}: {df.height:,} rows, PYs={pys_to_use}")

        for exp in experiments:
            result = run_experiment(
                df, quarter, pys_to_use, exp["n_bins"], "mtm_1st_mean",
                exp["name"], cv_mode="temporal",
                min_train_pys=1, apply_correction=exp.get("correction", False),
            )
            p95_cov = result["aggregate"]["coverage"]["overall"].get("p95", {}).get("actual", 0)
            p95_w = result["aggregate"]["widths"]["overall"].get("p95", {}).get("mean_width", 0)
            print(f"    {exp['name']}: P95 cov={p95_cov:.2f}%, width={p95_w:.1f}")
            r3_results[exp["name"]][quarter] = result

        del df
        gc.collect()

    all_round_results["r3"] = r3_results

    # ─── Summary table ───────────────────────────────────────────────
    print(f"\n{'='*120}")
    print(f"  5-BIN RESULTS SUMMARY")
    print(f"{'='*120}")

    # Load reference data for comparison
    refs = {}
    for rnd in ["r1", "r2", "r3"]:
        with open(ROOT / f"versions/bands/v7/{rnd}/metrics.json") as f:
            m7 = json.load(f)
        with open(ROOT / f"versions/bands/v3/{rnd}/metrics.json") as f:
            m3 = json.load(f)
        refs[rnd] = {"v7": m7, "v3": m3}

    for rnd in ["r1", "r2", "r3"]:
        print(f"\n  --- {rnd.upper()} ---")
        print(f"  {'Experiment':<18} {'AQ1 Cov':>8} {'AQ2 Cov':>8} {'AQ3 Cov':>8} {'AQ4 Cov':>8} {'AVG Cov':>8} | {'AQ1 Wid':>8} {'AQ2 Wid':>8} {'AQ3 Wid':>8} {'AQ4 Wid':>8} {'AVG Wid':>8} {'vs v3':>7}")

        rnd_results = all_round_results[rnd]
        v3_widths = [refs[rnd]["v3"]["widths"][aq]["overall"]["p95"]["mean_width"] for aq in QUARTERS]
        v3_avg_w = sum(v3_widths) / 4

        for exp in experiments:
            name = exp["name"]
            covs = []
            wids = []
            for aq in QUARTERS:
                r = rnd_results[name][aq]
                c = r["aggregate"]["coverage"]["overall"].get("p95", {}).get("actual", 0)
                w = r["aggregate"]["widths"]["overall"].get("p95", {}).get("mean_width", 0)
                covs.append(c)
                wids.append(w)
            avg_c = sum(covs) / 4
            avg_w = sum(wids) / 4
            vs_v3 = (avg_w - v3_avg_w) / v3_avg_w * 100
            print(f"  {name:<18} {covs[0]:>7.2f}% {covs[1]:>7.2f}% {covs[2]:>7.2f}% {covs[3]:>7.2f}% {avg_c:>7.2f}% | {wids[0]:>8.1f} {wids[1]:>8.1f} {wids[2]:>8.1f} {wids[3]:>8.1f} {avg_w:>8.1f} {vs_v3:>+6.1f}%")

        # Add v7 experiment_comparison reference rows
        ec7 = refs[rnd]["v7"]["experiment_comparison"]
        for ref_exp in ["asym_6b", "asym_6b_bidir", "asym_8b", "asym_8b_bidir"]:
            covs = [ec7[aq][ref_exp]["p95_coverage"] for aq in QUARTERS]
            wids = [ec7[aq][ref_exp]["p95_mean_width"] for aq in QUARTERS]
            avg_c = sum(covs) / 4
            avg_w = sum(wids) / 4
            vs_v3 = (avg_w - v3_avg_w) / v3_avg_w * 100
            print(f"  {ref_exp+' (ref)':<18} {covs[0]:>7.2f}% {covs[1]:>7.2f}% {covs[2]:>7.2f}% {covs[3]:>7.2f}% {avg_c:>7.2f}% | {wids[0]:>8.1f} {wids[1]:>8.1f} {wids[2]:>8.1f} {wids[3]:>8.1f} {avg_w:>8.1f} {vs_v3:>+6.1f}%")

        # v3 reference
        v3c = [refs[rnd]["v3"]["coverage"][aq]["overall"]["p95"]["actual"] for aq in QUARTERS]
        print(f"  {'v3 (promoted)':<18} {v3c[0]:>7.2f}% {v3c[1]:>7.2f}% {v3c[2]:>7.2f}% {v3c[3]:>7.2f}% {sum(v3c)/4:>7.2f}% | {v3_widths[0]:>8.1f} {v3_widths[1]:>8.1f} {v3_widths[2]:>8.1f} {v3_widths[3]:>8.1f} {v3_avg_w:>8.1f}   base")

    print(f"\nDone. Memory: {mem_mb():.0f} MB")


if __name__ == "__main__":
    run_5bin_test()
