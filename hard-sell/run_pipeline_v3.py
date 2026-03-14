"""
MISO Sell April 2026 (auc2604) V3 - Pipeline Runner

V3 changes from V1:
- f0 onpeak/offpeak: constraint-based picking for ALDRICH-FIFTHST, target SP=3000
  - Picked trades: MTM adjusted, plain counter params, 1 file per slice
  - Normal trades: excluded from picked, 5 strategies × 2 flows
- f1: no constraints, same as V1

Output dir: /opt/temp/qianli/miso/apr_v3/
Expected: 42 files (40 normal + 2 picked)
"""

import glob
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

import numpy as np
import pandas as pd
from miso_sell import generate_sell_files

# =============================================================================
# Data Paths (auc2604, prod1)
# =============================================================================

DATA_PATHS = {
    "f0": {
        "onpeak": "/opt/temp/shiyi/trash/sell_hard_miso/miso/f0_prod1_auc2604/trades_onpeak_f0_auc2604.parquet",
        "offpeak": "/opt/temp/shiyi/trash/sell_hard_miso/miso/f0_prod1_auc2604/trades_offpeak_f0_auc2604.parquet",
    },
    "f1": {
        "onpeak": "/opt/temp/shiyi/trash/sell_hard_miso/miso/f1_prod1_auc2604/trades_onpeak_f1_auc2604.parquet",
        "offpeak": "/opt/temp/shiyi/trash/sell_hard_miso/miso/f1_prod1_auc2604/trades_offpeak_f1_auc2604.parquet",
    },
}

# =============================================================================
# Constraint Config (f0 only)
# =============================================================================

CONSTRAINT_DICT = {
    "f0": {
        "onpeak": {
            "443664:ALDRICH-FIFTHST FLO COON CREEK TR9:NSP34X04|-1|auc": 3000,
        },
        "offpeak": {
            "443664:ALDRICH-FIFTHST FLO COON CREEK TR9:NSP34X04|-1|auc": 3000,
        },
    },
    # f1: no constraints
}

# =============================================================================
# Pipeline Configuration
# =============================================================================

OUTPUT_DIR = "/opt/temp/qianli/miso/apr_v3"
AUCTION = "auc2604"
VERSION = "v3"
AUCTION_ROUND = 1


# =============================================================================
# Validation
# =============================================================================


def validate_results(output_dir, auction, version, period_type, class_type):
    """Validate bid point monotonicity."""
    pattern = f"{output_dir}/trades_to_sell_miso_{auction}{version}_*_{period_type}_{class_type}_*.parquet"
    files = sorted(glob.glob(pattern))
    all_ok = True

    for f in files:
        try:
            df = pd.read_parquet(f)
            fname = Path(f).name

            if len(df) == 0:
                print(f"  WARNING: {fname} is empty")
                continue

            price_cols = [c for c in df.columns if c.startswith("bid_price_") and df[c].notna().any()]
            num_bids = len(price_cols)

            for i in range(num_bids - 1):
                col_a = price_cols[i]
                col_b = price_cols[i + 1]
                if not (df[col_a] <= df[col_b]).all():
                    print(f"  WARNING: Price monotonicity violated in {fname} ({col_a} > {col_b})")
                    all_ok = False
                    break

            vol_cols = [c for c in df.columns if c.startswith("bid_volume_") and df[c].notna().any()]
            if vol_cols:
                if not (df[vol_cols[0]] == 0).all():
                    print(f"  WARNING: {fname}: bid_volume_1 != 0")
                    all_ok = False
                if "bid_volume" in df.columns:
                    last_vol = vol_cols[-1]
                    if not np.allclose(df[last_vol], df["bid_volume"], rtol=1e-6):
                        print(f"  WARNING: {fname}: {last_vol} != bid_volume")
                        all_ok = False

            if "mtm_1st_mean" in df.columns:
                has_mtm = df["mtm_1st_mean"].notna()
                for col in price_cols:
                    if df.loc[has_mtm, col].isna().any():
                        nan_count = df.loc[has_mtm, col].isna().sum()
                        print(f"  WARNING: {fname}: {nan_count} NaN prices in {col}")
                        all_ok = False

        except Exception as e:
            print(f"  ERROR reading {Path(f).name}: {e}")
            all_ok = False

    return all_ok


# =============================================================================
# Pipeline
# =============================================================================


def run_pipeline(
    period_types=("f0", "f1"),
    class_types=("onpeak", "offpeak"),
    verbose=True,
):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    results = {}

    for period_type in period_types:
        for class_type in class_types:
            if period_type not in DATA_PATHS or class_type not in DATA_PATHS[period_type]:
                continue

            data_path = DATA_PATHS[period_type][class_type]
            constraint_dict = CONSTRAINT_DICT.get(period_type, {}).get(class_type, {})

            try:
                result = generate_sell_files(
                    data_path=data_path,
                    class_type=class_type,
                    period_type=period_type,
                    output_dir=OUTPUT_DIR,
                    auction=AUCTION,
                    version=VERSION,
                    auction_round=AUCTION_ROUND,
                    constraint_dict=constraint_dict if constraint_dict else None,
                    picked_strategy="plain",
                    verbose=verbose,
                )
                results[(period_type, class_type)] = result

                ok = validate_results(OUTPUT_DIR, AUCTION, VERSION, period_type, class_type)
                if ok:
                    print(f"[PASS] {period_type} {class_type} validation passed")
                else:
                    print(f"[FAIL] {period_type} {class_type} validation failed")

            except Exception as e:
                print(f"[ERROR] {period_type} {class_type}: {e}")
                import traceback

                traceback.print_exc()
                results[(period_type, class_type)] = {"error": str(e)}

    return results


def print_summary(results):
    print(f"\n{'=' * 60}")
    print("SUMMARY - MISO Sell Apr 2026 V3 (auc2604)")
    print(f"{'=' * 60}")

    total_original = 0
    total_normal = 0
    total_picked = 0

    for (period_type, class_type), result in sorted(results.items()):
        if "error" in result:
            print(f"{period_type} {class_type}: ERROR - {result['error']}")
        else:
            orig = result["original_count"]
            norm = result["normal_count"]
            pick = result["picked_count"]
            pct = (pick / orig * 100) if orig > 0 else 0
            print(f"{period_type} {class_type}: {orig:4d} total = {norm:4d} normal + {pick:3d} picked ({pct:.1f}%)")
            total_original += orig
            total_normal += norm
            total_picked += pick

    print(f"{'-' * 60}")
    pct = (total_picked / total_original * 100) if total_original > 0 else 0
    print(f"TOTAL:       {total_original:4d} total = {total_normal:4d} normal + {total_picked:3d} picked ({pct:.1f}%)")

    print(f"\n{'=' * 60}")
    print("ALL FILES GENERATED:")
    print(f"{'=' * 60}")

    files = sorted(glob.glob(f"{OUTPUT_DIR}/trades_to_sell_miso_{AUCTION}{VERSION}_*.parquet"))
    for f in files:
        df = pd.read_parquet(f)
        price_cols = [c for c in df.columns if c.startswith("bid_price_") and df[c].notna().any()]
        print(f"  {Path(f).name}: {len(df)} trades, {len(price_cols)} bid points")

    print(f"\nTotal files: {len(files)}")
    print("Expected: 42 (40 normal + 2 picked)")


if __name__ == "__main__":
    import pbase
    import ray

    import pmodel

    runtime_env = {
        "py_modules": [pbase, pmodel],
        "pip": ["lightgbm"],
        "excludes": ["*.ipynb", ".git/*", "*.html"],
    }
    ray.init(address="ray://10.8.0.36:10001", runtime_env=runtime_env)

    results = run_pipeline()
    print_summary(results)
