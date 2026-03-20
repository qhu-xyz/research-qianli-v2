"""Step 1: File-level sanity check on all published V7.0 files."""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import polars as pl

CONSTRAINT_ROOT = "/opt/data/xyz-dataset/signal_data/miso/constraints/TEST.Signal.MISO.SPICE_ANNUAL_V7.0.R1"
SF_ROOT = "/opt/data/xyz-dataset/signal_data/miso/sf/TEST.Signal.MISO.SPICE_ANNUAL_V7.0.R1"

PYS = ["2019-06", "2020-06", "2021-06", "2022-06", "2023-06", "2024-06", "2025-06"]
AQS = ["aq1", "aq2", "aq3", "aq4"]
CTYPES = ["onpeak", "offpeak"]

EXPECTED_COLS = [
    "constraint_id", "flow_direction", "mean_branch_max", "mean_branch_max_fillna",
    "ori_mean", "branch_name", "bus_key", "bus_key_group", "mix_mean",
    "shadow_price_da", "density_mix_rank_value", "density_ori_rank_value",
    "da_rank_value", "rank_ori", "density_mix_rank", "rank", "tier",
    "shadow_sign", "shadow_price", "equipment",
]

OUTPUT_DIR = Path("data/v7_verification")


def check_constraints():
    """Check all constraint files."""
    results = []
    missing = []

    print("--- Constraint Files ---")
    header = f"{'Slice':<30} {'Rows':>5} {'Cols':>5} {'Nulls':>6} {'UniqueB':>8} {'DupRows':>7} {'IdxFmt':>8} {'Status':>8}"
    print(header)
    print("-" * 90)

    for py in PYS:
        for aq in AQS:
            for ct in CTYPES:
                path = f"{CONSTRAINT_ROOT}/{py}/{aq}/{ct}/signal.parquet"
                slug = f"{py}/{aq}/{ct}"

                if not Path(path).exists():
                    missing.append(slug)
                    continue

                df = pl.read_parquet(path)
                n_rows = len(df)
                n_cols = len(df.columns)
                has_index = "__index_level_0__" in df.columns
                data_cols = [c for c in df.columns if c != "__index_level_0__"]
                total_nulls = sum(df[c].null_count() for c in df.columns)
                n_unique = df["branch_name"].n_unique() if "branch_name" in df.columns else -1
                n_dup = n_rows - n_unique if n_unique > 0 else -1

                # Index format
                idx_ok = True
                if has_index:
                    samples = df["__index_level_0__"].head(5).to_list()
                    for s in samples:
                        parts = str(s).split("|")
                        if len(parts) != 3 or parts[2] != "spice":
                            idx_ok = False
                            break

                # Schema
                missing_cols = [c for c in EXPECTED_COLS if c not in df.columns]
                extra_cols = [c for c in data_cols if c not in EXPECTED_COLS]

                status = "OK"
                issues = []
                if n_rows != 1000:
                    status = f"ROWS={n_rows}"
                    issues.append(f"expected 1000 rows, got {n_rows}")
                if total_nulls > 0:
                    status = f"NULLS={total_nulls}"
                    issues.append(f"{total_nulls} nulls")
                if missing_cols:
                    issues.append(f"missing cols: {missing_cols}")
                if not idx_ok:
                    issues.append("index format wrong")

                print(f"{slug:<30} {n_rows:>5} {n_cols:>5} {total_nulls:>6} {n_unique:>8} {n_dup:>7} {'OK' if idx_ok else 'BAD':>8} {status:>8}")

                results.append({
                    "slice": slug, "rows": n_rows, "cols": n_cols,
                    "nulls": total_nulls, "unique_branches": n_unique,
                    "dup_rows": n_dup, "idx_ok": idx_ok,
                    "missing_cols": missing_cols, "extra_cols": extra_cols,
                    "issues": issues,
                })

    return results, missing


def check_sf():
    """Check all SF files."""
    results = []

    print()
    print("--- SF Files ---")
    header = f"{'Slice':<30} {'Pnodes':>7} {'SFcols':>7} {'NaNs':>6} {'ZeroCols':>9} {'SFmin':>10} {'SFmax':>10} {'SF|p99|':>10} {'CIDmatch':>9}"
    print(header)
    print("-" * 110)

    for py in PYS:
        for aq in AQS:
            for ct in CTYPES:
                sf_path = f"{SF_ROOT}/{py}/{aq}/{ct}/signal.parquet"
                cstr_path = f"{CONSTRAINT_ROOT}/{py}/{aq}/{ct}/signal.parquet"
                slug = f"{py}/{aq}/{ct}"

                if not Path(sf_path).exists():
                    continue

                sf = pl.read_parquet(sf_path)
                n_pnodes = len(sf)
                sf_cols = [c for c in sf.columns if c != "pnode_id"]
                n_sf_cols = len(sf_cols)

                # NaN check on sampled columns
                sample_cols = sf_cols[:50]
                n_nans = 0
                for c in sample_cols:
                    if sf[c].dtype in [pl.Float64, pl.Float32]:
                        n_nans += sf[c].is_nan().sum()

                # Zero-column check on sampled columns
                zero_cols = 0
                for c in sf_cols[:100]:
                    if sf[c].dtype in [pl.Float64, pl.Float32]:
                        if sf[c].abs().sum() == 0:
                            zero_cols += 1

                # Distribution check on sampled columns
                vals = []
                for c in sample_cols:
                    if sf[c].dtype in [pl.Float64, pl.Float32]:
                        v = sf[c].to_numpy()
                        vals.extend(v[~np.isnan(v)])
                vals = np.array(vals)
                sf_min = float(np.min(vals)) if len(vals) > 0 else 0
                sf_max = float(np.max(vals)) if len(vals) > 0 else 0
                sf_p99_abs = float(np.percentile(np.abs(vals), 99)) if len(vals) > 0 else 0

                # CID match
                cid_match = "N/A"
                if Path(cstr_path).exists():
                    cstr = pl.read_parquet(cstr_path)
                    pub_cids = set(cstr["constraint_id"].to_list())
                    sf_cid_set = set()
                    for c in sf_cols:
                        if "|" in c:
                            sf_cid_set.add(c.split("|")[0])
                    if sf_cid_set == pub_cids:
                        cid_match = "EXACT"
                    elif len(sf_cid_set) == len(pub_cids):
                        cid_match = f"COUNT_OK"
                    else:
                        cid_match = f"MISMATCH({len(sf_cid_set)}vs{len(pub_cids)})"

                print(
                    f"{slug:<30} {n_pnodes:>7} {n_sf_cols:>7} {n_nans:>6} "
                    f"{zero_cols:>9} {sf_min:>10.4f} {sf_max:>10.4f} "
                    f"{sf_p99_abs:>10.4f} {cid_match:>9}"
                )

                results.append({
                    "slice": slug, "pnodes": n_pnodes, "sf_cols": n_sf_cols,
                    "nans_sampled": n_nans, "zero_cols_sampled": zero_cols,
                    "sf_min": sf_min, "sf_max": sf_max, "sf_p99_abs": sf_p99_abs,
                    "cid_match": cid_match,
                })

    return results


def main():
    print("=" * 120)
    print("STEP 1: File-Level Sanity — All Published V7.0 Files")
    print("=" * 120)
    print()

    cstr_results, missing = check_constraints()
    sf_results = check_sf()

    # Missing slices
    print()
    print("--- Missing Published Slices ---")
    for ms in missing:
        print(f"  MISSING: {ms}")

    # Summary
    print()
    ub = [r["unique_branches"] for r in cstr_results]
    dup = [r["dup_rows"] for r in cstr_results]
    all_issues = [iss for r in cstr_results for iss in r["issues"]]

    print("CONSTRAINT SUMMARY:")
    print(f"  Files: {len(cstr_results)}/56 expected ({len(missing)} missing)")
    print(f"  All 1000 rows: {all(r['rows'] == 1000 for r in cstr_results)}")
    print(f"  All zero nulls: {all(r['nulls'] == 0 for r in cstr_results)}")
    print(f"  All index format OK: {all(r['idx_ok'] for r in cstr_results)}")
    print(f"  Unique branches: min={min(ub)}, max={max(ub)}, mean={np.mean(ub):.0f}")
    print(f"  Duplicate rows: min={min(dup)}, max={max(dup)}, mean={np.mean(dup):.0f}")

    # SF summary
    pnodes = [r["pnodes"] for r in sf_results]
    sf_mins = [r["sf_min"] for r in sf_results]
    sf_maxs = [r["sf_max"] for r in sf_results]

    print()
    print("SF SUMMARY:")
    print(f"  Files: {len(sf_results)}/56 expected")
    print(f"  Pnodes: min={min(pnodes)}, max={max(pnodes)}")
    print(f"  SF value range: [{min(sf_mins):.4f}, {max(sf_maxs):.4f}]")
    print(f"  Any NaNs (sampled): {any(r['nans_sampled'] > 0 for r in sf_results)}")
    print(f"  Any zero columns (sampled): {any(r['zero_cols_sampled'] > 0 for r in sf_results)}")
    print(f"  All CID match: {all(r['cid_match'] in ('EXACT', 'COUNT_OK') for r in sf_results)}")

    if all_issues:
        print()
        print("ISSUES:")
        for iss in all_issues:
            print(f"  - {iss}")
    else:
        print()
        print("NO ISSUES FOUND")

    # Save
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    with open(OUTPUT_DIR / "step1_sanity.json", "w") as f:
        json.dump({
            "constraint_results": cstr_results,
            "sf_results": sf_results,
            "missing_slices": missing,
        }, f, indent=2, default=str)
    print(f"\nSaved: {OUTPUT_DIR / 'step1_sanity.json'}")


if __name__ == "__main__":
    main()
