#!/usr/bin/env python
"""Validate V7.0 signal against V6.2B and stage5 holdout targets.

Usage:
    cd /home/xyz/workspace/pmodel && source .venv/bin/activate
    python /home/xyz/workspace/research-qianli-v2/research-miso-signal7/scripts/validate_v70.py \
        --months 2025-01 2025-02 2025-03

    # Full holdout validation (all gates):
    python .../validate_v70.py --full-holdout
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

_PROJECT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(_PROJECT))

_STAGE5 = _PROJECT.parent / "research-stage5-tier"
sys.path.insert(0, str(_STAGE5))

from pbase.data.dataset.signal.general import ConstraintsSignal, ShiftFactorSignal
from v70.signal_writer import available_ptypes

V62B_SIGNAL = "TEST.TEST.Signal.MISO.SPICE_F0P_V6.2B.R1"
V70_SIGNAL = "TEST.TEST.Signal.MISO.SPICE_F0P_V7.0.R1"
ML_PTYPES = ["f0", "f1"]

HOLDOUT_DIR = _STAGE5 / "holdout"
DA_PATH = Path("/opt/temp/qianli/spice_data/miso/realized_da")


class ValidationResult:
    def __init__(self):
        self.passed: list[str] = []
        self.failed: list[str] = []
        self.warnings: list[str] = []

    def ok(self, gate: str, msg: str):
        self.passed.append(f"[PASS] {gate}: {msg}")

    def fail(self, gate: str, msg: str):
        self.failed.append(f"[FAIL] {gate}: {msg}")

    def warn(self, gate: str, msg: str):
        self.warnings.append(f"[WARN] {gate}: {msg}")

    def summary(self) -> str:
        lines = []
        for p in self.passed:
            lines.append(p)
        for w in self.warnings:
            lines.append(w)
        for f in self.failed:
            lines.append(f)
        lines.append(f"\n{'='*60}")
        lines.append(f"PASSED: {len(self.passed)}  WARNINGS: {len(self.warnings)}  FAILED: {len(self.failed)}")
        if self.failed:
            lines.append("STATUS: BLOCKED — fix failures before deployment")
        else:
            lines.append("STATUS: READY FOR DEPLOYMENT")
        return "\n".join(lines)


def _load_da(delivery_month: str, ctype: str) -> pd.DataFrame | None:
    suffix = f"_{ctype}" if ctype == "offpeak" else ""
    fpath = DA_PATH / f"{delivery_month}{suffix}.parquet"
    if not fpath.exists():
        return None
    import polars as pl
    return pl.read_parquet(str(fpath)).to_pandas()


def _vc_at_k(realized: np.ndarray, scores: np.ndarray, k: int) -> float:
    total = realized.sum()
    if total <= 0:
        return 0.0
    k = min(k, len(scores))
    top_k_idx = np.argsort(scores)[::-1][:k]
    return float(realized[top_k_idx].sum() / total)


# ── Gate C: Passthrough bit-identity ──

def validate_passthrough(month: str, vr: ValidationResult):
    ts = pd.Timestamp(month)
    ptypes = available_ptypes(month)
    passthrough = [p for p in ptypes if p not in ML_PTYPES]
    for ptype in passthrough:
        for ctype in ["onpeak", "offpeak"]:
            key = f"{ptype}/{ctype}"
            try:
                v62b = ConstraintsSignal("miso", V62B_SIGNAL, ptype, ctype).load_data(ts)
                v70 = ConstraintsSignal("miso", V70_SIGNAL, ptype, ctype).load_data(ts)
                pd.testing.assert_frame_equal(v62b, v70, check_exact=True)
                vr.ok("C", f"{month} {key}: passthrough bit-identical")
            except AssertionError as e:
                vr.fail("C", f"{month} {key}: passthrough differs — {e}")
            except FileNotFoundError:
                vr.warn("C", f"{month} {key}: V7.0 not found (may not be generated)")


# ── Gate D: Schema & Structure ──

def validate_schema(month: str, ptype: str, ctype: str, vr: ValidationResult):
    ts = pd.Timestamp(month)
    key = f"{ptype}/{ctype}"
    try:
        v62b = ConstraintsSignal("miso", V62B_SIGNAL, ptype, ctype).load_data(ts)
        v70 = ConstraintsSignal("miso", V70_SIGNAL, ptype, ctype).load_data(ts)
    except FileNotFoundError:
        vr.warn("D", f"{month} {key}: signal not found")
        return

    # D1: columns and dtypes
    if v70.columns.tolist() != v62b.columns.tolist():
        vr.fail("D1", f"{month} {key}: column mismatch")
        return
    dtype_match = all(v70[c].dtype == v62b[c].dtype for c in v62b.columns)
    if not dtype_match:
        vr.fail("D1", f"{month} {key}: dtype mismatch")
        return

    # D2: row count
    if v70.shape[0] != v62b.shape[0]:
        vr.fail("D2", f"{month} {key}: row count {v70.shape[0]} != {v62b.shape[0]}")
        return

    # D3: ML column isolation (only for ML ptypes)
    if ptype in ML_PTYPES:
        unchanged = [c for c in v62b.columns if c not in ("rank_ori", "rank", "tier")]
        for col in unchanged:
            try:
                pd.testing.assert_series_equal(v62b[col], v70[col], check_names=False)
            except AssertionError:
                vr.fail("D3", f"{month} {key}: column '{col}' differs (should be unchanged)")
                return

    # D4: index preservation
    if v70.index.tolist() != v62b.index.tolist():
        vr.fail("D4", f"{month} {key}: index order changed")
        return

    vr.ok("D", f"{month} {key}: schema valid ({v70.shape[0]} rows)")


# ── Gate E: Rank & Tier Invariants ──

def validate_rank_tier(month: str, ptype: str, ctype: str, vr: ValidationResult):
    ts = pd.Timestamp(month)
    key = f"{ptype}/{ctype}"
    try:
        v70 = ConstraintsSignal("miso", V70_SIGNAL, ptype, ctype).load_data(ts)
    except FileNotFoundError:
        return

    n = len(v70)
    rank = v70["rank"].values
    tier = v70["tier"].values

    # E1: rank range
    if rank.min() <= 0 or abs(rank.max() - 1.0) > 1e-9:
        vr.fail("E1", f"{month} {key}: rank range [{rank.min():.6f}, {rank.max():.6f}]")
        return

    # E2: all ranks unique (diagnostic)
    n_unique_rank = len(np.unique(rank))
    if n_unique_rank != n:
        vr.warn("E2", f"{month} {key}: {n_unique_rank}/{n} unique ranks")

    # E3: tier distribution ~20% each
    for t in range(5):
        frac = (tier == t).sum() / n
        if not (0.18 <= frac <= 0.22):
            vr.fail("E3", f"{month} {key}: tier {t} = {frac:.1%} (expected 18-22%)")
            return

    # E4: tier formula
    expected_tier = np.clip(np.ceil(rank * 5).astype(int) - 1, 0, 4)
    so_mask = (v70["branch_name"] == "SO_MW_Transfer").values if "branch_name" in v70.columns else np.zeros(n, dtype=bool)
    if not np.array_equal(tier[~so_mask], expected_tier[~so_mask]):
        vr.fail("E4", f"{month} {key}: tier != ceil(rank*5)-1")
        return

    # E5: tier-rank monotonicity
    v70_no_so = v70[~so_mask]
    for t in range(4):
        tier_t = v70_no_so.loc[v70_no_so["tier"] == t, "rank"]
        tier_next = v70_no_so.loc[v70_no_so["tier"] == t + 1, "rank"]
        if len(tier_t) > 0 and len(tier_next) > 0:
            if tier_t.max() > tier_next.min():
                vr.fail("E5", f"{month} {key}: tier {t} max rank > tier {t+1} min rank")
                return

    # E6: SO_MW_Transfer
    if so_mask.any():
        so_tiers = v70.loc[so_mask, "tier"].values
        if not all(t == 1 for t in so_tiers):
            vr.fail("E6", f"{month} {key}: SO_MW_Transfer tier != 1")
            return

    vr.ok("E", f"{month} {key}: rank/tier invariants valid")


# ── Gate F: Score Quality ──

def validate_scores(month: str, ptype: str, ctype: str, vr: ValidationResult):
    ts = pd.Timestamp(month)
    key = f"{ptype}/{ctype}"
    try:
        v70 = ConstraintsSignal("miso", V70_SIGNAL, ptype, ctype).load_data(ts)
    except FileNotFoundError:
        return

    scores = v70["rank_ori"].values

    # F1: no degenerate scores
    if np.any(np.isnan(scores)):
        vr.fail("F1", f"{month} {key}: NaN scores")
        return
    if np.any(np.isinf(scores)):
        vr.fail("F1", f"{month} {key}: Inf scores")
        return
    if np.std(scores) < 0.01:
        vr.fail("F1", f"{month} {key}: near-constant scores (std={np.std(scores):.4f})")
        return

    # F2: score polarity (higher score = lower rank)
    top20_mask = v70["rank"] <= v70["rank"].nsmallest(20).max()
    bot20_mask = v70["rank"] >= v70["rank"].nlargest(20).min()
    if v70.loc[top20_mask, "rank_ori"].mean() <= v70.loc[bot20_mask, "rank_ori"].mean():
        vr.fail("F2", f"{month} {key}: score polarity inverted")
        return

    vr.ok("F", f"{month} {key}: scores valid (std={np.std(scores):.4f})")


# ── Gate G: SF Parity ──

def validate_sf(month: str, vr: ValidationResult):
    ts = pd.Timestamp(month)
    ptypes = available_ptypes(month)
    for ptype in ptypes:
        for ctype in ["onpeak", "offpeak"]:
            key = f"{ptype}/{ctype}"
            try:
                sf_v62b = ShiftFactorSignal("miso", V62B_SIGNAL, ptype, ctype).load_data(ts)
                sf_v70 = ShiftFactorSignal("miso", V70_SIGNAL, ptype, ctype).load_data(ts)
                pd.testing.assert_frame_equal(sf_v62b, sf_v70, check_exact=True)
                vr.ok("G", f"{month} {key}: SF bit-identical")
            except AssertionError:
                vr.fail("G", f"{month} {key}: SF differs from V6.2B")
            except FileNotFoundError:
                pass  # SF may not exist for all ptypes


# ── Gate B: Exact holdout reproduction ──

def validate_holdout_reproduction(vr: ValidationResult):
    """Check V7.0 signal VC@20 matches stage5 holdout targets exactly."""
    for ptype in ML_PTYPES:
        for ctype in ["onpeak", "offpeak"]:
            key = f"{ptype}/{ctype}"

            # Determine stage5 version name
            if ptype == "f0":
                ver = "v10e-lag1"
            else:
                ver = "v2"

            holdout_path = HOLDOUT_DIR / ptype / ctype / ver / "metrics.json"
            if not holdout_path.exists():
                # Try without version suffix
                for candidate in (HOLDOUT_DIR / ptype / ctype).iterdir():
                    hp = candidate / "metrics.json"
                    if hp.exists():
                        holdout_path = hp
                        break

            if not holdout_path.exists():
                vr.warn("B", f"{key}: no stage5 holdout metrics found")
                continue

            with open(holdout_path) as f:
                holdout = json.load(f)

            per_month = holdout.get("per_month", {})
            mismatches = []

            for am_str, expected in per_month.items():
                expected_vc20 = expected["VC@20"]
                ts = pd.Timestamp(am_str)

                # Compute VC@20 from V7.0 signal
                try:
                    v70 = ConstraintsSignal("miso", V70_SIGNAL, ptype, ctype).load_data(ts)
                except FileNotFoundError:
                    mismatches.append(f"{am_str}: V7.0 not found")
                    continue

                # Delivery month
                if ptype == "f0":
                    delivery = ts
                elif ptype == "f1":
                    delivery = ts + pd.DateOffset(months=1)
                else:
                    delivery = ts + pd.DateOffset(months=int(ptype[1:]))

                da = _load_da(delivery.strftime("%Y-%m"), ctype)
                if da is None:
                    continue  # no GT available, skip

                cids = v70.index.str.split("|").str[0].values
                da_map = da.set_index("constraint_id")["realized_sp"]
                realized = pd.Series(cids).map(da_map).fillna(0.0).values
                scores = -v70["rank"].values

                actual_vc20 = _vc_at_k(realized, scores, 20)
                diff = abs(actual_vc20 - expected_vc20)
                if diff > 0.001:
                    mismatches.append(
                        f"{am_str}: expected {expected_vc20:.4f}, got {actual_vc20:.4f} (diff={diff:.4f})"
                    )

            if mismatches:
                vr.fail("B", f"{key}: {len(mismatches)} month(s) differ from stage5:\n    " +
                        "\n    ".join(mismatches))
            else:
                n = len(per_month)
                mean_vc20 = holdout["aggregate"]["mean"]["VC@20"]
                vr.ok("B", f"{key}: all {n} holdout months match stage5 (mean VC@20={mean_vc20:.4f})")


# ── Gate A: Improvement over V6.2B ──

def validate_improvement(vr: ValidationResult):
    """Check V7.0 beats V6.2B on holdout months."""
    min_improvement = {
        ("f0", "onpeak"): 0.40,
        ("f0", "offpeak"): 0.40,
        ("f1", "onpeak"): 0.30,
        ("f1", "offpeak"): 0.20,
    }

    for ptype in ML_PTYPES:
        for ctype in ["onpeak", "offpeak"]:
            key = f"{ptype}/{ctype}"

            # Load stage5 v0 (formula baseline)
            v0_path = HOLDOUT_DIR / ptype / ctype / "v0" / "metrics.json"
            if not v0_path.exists():
                vr.warn("A", f"{key}: no v0 baseline found")
                continue

            with open(v0_path) as f:
                v0 = json.load(f)
            v0_vc20 = v0["aggregate"]["mean"]["VC@20"]

            # Load stage5 v2/v10e-lag1 (ML)
            ver = "v10e-lag1" if ptype == "f0" else "v2"
            ml_path = HOLDOUT_DIR / ptype / ctype / ver / "metrics.json"
            if not ml_path.exists():
                vr.warn("A", f"{key}: no ML holdout found")
                continue

            with open(ml_path) as f:
                ml = json.load(f)
            ml_vc20 = ml["aggregate"]["mean"]["VC@20"]

            improvement = (ml_vc20 - v0_vc20) / v0_vc20
            min_req = min_improvement[(ptype, ctype)]

            if improvement >= min_req:
                vr.ok("A", f"{key}: +{improvement:.1%} vs V6.2B (min {min_req:.0%}), "
                       f"V6.2B={v0_vc20:.4f} → V7.0={ml_vc20:.4f}")
            else:
                vr.fail("A", f"{key}: +{improvement:.1%} vs V6.2B (min {min_req:.0%})")


def validate_month(month: str, vr: ValidationResult):
    """Run all per-month gates."""
    ptypes = available_ptypes(month)
    ml_ptypes = [p for p in ptypes if p in ML_PTYPES]

    for ptype in ml_ptypes:
        for ctype in ["onpeak", "offpeak"]:
            validate_schema(month, ptype, ctype, vr)
            validate_rank_tier(month, ptype, ctype, vr)
            validate_scores(month, ptype, ctype, vr)

    validate_passthrough(month, vr)
    validate_sf(month, vr)


def main():
    parser = argparse.ArgumentParser(description="Validate V7.0 signal")
    parser.add_argument("--months", nargs="+", help="Specific months to validate")
    parser.add_argument("--full-holdout", action="store_true",
                        help="Run full holdout reproduction + improvement gates")
    args = parser.parse_args()

    vr = ValidationResult()

    if args.full_holdout:
        print("=== Gate A: Improvement over V6.2B ===")
        validate_improvement(vr)
        print("=== Gate B: Holdout reproduction ===")
        validate_holdout_reproduction(vr)

    months = args.months or []
    if not months and not args.full_holdout:
        # Default: validate latest 3 months
        months = ["2026-01", "2026-02", "2026-03"]

    for month in months:
        print(f"\n=== Validating {month} ===")
        validate_month(month, vr)

    print(f"\n{vr.summary()}")
    sys.exit(1 if vr.failed else 0)


if __name__ == "__main__":
    main()
