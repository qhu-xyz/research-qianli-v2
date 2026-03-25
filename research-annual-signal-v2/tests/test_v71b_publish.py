"""Post-publish verification tests for V7.1B.

Validates the published signal files on NFS against the release contract.

Usage:
    PYTHONPATH=. pytest tests/test_v71b_publish.py -v
"""
from __future__ import annotations

from pathlib import Path

import polars as pl
import pytest

CSTR_BASE = Path("/opt/data/xyz-dataset/signal_data/miso/constraints")
SF_BASE = Path("/opt/data/xyz-dataset/signal_data/miso/sf")
V71B_PREFIX = "TEST.Signal.MISO.SPICE_ANNUAL_V7.1B"
V70B_PREFIX = "TEST.Signal.MISO.SPICE_ANNUAL_V7.0B"

PYS = ["2019-06", "2020-06", "2021-06", "2022-06", "2023-06", "2024-06", "2025-06"]
ROUNDS = [1, 2, 3]
AQS_FULL = ["aq1", "aq2", "aq3", "aq4"]
CTYPES = ["onpeak", "offpeak"]
NON_NULL_COLS = ["constraint_id", "branch_name", "flow_direction", "shadow_sign", "tier", "bus_key", "constraint_limit"]
EXPECTED_COL_COUNT = 22


def _cstr_path(R, py, aq, ct):
    return CSTR_BASE / f"{V71B_PREFIX}.R{R}" / py / aq / ct / "signal.parquet"


def _sf_path(R, py, aq, ct):
    return SF_BASE / f"{V71B_PREFIX}.R{R}" / py / aq / ct / "signal.parquet"


def _all_expected_cells():
    """Generate all expected (R, py, aq, ct) tuples."""
    cells = []
    for R in ROUNDS:
        for py in PYS:
            for aq in AQS_FULL:
                if py == "2025-06" and aq == "aq4":
                    continue
                for ct in CTYPES:
                    cells.append((R, py, aq, ct))
    return cells


ALL_CELLS = _all_expected_cells()


# ── 1. Presence ────────────────────────────────────────────────────────


class TestPresence:
    def test_signal_roots_exist(self):
        for R in ROUNDS:
            root = CSTR_BASE / f"{V71B_PREFIX}.R{R}"
            assert root.exists(), f"Signal root missing: {root}"

    def test_total_cell_count(self):
        assert len(ALL_CELLS) == 162

    @pytest.mark.parametrize("R,py,aq,ct", ALL_CELLS[:20])  # sample 20
    def test_constraint_file_exists(self, R, py, aq, ct):
        assert _cstr_path(R, py, aq, ct).exists()

    @pytest.mark.parametrize("R,py,aq,ct", ALL_CELLS[:20])
    def test_sf_file_exists(self, R, py, aq, ct):
        assert _sf_path(R, py, aq, ct).exists()

    def test_2025_aq4_absent(self):
        for R in ROUNDS:
            for ct in CTYPES:
                assert not _cstr_path(R, "2025-06", "aq4", ct).exists()

    def test_historical_aq4_present(self):
        for py in ["2022-06", "2023-06", "2024-06"]:
            for ct in CTYPES:
                assert _cstr_path(1, py, "aq4", ct).exists()


# ── 2. Schema ──────────────────────────────────────────────────────────


class TestSchema:
    def test_column_count(self):
        df = pl.read_parquet(str(_cstr_path(1, "2024-06", "aq1", "onpeak")))
        assert len(df.columns) == EXPECTED_COL_COUNT

    def test_constraint_limit_present(self):
        df = pl.read_parquet(str(_cstr_path(1, "2024-06", "aq1", "onpeak")))
        assert "constraint_limit" in df.columns

    def test_schema_matches_v70b(self):
        v71 = pl.read_parquet(str(_cstr_path(1, "2024-06", "aq1", "onpeak")))
        v70 = pl.read_parquet(str(CSTR_BASE / f"{V70B_PREFIX}.R1/2024-06/aq1/onpeak/signal.parquet"))
        assert set(v71.columns) == set(v70.columns)

    def test_dtype_parity_with_v70b(self):
        v71 = pl.read_parquet(str(_cstr_path(1, "2024-06", "aq1", "onpeak")))
        v70 = pl.read_parquet(str(CSTR_BASE / f"{V70B_PREFIX}.R1/2024-06/aq1/onpeak/signal.parquet"))
        for col in v70.columns:
            assert str(v71[col].dtype) == str(v70[col].dtype), f"dtype mismatch: {col}"

    @pytest.mark.parametrize("col", NON_NULL_COLS)
    def test_non_null_columns(self, col):
        df = pl.read_parquet(str(_cstr_path(1, "2025-06", "aq1", "onpeak")))
        assert df[col].null_count() == 0, f"{col} has nulls"

    def test_constraint_limit_positive(self):
        df = pl.read_parquet(str(_cstr_path(1, "2024-06", "aq1", "onpeak")))
        assert df["constraint_limit"].min() > 0


# ── 3. Row counts & uniqueness ─────────────────────────────────────────


class TestRowCounts:
    @pytest.mark.parametrize("py", ["2022-06", "2024-06", "2025-06"])
    def test_1000_rows(self, py):
        df = pl.read_parquet(str(_cstr_path(1, py, "aq1", "onpeak")))
        assert len(df) == 1000

    def test_no_duplicate_cids(self):
        df = pl.read_parquet(str(_cstr_path(1, "2024-06", "aq1", "onpeak")))
        assert df["constraint_id"].n_unique() == len(df)

    def test_sf_constraint_count_matches(self):
        sf = pl.read_parquet(str(_sf_path(1, "2024-06", "aq1", "onpeak")))
        assert len(sf.columns) - 1 == 1000  # minus pnode_id


# ── 4. Round differences ───────────────────────────────────────────────


class TestRoundDifferences:
    @pytest.mark.parametrize("py", ["2022-06", "2024-06", "2025-06"])
    def test_cid_sets_differ_across_rounds(self, py):
        r1 = set(pl.read_parquet(str(_cstr_path(1, py, "aq1", "onpeak")))["constraint_id"].to_list())
        r2 = set(pl.read_parquet(str(_cstr_path(2, py, "aq1", "onpeak")))["constraint_id"].to_list())
        r3 = set(pl.read_parquet(str(_cstr_path(3, py, "aq1", "onpeak")))["constraint_id"].to_list())
        assert r1 != r2 or r1 != r3, "All rounds have identical CID sets"

    def test_top1_differs_across_rounds(self):
        tops = []
        for R in ROUNDS:
            df = pl.read_parquet(str(_cstr_path(R, "2024-06", "aq1", "onpeak")))
            tops.append(df.sort("rank", descending=True).head(1)["constraint_id"][0])
        assert len(set(tops)) > 1, f"Same top-1 across all rounds: {tops}"


# ── 5. Ctype differences ──────────────────────────────────────────────


class TestCtypeDifferences:
    @pytest.mark.parametrize("py", ["2022-06", "2024-06", "2025-06"])
    def test_rankings_differ_between_ctypes(self, py):
        on = pl.read_parquet(str(_cstr_path(1, py, "aq1", "onpeak")))
        off = pl.read_parquet(str(_cstr_path(1, py, "aq1", "offpeak")))
        on_top5 = on.sort("rank", descending=True).head(5)["constraint_id"].to_list()
        off_top5 = off.sort("rank", descending=True).head(5)["constraint_id"].to_list()
        assert on_top5 != off_top5

    def test_da_rank_values_differ(self):
        on = pl.read_parquet(str(_cstr_path(1, "2024-06", "aq1", "onpeak")))
        off = pl.read_parquet(str(_cstr_path(1, "2024-06", "aq1", "offpeak")))
        common = set(on["constraint_id"].to_list()) & set(off["constraint_id"].to_list())
        sample_cids = list(common)[:20]
        on_vals = on.filter(pl.col("constraint_id").is_in(sample_cids)).sort("constraint_id")["da_rank_value"].to_list()
        off_vals = off.filter(pl.col("constraint_id").is_in(sample_cids)).sort("constraint_id")["da_rank_value"].to_list()
        identical = sum(1 for a, b in zip(on_vals, off_vals) if abs(a - b) < 0.0001)
        assert identical < len(sample_cids), "da_rank_value identical across ctypes"


# ── 6. V7.0B parity ───────────────────────────────────────────────────


class TestV70BParity:
    def test_similar_but_not_identical(self):
        v71 = pl.read_parquet(str(_cstr_path(1, "2024-06", "aq1", "onpeak")))
        v70 = pl.read_parquet(str(CSTR_BASE / f"{V70B_PREFIX}.R1/2024-06/aq1/onpeak/signal.parquet"))
        v71_cids = set(v71["constraint_id"].to_list())
        v70_cids = set(v70["constraint_id"].to_list())
        overlap = len(v71_cids & v70_cids)
        # Should have significant overlap but not 100%
        assert overlap > 500, f"Too little overlap: {overlap}"
        assert overlap < 1000, f"Identical to V7.0B: {overlap}"

    def test_constraint_limit_values_reasonable(self):
        v71 = pl.read_parquet(str(_cstr_path(1, "2024-06", "aq1", "onpeak")))
        v70 = pl.read_parquet(str(CSTR_BASE / f"{V70B_PREFIX}.R1/2024-06/aq1/onpeak/signal.parquet"))
        # Limit distributions should be in similar range
        assert abs(v71["constraint_limit"].mean() - v70["constraint_limit"].mean()) < 200


# ── 7. aq4 behavior ───────────────────────────────────────────────────


class TestAq4:
    def test_aq4_row_count(self):
        df = pl.read_parquet(str(_cstr_path(1, "2023-06", "aq4", "onpeak")))
        assert len(df) == 1000

    def test_aq4_has_constraint_limit(self):
        df = pl.read_parquet(str(_cstr_path(1, "2023-06", "aq4", "onpeak")))
        assert "constraint_limit" in df.columns
        assert df["constraint_limit"].null_count() == 0

    def test_aq4_sf_exists(self):
        assert _sf_path(1, "2023-06", "aq4", "onpeak").exists()
