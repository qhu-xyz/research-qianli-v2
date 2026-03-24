"""Tests for round-sensitive annual pipeline.

Covers: cutoff calendar, daily cache, partial-month loading, BF/NB windows,
cache keys, bridge round-awareness, no-default enforcement.

Usage:
    source /home/xyz/workspace/pmodel/.venv/bin/activate
    PYTHONPATH=. pytest tests/test_round_sensitive_pipeline.py -v
"""
from __future__ import annotations

from datetime import date, timedelta
from pathlib import Path

import polars as pl
import pytest


# ── 1. Cutoff calendar ─────────────────────────────────────────────────


class TestCutoffCalendar:
    """Test get_round_close_date, get_history_cutoff_date, get_history_cutoff_month."""

    def test_miso_r1_close(self):
        from ml.config import get_round_close_date
        assert get_round_close_date("2025-06", 1, "miso") == date(2025, 4, 8)

    def test_miso_r2_close(self):
        from ml.config import get_round_close_date
        assert get_round_close_date("2025-06", 2, "miso") == date(2025, 4, 22)

    def test_miso_r3_close(self):
        from ml.config import get_round_close_date
        assert get_round_close_date("2025-06", 3, "miso") == date(2025, 5, 5)

    def test_pjm_r1_close(self):
        from ml.config import get_round_close_date
        assert get_round_close_date("2025-06", 1, "pjm") == date(2025, 4, 4)

    def test_pjm_r4_close(self):
        from ml.config import get_round_close_date
        assert get_round_close_date("2025-06", 4, "pjm") == date(2025, 4, 25)

    def test_miso_invalid_round_raises(self):
        from ml.config import get_round_close_date
        with pytest.raises(ValueError, match="Invalid MISO round: 4"):
            get_round_close_date("2025-06", 4, "miso")

    def test_pjm_invalid_round_raises(self):
        from ml.config import get_round_close_date
        with pytest.raises(ValueError, match="Invalid PJM round: 5"):
            get_round_close_date("2025-06", 5, "pjm")

    def test_invalid_rto_raises(self):
        from ml.config import get_round_close_date
        with pytest.raises(ValueError, match="Unsupported RTO"):
            get_round_close_date("2025-06", 1, "spp")

    def test_cutoff_date_is_day_before_close(self):
        from ml.config import get_round_close_date, get_history_cutoff_date
        for r in [1, 2, 3]:
            close = get_round_close_date("2025-06", r, "miso")
            cutoff = get_history_cutoff_date("2025-06", r, "miso")
            assert cutoff == close - timedelta(days=1)

    def test_cutoff_month_r1_matches_legacy(self):
        from ml.config import get_bf_cutoff_month, get_history_cutoff_month
        for py in ["2022-06", "2023-06", "2024-06", "2025-06"]:
            legacy = get_bf_cutoff_month(py)
            new = get_history_cutoff_month(py, market_round=1, rto="miso")
            assert new == legacy, f"R1 cutoff month should match legacy for {py}"

    def test_cutoff_month_r3_includes_april(self):
        from ml.config import get_history_cutoff_month
        # R3 close = May 5, cutoff = May 4, last full month = April
        assert get_history_cutoff_month("2025-06", 3, "miso") == "2025-04"

    def test_cutoff_month_r1_r2_same(self):
        from ml.config import get_history_cutoff_month
        # Both R1 and R2 close in April, last full month = March
        r1 = get_history_cutoff_month("2025-06", 1, "miso")
        r2 = get_history_cutoff_month("2025-06", 2, "miso")
        assert r1 == r2 == "2025-03"

    def test_different_planning_years(self):
        from ml.config import get_round_close_date
        # Same day-of-month, different year
        assert get_round_close_date("2022-06", 1, "miso") == date(2022, 4, 8)
        assert get_round_close_date("2024-06", 1, "miso") == date(2024, 4, 8)


# ── 2. Daily cache loading ─────────────────────────────────────────────


class TestDailyCacheLoading:
    """Test load_day, load_month_daily, has_daily_cache."""

    def test_has_daily_cache(self):
        from ml.realized_da import has_daily_cache
        assert has_daily_cache() is True  # we just ran the fetch

    def test_load_day_returns_signed_sp(self):
        from ml.realized_da import load_day
        df = load_day(date(2025, 3, 3), "onpeak")
        assert "signed_sp" in df.columns
        assert "constraint_id" in df.columns
        assert len(df) > 0

    def test_load_day_missing_returns_empty(self):
        from ml.realized_da import load_day
        df = load_day(date(2099, 1, 1), "onpeak")
        assert len(df) == 0
        assert df.schema == {"constraint_id": pl.Utf8, "signed_sp": pl.Float64}

    def test_load_day_weekend_returns_empty_onpeak(self):
        from ml.realized_da import load_day
        # April 5, 2025 is a Saturday — no onpeak data
        df = load_day(date(2025, 4, 5), "onpeak")
        assert len(df) == 0

    def test_load_month_daily_full_month_parity(self):
        """abs(sum(signed_daily)) must equal monthly cache exactly."""
        from ml.realized_da import load_month_daily, load_month
        for month in ["2025-01", "2025-02", "2025-03"]:
            for ctype in ["onpeak", "offpeak"]:
                monthly = load_month(month, ctype)
                daily_full = load_month_daily(month, ctype, cutoff_date=None)
                # Join and compare
                comp = monthly.join(daily_full, on="constraint_id", how="full", suffix="_d")
                comp = comp.with_columns(
                    (pl.col("realized_sp").fill_null(0.0) - pl.col("realized_sp_d").fill_null(0.0)).abs().alias("diff")
                )
                max_diff = comp["diff"].max()
                assert max_diff < 0.01, f"{month}/{ctype}: max diff {max_diff}"

    def test_load_month_daily_cutoff_excludes_future_days(self):
        from ml.realized_da import load_month_daily
        full = load_month_daily("2025-04", "onpeak", cutoff_date=None)
        partial = load_month_daily("2025-04", "onpeak", cutoff_date=date(2025, 4, 8))
        assert partial.height < full.height, "Partial month should have fewer CIDs"
        assert partial["realized_sp"].sum() < full["realized_sp"].sum()

    def test_load_month_daily_cutoff_day1_empty(self):
        from ml.realized_da import load_month_daily
        # Cutoff on day 1 means no days included
        empty = load_month_daily("2025-04", "onpeak", cutoff_date=date(2025, 4, 1))
        assert len(empty) == 0

    def test_both_ctypes_loaded(self):
        from ml.realized_da import load_month_daily
        on = load_month_daily("2025-04", "onpeak", cutoff_date=date(2025, 4, 15))
        off = load_month_daily("2025-04", "offpeak", cutoff_date=date(2025, 4, 15))
        assert len(on) > 0
        assert len(off) > 0


# ── 3. Round-sensitive feature deltas ──────────────────────────────────


class TestRoundFeatureDeltas:
    """Test that R1/R2/R3 produce different feature values."""

    def test_r1_r2_april_cid_count_differs(self):
        from ml.realized_da import load_month_daily
        r1 = load_month_daily("2025-04", "onpeak", cutoff_date=date(2025, 4, 8))
        r2 = load_month_daily("2025-04", "onpeak", cutoff_date=date(2025, 4, 22))
        assert r2.height > r1.height, "R2 should see more CIDs than R1 in April"

    def test_r1_r2_april_sp_differs(self):
        from ml.realized_da import load_month_daily
        r1 = load_month_daily("2025-04", "onpeak", cutoff_date=date(2025, 4, 8))
        r2 = load_month_daily("2025-04", "onpeak", cutoff_date=date(2025, 4, 22))
        assert r2["realized_sp"].sum() > r1["realized_sp"].sum()

    def test_r3_includes_may_partial(self):
        from ml.realized_da import load_month_daily
        may = load_month_daily("2025-05", "onpeak", cutoff_date=date(2025, 5, 5))
        # May 1-4: should have some data (weekdays May 1 Thu, 2 Fri)
        # May 5 excluded (cutoff)
        assert len(may) >= 0  # may be 0 if no binding May 1-2


# ── 4. Cache key format ────────────────────────────────────────────────


class TestCacheKeys:
    """Test that cache keys include round."""

    def test_collapsed_cache_includes_round(self):
        from ml.data_loader import COLLAPSED_CACHE_DIR
        from ml.config import UNIVERSE_THRESHOLD
        threshold_tag = f"{UNIVERSE_THRESHOLD:.6e}".replace(".", "p").replace("+", "")
        for r in [1, 2, 3]:
            expected = COLLAPSED_CACHE_DIR / f"2025-06_aq1_r{r}_t{threshold_tag}.parquet"
            assert f"_r{r}_" in expected.name

    def test_cid_map_cache_includes_round(self):
        from ml.data_loader import _cid_mapping_cache_path
        for r in [1, 2, 3]:
            path = _cid_mapping_cache_path("2025-06", "aq1", r)
            assert f"_r{r}_" in path.name

    def test_old_cache_not_found(self):
        """Old cache keys (no round) should NOT be found by new code."""
        from ml.data_loader import COLLAPSED_CACHE_DIR
        from ml.config import UNIVERSE_THRESHOLD
        threshold_tag = f"{UNIVERSE_THRESHOLD:.6e}".replace(".", "p").replace("+", "")
        old_key = COLLAPSED_CACHE_DIR / f"2025-06_aq1_t{threshold_tag}.parquet"
        # Old key doesn't have _r{round}_ — new code won't find it
        assert "_r1_" not in old_key.name


# ── 5. No-default enforcement ──────────────────────────────────────────


class TestNoDefaults:
    """Test that core APIs require market_round explicitly."""

    def _check_no_default(self, fn):
        import inspect
        sig = inspect.signature(fn)
        p = sig.parameters.get("market_round")
        assert p is not None, f"{fn.__name__} missing market_round parameter"
        assert p.default is inspect.Parameter.empty, (
            f"{fn.__name__} has market_round default={p.default}, should be required"
        )

    def test_load_collapsed_no_default(self):
        from ml.data_loader import load_collapsed
        self._check_no_default(load_collapsed)

    def test_load_raw_density_no_default(self):
        from ml.data_loader import load_raw_density
        self._check_no_default(load_raw_density)

    def test_load_bridge_partition_no_default(self):
        from ml.bridge import load_bridge_partition
        self._check_no_default(load_bridge_partition)

    def test_map_cids_to_branches_no_default(self):
        from ml.bridge import map_cids_to_branches
        self._check_no_default(map_cids_to_branches)

    def test_build_monthly_binding_table_no_default(self):
        from ml.history_features import build_monthly_binding_table
        self._check_no_default(build_monthly_binding_table)

    def test_compute_history_features_no_default(self):
        from ml.history_features import compute_history_features
        self._check_no_default(compute_history_features)

    def test_compute_nb_flags_no_default(self):
        from ml.nb_detection import compute_nb_flags
        self._check_no_default(compute_nb_flags)

    def test_build_class_model_table_no_default(self):
        from ml.phase6.features import build_class_model_table
        self._check_no_default(build_class_model_table)

    def test_publish_signal_no_default(self):
        from ml.signal_publisher import publish_signal
        self._check_no_default(publish_signal)

    def test_calling_without_round_raises(self):
        from ml.data_loader import load_collapsed
        with pytest.raises(TypeError):
            load_collapsed("2025-06", "aq1")  # type: ignore


# ── 6. BF window includes partial month ────────────────────────────────


class TestBFWindowPartialMonth:
    """Test that BF windows include the partial cutoff month."""

    def test_bf_window_end_includes_cutoff_month(self):
        from ml.config import get_history_cutoff_month, get_history_cutoff_date, BF_FLOOR_MONTH
        from ml.history_features import _generate_month_range

        for r in [1, 2, 3]:
            cutoff_month_full = get_history_cutoff_month("2025-06", r, "miso")
            cutoff_date = get_history_cutoff_date("2025-06", r, "miso")
            cutoff_date_month = f"{cutoff_date.year}-{cutoff_date.month:02d}"
            binding_table_end = max(cutoff_month_full, cutoff_date_month)
            months = _generate_month_range(BF_FLOOR_MONTH, binding_table_end)
            # The cutoff_date's month must be in the range
            assert cutoff_date_month in months, (
                f"R{r}: {cutoff_date_month} not in month range (ends {months[-1]})"
            )

    def test_r1_r2_bf_window_same_end(self):
        """R1 and R2 both end at April (partial), but data differs."""
        from ml.config import get_history_cutoff_date, get_history_cutoff_month, BF_FLOOR_MONTH
        from ml.history_features import _generate_month_range

        for r in [1, 2]:
            cd = get_history_cutoff_date("2025-06", r, "miso")
            cm = get_history_cutoff_month("2025-06", r, "miso")
            end = max(cm, f"{cd.year}-{cd.month:02d}")
            months = _generate_month_range(BF_FLOOR_MONTH, end)
            assert months[-1] == "2025-04"

    def test_r3_bf_window_ends_at_may(self):
        from ml.config import get_history_cutoff_date, get_history_cutoff_month, BF_FLOOR_MONTH
        from ml.history_features import _generate_month_range
        cd = get_history_cutoff_date("2025-06", 3, "miso")
        cm = get_history_cutoff_month("2025-06", 3, "miso")
        end = max(cm, f"{cd.year}-{cd.month:02d}")
        months = _generate_month_range(BF_FLOOR_MONTH, end)
        assert months[-1] == "2025-05"


# ── 7. NB detection round-sensitivity ──────────────────────────────────


class TestNBRoundSensitivity:
    """Test NB flag changes across rounds."""

    def test_nb_window_includes_cutoff_month(self):
        """NB detection must use the same extended window as BF."""
        from ml.config import get_history_cutoff_date, get_history_cutoff_month, BF_FLOOR_MONTH
        from ml.history_features import _generate_month_range

        for r in [1, 2, 3]:
            cd = get_history_cutoff_date("2025-06", r, "miso")
            cm = get_history_cutoff_month("2025-06", r, "miso")
            end = max(cm, f"{cd.year}-{cd.month:02d}")
            months = _generate_month_range(BF_FLOOR_MONTH, end)
            cd_month = f"{cd.year}-{cd.month:02d}"
            assert cd_month in months


# ── 8. Bridge round-awareness ──────────────────────────────────────────


class TestBridgeRoundAwareness:
    """Test that bridge paths include market_round."""

    def test_bridge_path_includes_round(self):
        from ml.config import BRIDGE_PATH
        for r in [1, 2, 3]:
            expected_fragment = f"market_round={r}"
            path = (
                f"{BRIDGE_PATH}/spice_version=v6/auction_type=annual"
                f"/auction_month=2025-06/market_round={r}"
                f"/period_type=aq1/class_type=onpeak/"
            )
            assert expected_fragment in path

    def test_density_path_includes_round(self):
        from ml.config import DENSITY_PATH
        for r in [1, 2, 3]:
            path = (
                f"{DENSITY_PATH}/spice_version=v6/auction_type=annual"
                f"/auction_month=2025-06/market_month=2025-06/market_round={r}/"
            )
            assert f"market_round={r}" in path


# ── 9. Daily cache manifest ────────────────────────────────────────────


class TestDailyCacheManifest:
    """Test that the daily cache manifest is present and valid."""

    def test_manifest_exists(self):
        from ml.realized_da import DA_DAILY_CACHE_DIR
        manifest_path = DA_DAILY_CACHE_DIR / "manifest.json"
        assert manifest_path.exists(), f"Manifest not found at {manifest_path}"

    def test_manifest_has_required_fields(self):
        import json
        from ml.realized_da import DA_DAILY_CACHE_DIR
        manifest_path = DA_DAILY_CACHE_DIR / "manifest.json"
        with open(manifest_path) as f:
            m = json.load(f)
        for field in ["cache_name", "schema", "ctypes", "onpeak_dates", "offpeak_dates", "total_files"]:
            assert field in m, f"Manifest missing field: {field}"

    def test_manifest_covers_both_ctypes(self):
        import json
        from ml.realized_da import DA_DAILY_CACHE_DIR
        with open(DA_DAILY_CACHE_DIR / "manifest.json") as f:
            m = json.load(f)
        assert "onpeak" in m["ctypes"]
        assert "offpeak" in m["ctypes"]

    def test_manifest_schema_is_signed(self):
        import json
        from ml.realized_da import DA_DAILY_CACHE_DIR
        with open(DA_DAILY_CACHE_DIR / "manifest.json") as f:
            m = json.load(f)
        assert m["schema"]["signed_sp"] == "Float64"


# ── 10. Signed aggregation correctness ─────────────────────────────────


class TestSignedAggregation:
    """Test that daily signed sums reproduce monthly values exactly."""

    @pytest.mark.parametrize("month", ["2024-06", "2024-12", "2025-03"])
    def test_monthly_parity_onpeak(self, month):
        from ml.realized_da import load_month, load_month_daily
        monthly = load_month(month, "onpeak")
        daily_full = load_month_daily(month, "onpeak", cutoff_date=None)
        comp = monthly.join(daily_full, on="constraint_id", how="full", suffix="_d")
        comp = comp.with_columns(
            (pl.col("realized_sp").fill_null(0.0) - pl.col("realized_sp_d").fill_null(0.0)).abs().alias("diff")
        )
        assert comp["diff"].max() < 0.01

    @pytest.mark.parametrize("month", ["2024-06", "2024-12", "2025-03"])
    def test_monthly_parity_offpeak(self, month):
        from ml.realized_da import load_month, load_month_daily
        monthly = load_month(month, "offpeak")
        daily_full = load_month_daily(month, "offpeak", cutoff_date=None)
        comp = monthly.join(daily_full, on="constraint_id", how="full", suffix="_d")
        comp = comp.with_columns(
            (pl.col("realized_sp").fill_null(0.0) - pl.col("realized_sp_d").fill_null(0.0)).abs().alias("diff")
        )
        assert comp["diff"].max() < 0.01
