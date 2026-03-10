# ml/tests/test_config.py
"""Tests for PJM config module."""
import pytest
from ml.config import (
    delivery_month,
    has_period_type,
    period_offset,
    collect_usable_months,
    PJM_CLASS_TYPES,
    V62B_SIGNAL_BASE,
    SPICE6_DENSITY_BASE,
)


def test_period_offset():
    assert period_offset("f0") == 0
    assert period_offset("f1") == 1
    assert period_offset("f11") == 11


def test_delivery_month():
    assert delivery_month("2025-01", "f0") == "2025-01"
    assert delivery_month("2025-01", "f1") == "2025-02"
    assert delivery_month("2025-06", "f11") == "2026-05"


def test_has_period_type_may_f0_only():
    """May auctions have only f0."""
    assert has_period_type("2025-05", "f0") is True
    assert has_period_type("2025-05", "f1") is False


def test_has_period_type_june_all():
    """June auctions have f0-f11."""
    for i in range(12):
        assert has_period_type("2025-06", f"f{i}") is True


def test_class_types():
    assert PJM_CLASS_TYPES == ["onpeak", "dailyoffpeak", "wkndonpeak"]


def test_pjm_paths_exist():
    from pathlib import Path
    assert Path(V62B_SIGNAL_BASE).exists(), f"V6.2B path missing: {V62B_SIGNAL_BASE}"
    assert Path(SPICE6_DENSITY_BASE).exists(), f"Spice6 density path missing: {SPICE6_DENSITY_BASE}"


def test_collect_usable_months_f0():
    """f0 should return 8 contiguous months for a well-covered eval month."""
    months = collect_usable_months("2023-06", "f0", n_months=8)
    assert len(months) == 8
    # Most recent should be 2 months before target (lag built into collect_usable_months)
    assert months[0] <= "2023-04"


def test_collect_usable_months_f1_skips_gaps():
    """f1 should skip months where f1 doesn't exist (May)."""
    months = collect_usable_months("2023-06", "f1", n_months=8)
    assert len(months) >= 6  # may be < 8 due to gaps
    for m in months:
        assert has_period_type(m, "f1"), f"{m} should have f1"
