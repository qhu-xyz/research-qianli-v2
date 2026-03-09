# ml/tests/test_config_helpers.py
import pytest
from ml.config import period_offset, delivery_month, has_period_type, f1_eval_months, collect_usable_months


def test_period_offset():
    assert period_offset("f0") == 0
    assert period_offset("f1") == 1
    assert period_offset("f3") == 3


def test_period_offset_rejects_quarterly():
    with pytest.raises(ValueError, match="monthly types"):
        period_offset("q4")


def test_delivery_month_f0():
    assert delivery_month("2022-09", "f0") == "2022-09"


def test_delivery_month_f1():
    assert delivery_month("2022-09", "f1") == "2022-10"


def test_delivery_month_f2_year_wrap():
    assert delivery_month("2022-11", "f2") == "2023-01"


def test_has_period_type_f1_in_july():
    assert has_period_type("2022-07", "f1") is True


def test_has_period_type_f1_not_in_may():
    assert has_period_type("2022-05", "f1") is False


def test_f1_eval_months_excludes_may_june():
    months = f1_eval_months(full=True)
    for m in months:
        month_num = int(m.split("-")[1])
        assert month_num not in (5, 6), f"f1 should not include {m}"
    assert len(months) > 0


def test_collect_usable_f0_returns_8_contiguous():
    months = collect_usable_months("2022-09", "f0", n_months=8)
    assert len(months) == 8
    # f0 exists every month; latest safe: delivery(2022-07, f0)=2022-07 <= last_full_known=2022-07
    assert months[0] == "2022-07"
    assert months[-1] == "2021-12"


def test_collect_usable_f1_skips_may_june():
    months = collect_usable_months("2022-09", "f1", n_months=8)
    assert len(months) == 8
    for m in months:
        month_num = int(m.split("-")[1])
        assert month_num not in (5, 6), f"f1 should skip {m}"
    # last_full_known = 2022-07. delivery(2022-07, f1)=2022-08 > 2022-07 → NOT usable.
    # Latest safe: 2022-04, delivery=2022-05 <= 2022-07 → usable.
    assert months[0] == "2022-04"


def test_collect_usable_f0_matches_old_contiguous_window():
    """f0 collect_usable should produce identical months to the old lag=1 contiguous window."""
    months = collect_usable_months("2022-09", "f0", n_months=8)
    assert months == ["2022-07", "2022-06", "2022-05", "2022-04",
                      "2022-03", "2022-02", "2022-01", "2021-12"]


def test_collect_usable_f1_early_month_insufficient():
    """With a short max_lookback, early f1 months can't find 6 usable rows."""
    # 2020-07 with max_lookback=6: only ~4 usable f1 months in that range
    months = collect_usable_months("2020-07", "f1", n_months=8, min_months=6, max_lookback=6)
    assert len(months) < 6, f"Expected <6 usable, got {len(months)}: {months}"
    assert months == [], f"Expected empty (min_months guard), got {months}"


def test_collect_usable_f1_enough_history():
    """Later f1 months should have sufficient history."""
    months = collect_usable_months("2021-01", "f1", n_months=8, min_months=6)
    assert len(months) >= 6
