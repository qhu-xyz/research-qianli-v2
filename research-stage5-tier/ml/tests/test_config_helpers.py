# ml/tests/test_config_helpers.py
import pytest
from ml.config import period_offset, delivery_month, has_period_type, f1_eval_months


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
