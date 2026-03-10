from __future__ import annotations

from pathlib import Path

import polars as pl

from ml.lookback_history import (
    clamp_day,
    partial_month_binding_set,
    verify_month_collapse,
)


def _write_daily_month(base: Path, month: str, rows: list[tuple[str, str, float]]) -> None:
    month_path = base / f"{month}.parquet"
    df = pl.DataFrame(
        {
            "market_date": [r[0] for r in rows],
            "constraint_id": [r[1] for r in rows],
            "shadow_price_net_day": [r[2] for r in rows],
        }
    ).with_columns(
        pl.col("market_date").str.strptime(pl.Date, "%Y-%m-%d"),
        pl.col("shadow_price_net_day").abs().alias("realized_sp_day"),
    )
    df.write_parquet(str(month_path))


def _write_monthly(base: Path, month: str, rows: list[tuple[str, float]]) -> None:
    month_path = base / f"{month}.parquet"
    pl.DataFrame(
        {"constraint_id": [r[0] for r in rows], "realized_sp": [r[1] for r in rows]}
    ).write_parquet(str(month_path))


def test_clamp_day_handles_short_months() -> None:
    assert clamp_day("2025-02", 31) == 28
    assert clamp_day("2024-02", 31) == 29
    assert clamp_day("2025-04", 31) == 30
    assert clamp_day("2025-03", 12) == 12


def test_partial_binding_preserves_signed_netting() -> None:
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        daily_dir = Path(tmp) / "daily"
        daily_dir.mkdir()
        _write_daily_month(
            daily_dir,
            "2025-03",
            [
                ("2025-03-01", "A", 10.0),
                ("2025-03-02", "A", -10.0),
                ("2025-03-03", "B", 5.0),
            ],
        )

        binding = partial_month_binding_set("2025-03", 31, cache_dir=str(daily_dir))
        assert "A" not in binding
        assert "B" in binding


def test_verify_month_collapse_matches_when_daily_netting_is_consistent() -> None:
    import tempfile

    with tempfile.TemporaryDirectory() as tmp:
        monthly_dir = Path(tmp) / "monthly"
        daily_dir = Path(tmp) / "daily"
        monthly_dir.mkdir()
        daily_dir.mkdir()

        _write_daily_month(
            daily_dir,
            "2025-03",
            [
                ("2025-03-01", "A", 10.0),
                ("2025-03-02", "A", -10.0),
                ("2025-03-03", "B", 5.0),
                ("2025-03-04", "C", -2.0),
            ],
        )
        _write_monthly(
            monthly_dir,
            "2025-03",
            [
                ("A", 0.0),
                ("B", 5.0),
                ("C", 2.0),
            ],
        )

        check = verify_month_collapse(
            "2025-03",
            look_back_days=31,
            monthly_cache_dir=str(monthly_dir),
            daily_cache_dir=str(daily_dir),
        )
        assert check.matches
