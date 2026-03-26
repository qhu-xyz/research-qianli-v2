from __future__ import annotations

import pandas as pd
import polars as pl

from ml.markets.miso.signal_publisher import _walk_tiers_72b


def _sf_df(columns: dict[str, list[float]]) -> pd.DataFrame:
    return pd.DataFrame(columns, index=pd.Index(["n1", "n2"], name="pnode_id"))


def test_walk_tiers_72b_backfills_specialist_shortfall_within_tier():
    specialist_pool = pl.DataFrame(
        {
            "candidate_idx": [1],
            "constraint_id": ["s1"],
            "branch_name": ["bs1"],
            "bus_key_group": ["g1"],
        }
    )
    base_pool = pl.DataFrame(
        {
            "candidate_idx": [2, 3, 4, 5, 6, 7, 8],
            "constraint_id": ["b1", "b2", "b3", "b4", "b5", "b6", "b7"],
            "branch_name": ["bb1", "bb2", "bb3", "bb4", "bb5", "bb6", "bb7"],
            "bus_key_group": ["g2", "g3", "g4", "g5", "g6", "g7", "g8"],
        }
    )
    sf_pd = _sf_df(
        {
            "s1": [1.0, 0.0],
            "b1": [0.0, 1.0],
            "b2": [0.0, 2.0],
            "b3": [0.0, 3.0],
            "b4": [0.0, 4.0],
            "b5": [0.0, 5.0],
            "b6": [0.0, 6.0],
            "b7": [0.0, 7.0],
        }
    )

    selected, audit = _walk_tiers_72b(
        specialist_pool=specialist_pool,
        base_pool=base_pool,
        sf_pd=sf_pd,
        tier_sizes=[4, 4],
        branch_cap=3,
        chebyshev_threshold=0.0,
        correlation_threshold=2.0,
        specialist_per_tier=1,
    )

    assert selected["candidate_idx"].to_list() == [1, 2, 3, 4, 5, 6, 7, 8]
    assert selected["tier"].to_list() == [0, 0, 0, 0, 1, 1, 1, 1]
    assert audit[0]["selected_specialist"] == 1
    assert audit[0]["selected_base"] == 3
    assert audit[0]["specialist_shortfall"] == 0
    assert audit[1]["selected_specialist"] == 0
    assert audit[1]["selected_base"] == 4
    assert audit[1]["specialist_shortfall"] == 1


def test_walk_tiers_72b_shares_sf_dedup_state_across_subwalks():
    specialist_pool = pl.DataFrame(
        {
            "candidate_idx": [1],
            "constraint_id": ["s1"],
            "branch_name": ["bs1"],
            "bus_key_group": ["g1"],
        }
    )
    base_pool = pl.DataFrame(
        {
            "candidate_idx": [2, 3],
            "constraint_id": ["b_blocked", "b_ok"],
            "branch_name": ["bb1", "bb2"],
            "bus_key_group": ["g1", "g1"],
        }
    )
    sf_pd = _sf_df(
        {
            "s1": [1.0, 0.0],
            "b_blocked": [1.0, 0.0],
            "b_ok": [0.0, 1.0],
        }
    )

    selected, audit = _walk_tiers_72b(
        specialist_pool=specialist_pool,
        base_pool=base_pool,
        sf_pd=sf_pd,
        tier_sizes=[2],
        branch_cap=3,
        chebyshev_threshold=0.05,
        correlation_threshold=2.0,
        specialist_per_tier=1,
    )

    assert selected["candidate_idx"].to_list() == [1, 3]
    assert selected["tier"].to_list() == [0, 0]
    assert audit[0]["rejected_sf_dedup"] == 1


def test_walk_tiers_72b_enforces_branch_cap_during_walk():
    specialist_pool = pl.DataFrame(
        {
            "candidate_idx": [1],
            "constraint_id": ["s1"],
            "branch_name": ["shared_branch"],
            "bus_key_group": ["g1"],
        }
    )
    base_pool = pl.DataFrame(
        {
            "candidate_idx": [2, 3, 4],
            "constraint_id": ["b_same_branch", "b_other_1", "b_other_2"],
            "branch_name": ["shared_branch", "bb2", "bb3"],
            "bus_key_group": ["g2", "g3", "g4"],
        }
    )
    sf_pd = _sf_df(
        {
            "s1": [1.0, 0.0],
            "b_same_branch": [0.0, 1.0],
            "b_other_1": [0.0, 2.0],
            "b_other_2": [0.0, 3.0],
        }
    )

    selected, audit = _walk_tiers_72b(
        specialist_pool=specialist_pool,
        base_pool=base_pool,
        sf_pd=sf_pd,
        tier_sizes=[3],
        branch_cap=1,
        chebyshev_threshold=0.0,
        correlation_threshold=2.0,
        specialist_per_tier=1,
    )

    assert selected["candidate_idx"].to_list() == [1, 3, 4]
    assert audit[0]["rejected_branch_cap"] == 1
