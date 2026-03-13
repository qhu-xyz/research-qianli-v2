"""Tests for ml/merge.py — two-track merge logic."""
import numpy as np
import polars as pl
import pytest


def _make_track_dfs():
    """Create minimal Track A and Track B DataFrames."""
    track_a = pl.DataFrame({
        "branch_name": [f"est_{i}" for i in range(8)],
        "score": [10.0, 9.0, 8.0, 7.0, 6.0, 5.0, 4.0, 3.0],
        "cohort": ["established"] * 8,
        "realized_shadow_price": [100.0, 80.0, 60.0, 40.0, 20.0, 0.0, 0.0, 0.0],
    })
    track_b = pl.DataFrame({
        "branch_name": [f"nb_{i}" for i in range(4)],
        "score": [0.9, 0.7, 0.3, 0.1],
        "cohort": ["history_dormant"] * 4,
        "realized_shadow_price": [50.0, 30.0, 0.0, 0.0],
    })
    return track_a, track_b


def test_merge_tracks_basic():
    """merge_tracks returns correct top_k_indices for K=5, R=2."""
    from ml.merge import merge_tracks
    track_a, track_b = _make_track_dfs()
    merged, top_k_idx = merge_tracks(track_a, track_b, k=5, r=2)

    assert len(merged) == 12
    assert len(top_k_idx) == 5

    merged_names = merged["branch_name"].to_list()
    for idx in top_k_idx[:3]:
        assert merged_names[idx].startswith("est_")
    for idx in top_k_idx[3:]:
        assert merged_names[idx].startswith("nb_")


def test_merge_tracks_r_zero():
    """R=0 means all slots from Track A."""
    from ml.merge import merge_tracks
    track_a, track_b = _make_track_dfs()
    merged, top_k_idx = merge_tracks(track_a, track_b, k=5, r=0)

    assert len(top_k_idx) == 5
    merged_names = merged["branch_name"].to_list()
    for idx in top_k_idx:
        assert merged_names[idx].startswith("est_")


def test_merge_tracks_r_exceeds_track_b():
    """If R > Track B population, use all Track B branches."""
    from ml.merge import merge_tracks
    track_a, track_b = _make_track_dfs()
    merged, top_k_idx = merge_tracks(track_a, track_b, k=5, r=10)

    merged_names = merged["branch_name"].to_list()
    track_b_in_top = sum(1 for idx in top_k_idx if merged_names[idx].startswith("nb_"))
    assert track_b_in_top == 4


def test_merge_tracks_small_populations():
    """When total population < K, top_k_indices has len < K."""
    from ml.merge import merge_tracks
    track_a = pl.DataFrame({
        "branch_name": ["a0", "a1"],
        "score": [2.0, 1.0],
        "cohort": ["established"] * 2,
        "realized_shadow_price": [100.0, 50.0],
    })
    track_b = pl.DataFrame({
        "branch_name": ["b0"],
        "score": [0.5],
        "cohort": ["history_dormant"],
        "realized_shadow_price": [30.0],
    })
    merged, top_k_idx = merge_tracks(track_a, track_b, k=50, r=10)
    assert len(top_k_idx) == 3
    assert len(merged) == 3


def test_merge_tracks_provenance():
    """Merged DataFrame has a 'track' column."""
    from ml.merge import merge_tracks
    track_a, track_b = _make_track_dfs()
    merged, _ = merge_tracks(track_a, track_b, k=5, r=2)

    assert "track" in merged.columns
    tracks = merged["track"].to_list()
    assert tracks.count("A") == 8
    assert tracks.count("B") == 4
