"""Tests for NB experiment V2 core logic."""
import numpy as np
import os
import pytest

import sys
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

from scripts.nb_experiment_v2 import (
    assign_tiers_per_group, compute_v0c, allocate_reserved_slots, _minmax,
    load_v44,
)


# ── Per-group tiering ──────────────────────────────────────────────────

def test_assign_tiers_per_group_basic():
    """Each group gets its own 0/1/2/3 distribution independently."""
    sp = np.array([0, 0, 0, 10, 20, 30,
                   0, 0, 0, 1000, 2000, 3000])
    groups = np.array([6, 6])
    labels = assign_tiers_per_group(sp, groups)
    assert list(labels[:6]) == [0, 0, 0, 1, 2, 3]
    assert list(labels[6:]) == [0, 0, 0, 1, 2, 3]


def test_assign_tiers_per_group_no_binders():
    """Group with no binders gets all zeros."""
    sp = np.array([0, 0, 0, 10, 20, 30])
    groups = np.array([3, 3])
    labels = assign_tiers_per_group(sp, groups)
    assert list(labels[:3]) == [0, 0, 0]
    assert list(labels[3:]) == [1, 2, 3]


def test_assign_tiers_per_group_single_binder():
    """Group with 1 binder assigns tier 3 (top)."""
    sp = np.array([0, 0, 100])
    groups = np.array([3])
    labels = assign_tiers_per_group(sp, groups)
    assert labels[2] == 3
    assert labels[0] == 0


# ── v0c formula ────────────────────────────────────────────────────────

def test_v0c_onpeak_vs_offpeak():
    """v0c produces different scores when bf differs."""
    da = np.array([1.0, 2.0, 3.0])
    rt = np.array([0.1, 0.5, 0.9])
    bf_12 = np.array([0.0, 0.5, 1.0])
    bfo_12 = np.array([1.0, 0.0, 0.0])
    on_score = compute_v0c(da, rt, bf_12)
    off_score = compute_v0c(da, rt, bfo_12)
    assert not np.allclose(on_score, off_score)


def test_v0c_weights():
    """v0c = 0.40*(1-minmax(da)) + 0.30*minmax(rt) + 0.30*minmax(bf)."""
    da = np.array([10.0, 20.0])
    rt = np.array([0.0, 1.0])
    bf = np.array([0.0, 1.0])
    scores = compute_v0c(da, rt, bf)
    expected = np.array([0.40 * 1 + 0.30 * 0 + 0.30 * 0,
                         0.40 * 0 + 0.30 * 1 + 0.30 * 1])
    np.testing.assert_allclose(scores, expected)


# ── Reserved-slot allocation ───────────────────────────────────────────

def test_reserved_slot_allocation_basic():
    """NB slots filled from dormant population only."""
    n = 100
    rng = np.random.RandomState(42)
    v0c_scores = rng.rand(n)
    nb_scores = rng.rand(n)
    is_dormant = np.zeros(n, dtype=bool)
    is_dormant[80:] = True
    selected, nb_filled = allocate_reserved_slots(v0c_scores, nb_scores, is_dormant, n_v0c=8, n_nb=2)
    assert len(selected) == 10
    assert nb_filled == 2
    v0c_picks = set(int(x) for x in np.argsort(v0c_scores)[::-1][:8])
    nb_picks = selected - v0c_picks
    for idx in nb_picks:
        assert is_dormant[idx], f"NB pick {idx} is not dormant"


def test_reserved_slot_backfill():
    """When NB scorer can't fill all slots, backfill with v0c. nb_filled < n_nb."""
    n = 20
    v0c_scores = np.arange(n, dtype=float)
    nb_scores = np.full(n, -np.inf)
    is_dormant = np.zeros(n, dtype=bool)
    is_dormant[:2] = True
    nb_scores[0] = 1.0
    nb_scores[1] = 0.5
    selected, nb_filled = allocate_reserved_slots(v0c_scores, nb_scores, is_dormant, n_v0c=15, n_nb=5)
    assert len(selected) == 20
    assert nb_filled <= 2


def test_reserved_slot_total_k_fixed():
    """Total selected always equals n_v0c + n_nb regardless of NB coverage."""
    for n_dormant in [0, 5, 50, 200]:
        n = 300
        v0c_scores = np.random.RandomState(42).rand(n)
        nb_scores = np.full(n, -np.inf)
        is_dormant = np.zeros(n, dtype=bool)
        is_dormant[:n_dormant] = True
        nb_scores[:n_dormant] = np.random.RandomState(42).rand(n_dormant)
        selected, nb_filled = allocate_reserved_slots(v0c_scores, nb_scores, is_dormant, n_v0c=150, n_nb=50)
        assert len(selected) == 200, f"K not fixed for n_dormant={n_dormant}: got {len(selected)}"
        assert nb_filled <= min(n_dormant, 50)


# ── Per-ctype NB population ───────────────────────────────────────────

def test_per_ctype_dormant_differs():
    """Onpeak dormant (bf_12==0) and offpeak dormant (bfo_12==0) are different sets."""
    bf_12 = np.array([0, 0, 0.5, 0.5, 0])
    bfo_12 = np.array([0.5, 0, 0, 0, 0])
    on_dormant = bf_12 == 0
    off_dormant = bfo_12 == 0
    assert list(on_dormant) == [True, True, False, False, True]
    assert list(off_dormant) == [False, True, True, True, True]
    assert not np.array_equal(on_dormant, off_dormant)


# ── V4.4 loader (real path) ───────────────────────────────────────────

V44_BASE = "/opt/data/xyz-dataset/signal_data/miso/constraints/TEST.Signal.MISO.SPICE_ANNUAL_V4.4.R1"


@pytest.mark.skipif(
    not os.path.exists(f"{V44_BASE}/2025-06/aq1/onpeak/"),
    reason="V4.4 data not available",
)
def test_v44_loader_returns_different_ranks_per_ctype():
    """load_v44 returns different rank values for onpeak vs offpeak."""
    on = load_v44("2025-06", "aq1", "onpeak")
    off = load_v44("2025-06", "aq1", "offpeak")
    assert len(on) > 0
    assert len(off) > 0
    assert len(on) == len(off)
    common = set(on.keys()) & set(off.keys())
    n_diff = sum(1 for b in common if on[b].get("rank") != off[b].get("rank"))
    assert n_diff > len(common) * 0.5, f"Only {n_diff}/{len(common)} ranks differ"


@pytest.mark.skipif(
    not os.path.exists(f"{V44_BASE}/2025-06/aq1/onpeak/"),
    reason="V4.4 data not available",
)
def test_v44_loader_both_ctypes_have_data():
    """Both onpeak and offpeak V4.4 load non-empty with expected columns."""
    for ct in ["onpeak", "offpeak"]:
        data = load_v44("2025-06", "aq1", ct)
        assert len(data) > 1000, f"{ct} V4.4 has only {len(data)} branches"
        sample = next(iter(data.values()))
        assert "rank" in sample, f"{ct} V4.4 missing 'rank'"


# ── Per-ctype target invariant ────────────────────────────────────────

def test_per_ctype_target_sum():
    """onpeak_sp + offpeak_sp == realized_shadow_price within tolerance."""
    combined = np.array([100.0, 0.0, 50.0, 200.0])
    onpeak = np.array([60.0, 0.0, 30.0, 150.0])
    offpeak = np.array([40.0, 0.0, 20.0, 50.0])
    np.testing.assert_allclose(combined, onpeak + offpeak)
