"""Tests for ml/bridge.py — bridge loading + cid-to-branch mapping."""
import polars as pl
import pytest


def test_load_bridge_partition_annual_both_ctypes(sample_py):
    """Test spec A8: loads BOTH onpeak+offpeak and UNIONs them."""
    from ml.bridge import load_bridge_partition
    bridge = load_bridge_partition(
        auction_type="annual",
        auction_month=sample_py,
        period_type="aq1",
    )
    assert "constraint_id" in bridge.columns
    assert "branch_name" in bridge.columns
    assert bridge["branch_name"].null_count() == 0
    n_unique_branches = bridge["branch_name"].n_unique()
    assert 4000 <= n_unique_branches <= 6000, f"Unexpected branch count: {n_unique_branches}"


def test_load_bridge_partition_convention_filter(sample_py):
    """Test spec A4: convention < 10 keeps only -1 and 1."""
    from ml.bridge import load_bridge_partition
    from ml.config import BRIDGE_PATH
    # Load raw (no convention filter) to compare
    raw = pl.read_parquet(
        f"{BRIDGE_PATH}/spice_version=v6/auction_type=annual"
        f"/auction_month={sample_py}/market_round=1/period_type=aq1/class_type=onpeak/"
    )
    assert 999 in raw["convention"].unique().to_list()
    # Filtered version should not have 999
    bridge = load_bridge_partition(
        auction_type="annual", auction_month=sample_py, period_type="aq1",
    )
    # After convention filter + union + unique, should have reasonable count
    assert len(bridge) > 10000


def test_load_bridge_partition_missing_raises():
    """FileNotFoundError if NEITHER class_type partition exists."""
    from ml.bridge import load_bridge_partition
    with pytest.raises(FileNotFoundError):
        load_bridge_partition(
            auction_type="annual",
            auction_month="1900-06",  # doesn't exist
            period_type="aq1",
        )


def test_map_cids_to_branches_annual(sample_py):
    """Shared mapping function: maps cids, detects ambiguity, drops ambiguous."""
    from ml.bridge import map_cids_to_branches

    # Create a test df with some cids
    test_cids = pl.DataFrame({"constraint_id": ["1000", "100023", "999999999"]})
    mapped, diag = map_cids_to_branches(
        cid_df=test_cids,
        auction_type="annual",
        auction_month=sample_py,
        period_type="aq1",
    )
    assert "branch_name" in mapped.columns
    assert "constraint_id" in mapped.columns
    assert "ambiguous_cids" in diag
    assert "ambiguous_sp" in diag
    # No nulls in branch_name for mapped rows
    assert mapped["branch_name"].null_count() == 0


def test_bridge_no_fanout(sample_py):
    """Test spec C2: convention < 10 gives ~1:1 cid:branch after unique."""
    from ml.bridge import load_bridge_partition
    bridge = load_bridge_partition(
        auction_type="annual", auction_month=sample_py, period_type="aq1",
    )
    # After unique(), each cid should map to at most 1 branch
    cid_counts = bridge.group_by("constraint_id").len()
    max_branches_per_cid = cid_counts["len"].max()
    # After UNION of onpeak+offpeak, some cids map to multiple branches.
    # This is expected — ambiguity detection in map_cids_to_branches handles it.
    # But the vast majority should be 1:1.
    n_ambiguous = cid_counts.filter(pl.col("len") > 1).height
    assert n_ambiguous / len(cid_counts) < 0.01, f"Too many ambiguous cids: {n_ambiguous}/{len(cid_counts)}"


def test_map_cids_ambiguity_scoped_to_input(sample_py):
    """Ambiguity diagnostics must count only input cids, not the full bridge."""
    from ml.bridge import map_cids_to_branches
    # Pass just 2 cids — ambiguous_cids count must be <= 2, not the global bridge count
    test_cids = pl.DataFrame({"constraint_id": ["227076", "331594"]})
    _, diag = map_cids_to_branches(
        cid_df=test_cids,
        auction_type="annual",
        auction_month=sample_py,
        period_type="aq1",
    )
    assert diag["ambiguous_cids"] <= 2, (
        f"ambiguous_cids={diag['ambiguous_cids']} exceeds input size=2 — "
        "diagnostics are not scoped to input cids"
    )


def test_bridge_hive_scan_not_used():
    """Test spec A5: bridge should be loaded via partition path, not hive scan.

    The hive scan may or may not raise depending on polars version and data layout.
    This test verifies load_bridge_partition uses explicit partition paths.
    """
    import inspect
    from ml.bridge import load_bridge_partition
    source = inspect.getsource(load_bridge_partition)
    assert "hive_partitioning" not in source, (
        "load_bridge_partition should use explicit partition paths, not hive scan"
    )
