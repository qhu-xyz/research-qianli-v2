"""Two-track merge logic — combines Track A and Track B rankings.

Track A: established branches scored by existing model (v0c or v3a).
Track B: NB candidates scored by Track B binary classifier.

merge_tracks() returns a pre-computed top_k_indices array for use with
evaluate_group(top_k_override=...).
"""
from __future__ import annotations

import numpy as np
import polars as pl


def merge_tracks(
    track_a: pl.DataFrame,
    track_b: pl.DataFrame,
    k: int,
    r: int,
    tau: float | None = None,
) -> tuple[pl.DataFrame, np.ndarray]:
    """Merge Track A and Track B into a single DataFrame with top-K indices.

    Args:
        track_a: established branches with 'score' column (Track A model scores)
        track_b: NB candidates with 'score' column (Track B model scores)
        k: total top-K slots (e.g. 50)
        r: max reserved slots for Track B
        tau: score threshold — only Track B branches with score >= tau get
            reserved slots. Unused slots return to Track A. If None, all
            Track B branches are eligible (backward compatible).

    Returns:
        (merged_df, top_k_indices):
        - merged_df: vertical concat of track_a + track_b with 'track' provenance column
        - top_k_indices: np.ndarray of length min(k, len(merged_df)),
          first (k - r_actual) indices from Track A, last r_actual from Track B
          (all in merged_df index space)
    """
    a = track_a.with_columns(pl.lit("A").alias("track"))
    b = track_b.with_columns(pl.lit("B").alias("track"))

    merged = pl.concat([a, b], how="diagonal")

    n_a = len(track_a)
    n_b = len(track_b)

    b_scores = track_b["score"].to_numpy().astype(np.float64)

    # Score-thresholded R: only include Track B branches with score >= tau
    if tau is not None:
        qualified_mask = b_scores >= tau
        r_actual = min(r, int(qualified_mask.sum()))
    else:
        qualified_mask = None
        r_actual = min(r, n_b)

    n_a_slots = min(k - r_actual, n_a)

    # Top Track A indices (in merged_df space: 0..n_a-1)
    a_scores = track_a["score"].to_numpy().astype(np.float64)
    a_order = np.argsort(a_scores)[::-1][:n_a_slots]

    # Top Track B indices (in merged_df space: n_a..n_a+n_b-1)
    if tau is not None and qualified_mask is not None:
        qualified_indices = np.where(qualified_mask)[0]
        qualified_scores = b_scores[qualified_indices]
        top_qualified = qualified_indices[np.argsort(qualified_scores)[::-1][:r_actual]]
        b_order_merged = top_qualified + n_a
    else:
        b_order = np.argsort(b_scores)[::-1][:r_actual]
        b_order_merged = b_order + n_a

    top_k_indices = np.concatenate([a_order, b_order_merged])

    return merged, top_k_indices
