"""Controlled experiment: separate upstream data mismatch from LightGBM nondeterminism.

Verification plan (from user):
  1. Freeze one golden training slice (2025-06/aq1/onpeak/R1)
  2. Materialize the exact training matrix, save to disk
  3. Hash X, labels, groups, feature names
  4. Train twice same-process on frozen matrix — compare predictions
  5. Train in fresh process from saved matrix — compare to step 4
  6. Repeat with num_threads=1

Acceptance rule:
  - frozen matrix hashes identical across runs → upstream is stable
  - same-process training identical → model is deterministic given same input
  - num_threads=1 collapses any cross-process drift → multi-thread summation order
  - drift ONLY in cross-process + multi-thread → LightGBM thread nondeterminism

Usage:
    # First run: builds + saves frozen matrix, trains twice, reports
    source /home/xyz/workspace/pmodel/.venv/bin/activate
    PYTHONPATH=. python scripts/verify_base_nondeterminism.py

    # Second run: loads saved matrix (no rebuild), trains, compares to first run
    PYTHONPATH=. python scripts/verify_base_nondeterminism.py

    # Third run: num_threads=1 tiebreaker
    PYTHONPATH=. python scripts/verify_base_nondeterminism.py --threads 1
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import time

os.environ.setdefault("RAY_ADDRESS", "ray://10.8.0.36:10001")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import lightgbm as lgb
import numpy as np

FROZEN_DIR = "/opt/tmp/qianli/miso/trash/annual_candidate_7.2b/frozen_matrix"
CHECKPOINT_PATH = "/opt/tmp/qianli/miso/trash/annual_candidate_7.2b/checkpoints.json"

# Golden slice
EVAL_PY = "2025-06"
AQ = "aq1"
CT = "onpeak"
ROUND = 1


def hash_array(arr: np.ndarray) -> str:
    return hashlib.sha256(arr.tobytes()).hexdigest()[:16]


def build_and_save_frozen_matrix():
    """Build training + eval matrices from production path, save to disk."""
    from pbase.config.ray import init_ray
    import pmodel
    init_ray(extra_modules=[pmodel])

    import polars as pl
    from ml.markets.miso.release_candidate import (
        BASE_FEATURES, build_eval_table,
        assign_bucket_labels, assign_bucket_weights,
        ALL_PYS, AQS_FULL,
    )
    from ml.markets.miso.config import CLASS_BF_COL, CROSS_CLASS_BF_COL

    TRAIN_PYS = [py for py in ALL_PYS if py < EVAL_PY]

    # Build eval table
    print("Building eval table...")
    eq = build_eval_table(EVAL_PY, AQ, CT, ROUND)
    base_feats = [f for f in BASE_FEATURES if f in eq.columns]

    X_eval = eq.select(base_feats).to_numpy().astype(np.float64)
    branches = eq["branch"].to_list()

    # Build training data
    print("Building training data...")
    train_frames = []
    for tpy in TRAIN_PYS:
        for taq in AQS_FULL:
            t = build_eval_table(tpy, taq, CT, ROUND)
            t = t.with_columns(
                pl.lit(tpy).alias("py"), pl.lit(taq).alias("aq_label"),
            )
            train_frames.append(t)

    train_all = pl.concat(train_frames, how="diagonal")

    X_train = train_all.select(base_feats).to_numpy().astype(np.float64)
    sp_t = train_all["sp"].to_numpy().astype(np.float64)
    labels = assign_bucket_labels(sp_t)
    weights = assign_bucket_weights(labels)
    groups = train_all.group_by(
        ["py", "aq_label"], maintain_order=True
    ).len()["len"].to_numpy()

    import ray
    ray.shutdown()

    # Save
    os.makedirs(FROZEN_DIR, exist_ok=True)
    np.save(f"{FROZEN_DIR}/X_train.npy", X_train)
    np.save(f"{FROZEN_DIR}/X_eval.npy", X_eval)
    np.save(f"{FROZEN_DIR}/labels.npy", labels)
    np.save(f"{FROZEN_DIR}/weights.npy", weights)
    np.save(f"{FROZEN_DIR}/groups.npy", groups)
    with open(f"{FROZEN_DIR}/feature_names.json", "w") as f:
        json.dump(base_feats, f)
    with open(f"{FROZEN_DIR}/branches.json", "w") as f:
        json.dump(branches, f)

    print(f"Saved frozen matrix to {FROZEN_DIR}")
    print(f"  X_train: {X_train.shape}, X_eval: {X_eval.shape}")
    print(f"  labels: {labels.shape}, groups: {groups.shape}")

    return X_train, X_eval, labels, weights, groups, base_feats, branches


def load_frozen_matrix():
    """Load previously saved frozen matrix."""
    X_train = np.load(f"{FROZEN_DIR}/X_train.npy")
    X_eval = np.load(f"{FROZEN_DIR}/X_eval.npy")
    labels = np.load(f"{FROZEN_DIR}/labels.npy")
    weights = np.load(f"{FROZEN_DIR}/weights.npy")
    groups = np.load(f"{FROZEN_DIR}/groups.npy")
    with open(f"{FROZEN_DIR}/feature_names.json") as f:
        base_feats = json.load(f)
    with open(f"{FROZEN_DIR}/branches.json") as f:
        branches = json.load(f)

    print(f"Loaded frozen matrix from {FROZEN_DIR}")
    print(f"  X_train: {X_train.shape}, X_eval: {X_eval.shape}")

    return X_train, X_eval, labels, weights, groups, base_feats, branches


def train_and_predict(X_train, labels, groups, weights, base_feats, X_eval, num_threads=4):
    """Train base model and return predictions on eval set."""
    from ml.markets.miso.release_candidate import BASE_LGB, BASE_BOOST_ROUNDS

    params = dict(BASE_LGB)
    params["num_threads"] = num_threads

    ds = lgb.Dataset(
        X_train, label=labels, group=groups, weight=weights,
        feature_name=base_feats, free_raw_data=False,
    )
    model = lgb.train(params, ds, num_boost_round=BASE_BOOST_ROUNDS)
    preds = model.predict(X_eval)
    return preds


def compare_predictions(name_a, preds_a, name_b, preds_b, branches):
    """Compare two prediction vectors."""
    exact = np.array_equal(preds_a, preds_b)
    max_diff = float(np.max(np.abs(preds_a - preds_b)))
    mean_diff = float(np.mean(np.abs(preds_a - preds_b)))
    corr = float(np.corrcoef(preds_a, preds_b)[0, 1])

    order_a = np.argsort(preds_a)[::-1]
    order_b = np.argsort(preds_b)[::-1]
    top5_a = [branches[order_a[i]] for i in range(5)]
    top5_b = [branches[order_b[i]] for i in range(5)]
    top20_a = set(branches[order_a[i]] for i in range(20))
    top20_b = set(branches[order_b[i]] for i in range(20))

    print(f"\n  {name_a} vs {name_b}:")
    print(f"    exact match:     {exact}")
    print(f"    max abs diff:    {max_diff:.10f}")
    print(f"    mean abs diff:   {mean_diff:.10f}")
    print(f"    correlation:     {corr:.10f}")
    print(f"    top-5 A:         {top5_a}")
    print(f"    top-5 B:         {top5_b}")
    print(f"    top-5 match:     {top5_a == top5_b}")
    print(f"    top-20 overlap:  {len(top20_a & top20_b)}/20")

    return {
        "exact": exact, "max_diff": max_diff, "mean_diff": mean_diff,
        "corr": corr, "top5_match": top5_a == top5_b,
        "top20_overlap": len(top20_a & top20_b),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--threads", type=int, default=4, help="num_threads for LightGBM (default 4, use 1 for tiebreaker)")
    parser.add_argument("--rebuild", action="store_true", help="Force rebuild of frozen matrix")
    args = parser.parse_args()

    t0 = time.time()
    num_threads = args.threads

    # Step 1-2: Build or load frozen matrix
    if not os.path.exists(f"{FROZEN_DIR}/X_train.npy") or args.rebuild:
        print("=" * 60)
        print("  STEP 1-2: BUILD + SAVE FROZEN MATRIX")
        print("=" * 60)
        X_train, X_eval, labels, weights, groups, base_feats, branches = build_and_save_frozen_matrix()
    else:
        print("=" * 60)
        print("  STEP 1-2: LOAD EXISTING FROZEN MATRIX")
        print("=" * 60)
        X_train, X_eval, labels, weights, groups, base_feats, branches = load_frozen_matrix()

    # Step 3: Hash
    print("\n" + "=" * 60)
    print("  STEP 3: MATRIX HASHES")
    print("=" * 60)
    h_X = hash_array(X_train)
    h_labels = hash_array(labels)
    h_groups = hash_array(groups)
    h_weights = hash_array(weights)
    h_feats = hashlib.sha256(json.dumps(base_feats).encode()).hexdigest()[:16]

    hashes = {
        "X_train": h_X, "labels": h_labels, "groups": h_groups,
        "weights": h_weights, "features": h_feats,
    }
    for k, v in hashes.items():
        print(f"  {k}: {v}")

    # Save hashes for cross-run comparison
    hash_file = f"{FROZEN_DIR}/hashes.json"
    hash_match = True
    if os.path.exists(hash_file):
        with open(hash_file) as f:
            prev_hashes = json.load(f)
        hash_match = all(prev_hashes.get(k) == v for k, v in hashes.items())
        print(f"\n  Cross-run hash match: {hash_match}")
        if not hash_match:
            for k in hashes:
                if prev_hashes.get(k) != hashes[k]:
                    print(f"    MISMATCH: {k}: prev={prev_hashes.get(k)} now={hashes[k]}")
            print("  STOP: upstream data mismatch detected. Not LightGBM nondeterminism.")
            return
    with open(hash_file, "w") as f:
        json.dump(hashes, f)

    # Step 4: Train twice same-process
    print("\n" + "=" * 60)
    print(f"  STEP 4: TRAIN TWICE SAME-PROCESS (num_threads={num_threads})")
    print("=" * 60)
    print("  Training run A...")
    preds_a = train_and_predict(X_train, labels, groups, weights, base_feats, X_eval, num_threads=num_threads)
    print("  Training run B...")
    preds_b = train_and_predict(X_train, labels, groups, weights, base_feats, X_eval, num_threads=num_threads)

    same_proc = compare_predictions("run_A", preds_a, "run_B", preds_b, branches)

    # Save run A predictions for cross-process comparison
    pred_file = f"{FROZEN_DIR}/preds_threads{num_threads}.npy"
    if os.path.exists(pred_file):
        print("\n" + "=" * 60)
        print(f"  STEP 5: CROSS-PROCESS COMPARISON (num_threads={num_threads})")
        print("=" * 60)
        preds_prev = np.load(pred_file)
        cross_proc = compare_predictions("prev_process", preds_prev, "this_process_A", preds_a, branches)
    else:
        cross_proc = None

    np.save(pred_file, preds_a)
    print(f"\n  Saved predictions to {pred_file}")
    print(f"  Re-run this script to test cross-process comparison.")

    # Also compare against research checkpoint if available
    if os.path.exists(CHECKPOINT_PATH):
        with open(CHECKPOINT_PATH) as f:
            cp = json.load(f)
        print("\n" + "=" * 60)
        print("  RESEARCH CHECKPOINT COMPARISON")
        print("=" * 60)
        print(f"  run_A mean: {preds_a.mean():.6f}  checkpoint: {cp['CP3_base_score_mean']:.6f}  diff: {preds_a.mean() - cp['CP3_base_score_mean']:+.6f}")
        print(f"  run_A min:  {preds_a.min():.6f}  checkpoint: {cp['CP3_base_score_min']:.6f}  diff: {preds_a.min() - cp['CP3_base_score_min']:+.6f}")
        print(f"  run_A max:  {preds_a.max():.6f}  checkpoint: {cp['CP3_base_score_max']:.6f}  diff: {preds_a.max() - cp['CP3_base_score_max']:+.6f}")

    # Summary
    print("\n" + "=" * 60)
    print("  SUMMARY")
    print("=" * 60)
    print(f"  Matrix hashes stable:          {'YES' if hash_match else 'NO'}")
    print(f"  Same-process exact:            {same_proc['exact']}")
    print(f"  Same-process max diff:         {same_proc['max_diff']:.10f}")
    print(f"  Same-process top-5 match:      {same_proc['top5_match']}")
    if cross_proc:
        print(f"  Cross-process exact:           {cross_proc['exact']}")
        print(f"  Cross-process max diff:        {cross_proc['max_diff']:.10f}")
        print(f"  Cross-process top-5 match:     {cross_proc['top5_match']}")
        print(f"  Cross-process top-20 overlap:  {cross_proc['top20_overlap']}/20")
    print(f"  num_threads:                   {num_threads}")
    print(f"  Time: {time.time()-t0:.0f}s")

    if same_proc['exact'] and (cross_proc is None or cross_proc['exact']):
        print("\n  VERDICT: Fully deterministic at num_threads={num_threads}")
    elif same_proc['exact'] and cross_proc and not cross_proc['exact']:
        print(f"\n  VERDICT: Same-process deterministic, cross-process drift.")
        print(f"  Re-run with --threads 1 to confirm thread-order cause.")
    elif not same_proc['exact']:
        print(f"\n  VERDICT: Same-process NOT deterministic. Investigate further.")


if __name__ == "__main__":
    main()
