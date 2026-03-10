# Review: V7.0 Deployment Handoff

Reviewed file:
- [v70-deployment-handoff.md](/home/xyz/workspace/research-qianli-v2/research-miso-signal7/docs/v70-deployment-handoff.md)

## Findings

1. **High: Gap 3 rank/tier logic is not faithful to the current V6.2B reproduction contract.**

The handoff states:

```text
rank = dense_rank(rank_ori) / n
tier = qcut(rank, 5)
```

at [v70-deployment-handoff.md](/home/xyz/workspace/research-qianli-v2/research-miso-signal7/docs/v70-deployment-handoff.md#L17), but the actual helper in [v62b_formula.py](/home/xyz/workspace/research-qianli-v2/research-stage5-tier/ml/v62b_formula.py#L27) uses:

```text
rank = dense_rank(score) / max_dense_rank
```

That is: divide by the number of unique dense ranks `K`, not row count `n`.

This difference matters whenever scores tie.

Example:

```text
scores:
A 0.90
B 0.80
C 0.80
D 0.10
E 0.00

dense ranks on -score:
A 1
B 2
C 2
D 3
E 4
```

If normalized by `n = 5`:

```text
A 0.20
B 0.40
C 0.40
D 0.60
E 0.80
```

If normalized by `K = 4`:

```text
A 0.25
B 0.50
C 0.50
D 0.75
E 1.00
```

So the handoff’s formula compresses the rank scale whenever ties exist.

The sample implementation is also inconsistent with the doc’s own `qcut` description. In [v70-deployment-handoff.md](/home/xyz/workspace/research-qianli-v2/research-miso-signal7/docs/v70-deployment-handoff.md#L301), it computes:

```python
tier = np.minimum((rank * 5).astype(int), 4)
```

That is not the same as `qcut(rank, 5)`. These methods differ at cut boundaries and under ties, so Gap 3 should not be described as “trivial” until the exact output contract is pinned down and validated.

2. **High: The performance table is stale and no longer matches current stage5 artifacts.**

At [v70-deployment-handoff.md](/home/xyz/workspace/research-qianli-v2/research-miso-signal7/docs/v70-deployment-handoff.md#L39), the handoff reports:

- `f0/offpeak`: `0.2488 -> 0.3561`
- `f1/onpeak`: `0.1803 -> 0.3290`
- `f1/offpeak`: `0.2457 -> 0.3561`

Current holdout artifacts show:

- `f0/offpeak`: `0.2075 -> 0.3780`
- `f1/onpeak`: `0.2209 -> 0.3677`
- `f1/offpeak`: `0.2492 -> 0.3561`

Sources:
- [metrics.json](/home/xyz/workspace/research-qianli-v2/research-stage5-tier/holdout/f0/offpeak/v0/metrics.json)
- [metrics.json](/home/xyz/workspace/research-qianli-v2/research-stage5-tier/holdout/f0/offpeak/v10e-lag1/metrics.json)
- [metrics.json](/home/xyz/workspace/research-qianli-v2/research-stage5-tier/holdout/f1/onpeak/v0/metrics.json)
- [metrics.json](/home/xyz/workspace/research-qianli-v2/research-stage5-tier/holdout/f1/onpeak/v2/metrics.json)
- [metrics.json](/home/xyz/workspace/research-qianli-v2/research-stage5-tier/holdout/f1/offpeak/v0/metrics.json)
- [metrics.json](/home/xyz/workspace/research-qianli-v2/research-stage5-tier/holdout/f1/offpeak/v2/metrics.json)

A deployment handoff should not mix old and new evidence. The table needs to be refreshed from the live artifacts.

3. **Medium: The temporal-lag explanation for `f1` is internally inconsistent.**

The handoff says in [v70-deployment-handoff.md](/home/xyz/workspace/research-qianli-v2/research-miso-signal7/docs/v70-deployment-handoff.md#L141):

- for `f0`, binding frequency uses months `< M-1`
- for `f1`, binding frequency uses months `< M-2`

But later, the training-config table says in [v70-deployment-handoff.md](/home/xyz/workspace/research-qianli-v2/research-miso-signal7/docs/v70-deployment-handoff.md#L432):

- `f0`: BF lag `1 (months < M-1)`
- `f1`: BF lag `1 (months < M-1)`

The current stage5 implementation is auction-month keyed and uses `BF_LAG = 1` for both `f0` and `f1`; row inclusion is what shifts for `f1`, not the per-row BF lag. See:

- [collect_usable_months()](/home/xyz/workspace/research-qianli-v2/research-stage5-tier/ml/config.py#L169)
- [enrich_df()](/home/xyz/workspace/research-qianli-v2/research-stage5-tier/scripts/run_v10e_lagged.py#L105)
- [run_variant()](/home/xyz/workspace/research-qianli-v2/research-stage5-tier/scripts/run_v10e_lagged.py#L168)

The handoff needs one clock and one definition. Right now it mixes auction-month and delivery-month interpretations in a way that could easily lead to the wrong production cutoff.

4. **Medium: The handoff overstates the evaluation claim as “fully out-of-sample.”**

[v70-deployment-handoff.md](/home/xyz/workspace/research-qianli-v2/research-miso-signal7/docs/v70-deployment-handoff.md#L39) describes the 2024-2025 holdout as “fully out-of-sample.”

That is too strong. The current stage5 holdout is a walk-forward retrain: each eval month re-collects trailing usable months and re-trains a fresh model. See:

- [run_variant()](/home/xyz/workspace/research-qianli-v2/research-stage5-tier/scripts/run_v10e_lagged.py#L168)
- [train_ltr_model call](/home/xyz/workspace/research-qianli-v2/research-stage5-tier/scripts/run_v10e_lagged.py#L204)

This is still valid deployment-style evidence, but it is not a frozen untouched holdout. The wording should be corrected so the reviewer is not misled about the strength of the claim.

## What Looks Good

- The overall hybrid deployment design is correct: ML for `f0/f1`, exact V6.2B passthrough for all other period types.
- The cache preflight and inference-only loading are correctly identified as the main engineering gaps.
- The signal output contract section is directionally good and matches how `ConstraintsSignal.load_data()` / `save_data()` are used.
- Keeping shift factors as pure passthrough is the right design.

## Bottom Line

The handoff is directionally strong, but it is not ready to hand to an independent implementer yet.

The main blocker is Gap 3: the current rank/tier section can produce the wrong output if followed literally. After that, the stale performance table and mixed `f1` lag explanation should be fixed before review.
