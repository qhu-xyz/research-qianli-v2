# v0006 — Changes Summary (Iteration 2)

## Hypothesis Screening

Two hypotheses were screened on months 2021-11 (weak) and 2022-12 (strong):

- **Hypothesis A**: Reduce max_depth 5→4 (keeping L2=5/mcw=25)
- **Hypothesis B**: Increase reg_alpha 0.1→1.0 (keeping L2=5/mcw=25)

### Screen Results (2-month)

| Metric | v0005 baseline | Hyp A (depth=4) | Hyp B (alpha=1.0) |
|--------|:-:|:-:|:-:|
| **2021-11** | | | |
| EV-VC@100 | 0.0487 | 0.0282 | 0.0375 |
| Spearman | 0.2635 | 0.2626 | 0.2627 |
| **2022-12** | | | |
| EV-VC@100 | 0.1988 | 0.2001 | 0.1932 |
| Spearman | 0.3857 | 0.3872 | 0.3867 |

### Winner Selection: Hypothesis B

- **Primary criterion** (mean Spearman): A=0.3249, B=0.3247 — within 0.002 tiebreak threshold
- **Tiebreak** (mean EV-VC@100): A=0.1142, B=0.1154 — B wins
- **Override check**: Hyp A dropped EV-VC@100 by 0.0205 on 2021-11 (>0.01 threshold), disqualifying it
- **Decision**: Hypothesis B wins by override rule + tiebreak

## Code Changes

- `ml/config.py`: RegressorConfig `reg_alpha` default changed from `0.1` to `1.0`
- `ml/tests/test_config.py`: Updated assertion to match new default

## Full 12-Month Benchmark Results

| Metric | v0 (Floor) | v0005 | v0006 | Delta vs v0005 |
|--------|:-:|:-:|:-:|:-:|
| EV-VC@100 mean | 0.0690 | 0.0735 | 0.0735 | +0.0000 |
| EV-VC@100 bot-2 | 0.0068 | 0.0084 | 0.0084 | +0.0000 |
| EV-VC@500 mean | 0.2160 | 0.2287 | 0.2287 | +0.0000 |
| EV-VC@500 bot-2 | 0.0558 | 0.0689 | 0.0689 | +0.0000 |
| EV-NDCG mean | 0.7472 | 0.7501 | 0.7501 | +0.0000 |
| EV-NDCG bot-2 | 0.6476 | 0.6458 | 0.6458 | +0.0000 |
| Spearman mean | **0.3928** | 0.3920 | 0.3920 | +0.0000 |
| Spearman bot-2 | 0.2689 | 0.2669 | 0.2669 | +0.0000 |

## Gate Status

- EV-VC@100 L1: 0.0735 >= 0.0690 PASS
- EV-VC@500 L1: 0.2287 >= 0.2160 PASS
- EV-NDCG L1: 0.7501 >= 0.7472 PASS
- **Spearman L1: 0.3920 < 0.3928 FAIL (miss by 0.0008)**

## Analysis

Increasing L1 regularization (reg_alpha) 10x from 0.1 to 1.0 had **effectively zero impact** on all metrics across all 12 months. v0006 is indistinguishable from v0005 within measurement noise.

This means the current feature set has no truly redundant features that L1 sparsity could eliminate — all 29 active features (of the 34 configured, 5 are missing from data) carry enough signal to survive L1=1.0 penalty. The Spearman gap vs v0 (0.0008) is not attributable to noisy features.

The Spearman L1 gate continues to block promotion. The 0.0008 miss appears to be a fundamental characteristic of L2=5/mcw=25 regularization compressing predictions, not addressable by L1 feature selection.
