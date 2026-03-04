# v0012 Changes Summary — Iteration 2

## Screening Winner: Hypothesis A (n_estimators=600, learning_rate=0.03)

### Why Hypothesis A Won

Both hypotheses were technically vetoed on EV-VC@100 (>15% drop on 2022-12 vs champion), but per direction: pick the less-bad one. Hypothesis A won on the primary criterion (higher mean EV-VC@500 across screen months):

| Metric | Month | Champion (v0011) | Hyp A (600t, lr=0.03) | Hyp B (col=0.9, 500t, lr=0.04) |
|--------|-------|------------------|-----------------------|--------------------------------|
| EV-VC@500 | 2022-09 | 0.0527 | 0.0720 (+36.7%) | 0.0544 (+3.2%) |
| EV-VC@500 | 2022-12 | 0.3458 | 0.3443 (-0.4%) | 0.3441 (-0.5%) |
| **Mean EV-VC@500** | — | 0.1992 | **0.2082** | 0.1992 |
| EV-VC@100 | 2022-09 | 0.0280 | 0.0316 (+12.9%) | 0.0297 (+6.2%) |
| EV-VC@100 | 2022-12 | 0.1581 | 0.1236 (-21.8%) | 0.1101 (-30.3%) |
| Spearman | 2022-09 | 0.3284 | 0.3304 (+0.006) | 0.3283 (-0.001) |
| Spearman | 2022-12 | 0.3863 | 0.3866 (+0.003) | 0.3809 (-0.054) |

Hypothesis A showed a massive +36.7% lift on the critical tail-failure month (2022-09 EV-VC@500: 0.0527→0.0720, well above tail_floor 0.0536). Hypothesis B barely moved the needle (+3.2%) and also failed the Spearman veto (-0.054 on 2022-12).

### Code Changes

**`ml/config.py`** — RegressorConfig defaults:
- `n_estimators`: 400 → 600
- `learning_rate`: 0.05 → 0.03

**`ml/tests/test_config.py`** — Updated test assertions to match new defaults.

No feature changes (34 features frozen per batch constraint).

### Full 12-Month Results (v0012 vs v0011 champion)

| Metric | v0012 | v0011 | Delta |
|--------|-------|-------|-------|
| **EV-VC@100** | 0.0758 | 0.0801 | **-5.3%** |
| **EV-VC@500** | 0.2348 | 0.2270 | **+3.5%** |
| **EV-NDCG** | 0.7518 | 0.7499 | **+0.2%** |
| **Spearman** | 0.3940 | 0.3925 | **+0.4%** |
| C-RMSE | 2855.26 | 2866.62 | -0.4% |
| C-MAE | 1135.00 | 1142.52 | -0.7% |

### Key Observations

1. **EV-VC@500 recovered**: +3.5% mean improvement, achieving the primary objective of breadth recovery.
2. **2022-09 tail failure eliminated**: EV-VC@500 lifted from 0.0527 to 0.0720 (+36.7%), clearing the tail_floor of 0.0536 with substantial margin.
3. **EV-VC@100 traded down**: -5.3% mean, concentrated in 2022-12 (-21.8%) and 2021-05 (-81%, but from a tiny base of 0.0034). The large EV-VC@100 margin to floor (+20.6%) absorbs this.
4. **Spearman slightly improved**: +0.4%, consistent with learning #2 (L2=1.0 is optimal for Spearman).
5. **Calibration improved**: C-RMSE -0.4%, C-MAE -0.7% — more trees with lower LR produced better-calibrated predictions.
6. **2021-05 EV-VC@500 outlier**: +65.2% improvement from a low base (0.0556→0.0918), suggesting the extra ensemble capacity helps on difficult months.
