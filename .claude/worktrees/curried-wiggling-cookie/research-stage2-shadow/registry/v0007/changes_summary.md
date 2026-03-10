# v0007 Changes Summary — Iteration 3 (batch ralph-v2-20260304-031811)

## Hypothesis Winner: B (MCW-Only: reg_lambda=1.0, mcw=25)

### Screening Results (2-month)

**Hypothesis A** (reg_lambda=3.0, mcw=10):

| Metric | 2021-11 | 2022-12 |
|--------|---------|---------|
| EV-VC@100 | 0.0215 | 0.1985 |
| EV-VC@500 | 0.1859 | 0.3511 |
| EV-NDCG | 0.7459 | 0.8167 |
| Spearman | 0.2623 | 0.3859 |
| C-RMSE | 4744 | 2860 |

**Hypothesis B** (reg_lambda=1.0, mcw=25):

| Metric | 2021-11 | 2022-12 |
|--------|---------|---------|
| EV-VC@100 | 0.0392 | 0.1922 |
| EV-VC@500 | 0.1996 | 0.3497 |
| EV-NDCG | 0.7498 | 0.8148 |
| Spearman | 0.2625 | 0.3866 |
| C-RMSE | 4746 | 2961 |

**Winner selection**: Spearman difference was < 0.002 (0.3246 vs 0.3241 mean), triggering tiebreak on EV-VC@100. Hypothesis B won tiebreak with higher mean EV-VC@100 (0.1157 vs 0.1100). Hypothesis A had catastrophically low EV-VC@100 on 2021-11 (0.0215 vs B's 0.0392).

### Key Insight

The v0005 Spearman compression was caused by **reg_lambda (L2)**, not by **min_child_weight (mcw)**:
- Reverting reg_lambda from 5.0 to 1.0 (v0's default) while keeping mcw=25 recovered Spearman to v0 levels
- mcw=25 independently provides EV-VC value capture benefit without harming Spearman
- L2 regularization compresses the prediction distribution, hurting rank correlation; mcw constrains leaf sizes, which is orthogonal to rank correlation

### Code Changes

1. `ml/config.py`: `RegressorConfig.reg_lambda` default changed from `5.0` to `1.0`
2. `ml/tests/test_config.py`: Updated test assertions to match actual code defaults (14 clf features, 34 reg features, train_months=6, reg_lambda=1.0; removed incorrect frozen-dataclass test)
3. `ml/tests/test_features.py`: Fixed `_make_sample_df` to include both v0 clf and v1 regressor features; updated expected shapes from (3,13)/(3,24) to match actual feature counts
4. `ml/tests/test_data_loader.py`: Updated expected feature count from 24 to 34

### Full 12-Month Results (v0007)

| Metric | Mean | Bot-2 |
|--------|------|-------|
| EV-VC@100 | 0.0699 | 0.0071 |
| EV-VC@500 | 0.2294 | 0.0662 |
| EV-NDCG | 0.7513 | 0.6502 |
| Spearman | 0.3932 | 0.2669 |
| C-RMSE | 2916 | 5370 |

### Comparison vs Champion (v0)

| Metric | v0007 Mean | v0 Mean (Floor) | Delta |
|--------|-----------|----------------|-------|
| EV-VC@100 | 0.0699 | 0.0690 | **+1.3%** |
| EV-VC@500 | 0.2294 | 0.2160 | **+6.2%** |
| EV-NDCG | 0.7513 | 0.7472 | **+0.5%** |
| Spearman | 0.3932 | 0.3928 | **+0.1%** |

All 4 hard-gate metrics beat v0 champion mean.

### Comparison vs v0005 (reg_lambda=5.0, mcw=25)

| Metric | v0007 Mean | v0005 Mean | Delta |
|--------|-----------|-----------|-------|
| EV-VC@100 | 0.0699 | 0.0735 | -4.9% |
| EV-VC@500 | 0.2294 | 0.2287 | +0.3% |
| EV-NDCG | 0.7513 | 0.7501 | +0.2% |
| Spearman | 0.3932 | 0.3920 | **+0.3%** |

v0007 trades some EV-VC@100 vs v0005 (less L2 regularization → less overfitting protection at top-100) but recovers Spearman above the v0 gate floor.

### Config Provenance

Verified `registry/v0007/config.json` contains:
- `regressor.reg_lambda = 1.0` (intended)
- `regressor.min_child_weight = 25` (intended)
