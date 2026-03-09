# V10e-lag1: Production-Safe Binding Frequency (1-Month Lag)

## The Problem: Temporal Leakage in v9-v10e

All previous binding_freq versions (v9, v10, v10e) had a timing assumption bug:

For f0 (front-month), auction month M (e.g., March 2025):
- **Signal is submitted ~mid of M-1** (mid-February)
- At submission time, we know realized DA through **M-2** (January — complete)
- We do NOT know M-1's full realized DA (February — only ~12 days)

But the code computed `binding_freq` using all months < M, which includes M-1.
This means the model saw data it wouldn't have at decision time.

The leakage affected TWO layers:
1. **Training data**: included month M-1 as a training row (with realized_sp label not yet available)
2. **Binding freq features**: used M-1's realized DA for both training and test bf computation

## The Fix: 1-Month Production Lag

For eval month M with lag=1:

| Component | Leaky (v10e) | Production-safe (v10e-lag1) |
|-----------|-------------|----------------------------|
| Training months | M-8..M-1 | M-9..M-2 |
| Training bf for month T | months < T | months < T-1 |
| Test bf for month M | months < M | months < M-1 |
| Test label | M realized_sp | M realized_sp (unchanged) |

### Concrete example: eval month 2025-03 (March f0, submitted mid-Feb)

**Training**: months 2024-06 through 2025-01 (not 2025-02)

**Training bf for month 2025-01** (Jan, submitted mid-Dec):
- bf_1 = "bound in Nov 2024?" (months < 2024-12, last 1)
- bf_3 = "how often in Sep-Oct-Nov 2024?" (months < 2024-12, last 3)
- bf_6 = "how often in Jun..Nov 2024?" (months < 2024-12, last 6)

**Test bf for month 2025-03** (Mar, submitted mid-Feb):
- bf_1 = "bound in Jan 2025?" (months < 2025-02, last 1)
- bf_3 = "how often in Nov-Dec-Jan?" (months < 2025-02, last 3)
- bf_6 = "how often in Aug..Jan?" (months < 2025-02, last 6)

**2025-02 data is NOT used anywhere** — not as training label, not as training feature, not as test feature.

## Results

### Dev (36 months, 2020-06 to 2023-05)

| Metric | v0 (formula) | v10e (leaky) | v10e-lag1 (safe) | lag1 vs v0 |
|--------|-------------|-------------|-----------------|-----------|
| VC@20 | 0.2817 | 0.4536 | 0.4137 | +47% |
| VC@50 | 0.4653 | 0.6543 | 0.5631 | +21% |
| VC@100 | 0.6008 | 0.7838 | 0.7195 | +20% |
| Recall@20 | 0.1833 | 0.3597 | 0.3278 | +79% |
| NDCG | 0.4423 | 0.6231 | 0.5837 | +32% |
| Spearman | 0.2045 | 0.3482 | 0.2989 | +46% |

### Holdout (24 months, 2024-2025)

| Metric | v0 (formula) | v10e (leaky) | v10e-lag1 (safe) | lag1 vs v0 |
|--------|-------------|-------------|-----------------|-----------|
| VC@20 | 0.1835 | 0.4230 | 0.3529 | +92% |
| VC@50 | 0.3947 | 0.6027 | 0.5442 | +38% |
| VC@100 | 0.5924 | 0.7608 | 0.6807 | +15% |
| Recall@20 | 0.1500 | 0.3792 | 0.3021 | +101% |
| NDCG | 0.4224 | 0.5818 | 0.5497 | +30% |
| Spearman | 0.1946 | 0.3616 | 0.3226 | +66% |

### Leakage cost: v10e-lag1 vs v10e (no lag)

| Metric | Dev | Holdout |
|--------|-----|---------|
| VC@20 | -8.8% | -16.6% |
| VC@50 | -13.9% | -9.7% |
| VC@100 | -8.2% | -10.5% |
| Recall@20 | -8.9% | -20.3% |
| NDCG | -6.3% | -5.5% |
| Spearman | -14.2% | -10.8% |

The lag costs 6-20% depending on metric, but the signal is still very real.

### Feature importance shifted with lag

| Feature | v10e (no lag) | v10e-lag1 |
|---------|-------------|-----------|
| binding_freq_12 | ~12% | 36.4% |
| v7_formula_score | ~6% | 19.4% |
| binding_freq_15 | ~4% | 16.3% |
| da_rank_value | ~2% | 10.0% |
| binding_freq_6 | ~8% | 9.0% |
| binding_freq_1 | ~44% | 2.9% |
| binding_freq_3 | ~8% | 2.7% |
| prob_exceed_110 | ~2% | 2.3% |
| constraint_limit | ~1% | 0.9% |

bf_1 collapses from 44% to 3% — makes sense, "2 months ago" is much weaker than "last month".
Longer windows (bf_12, bf_15) and the formula become more important.

## Verdict

The production-safe v10e-lag1 is the correct version for deployment. The binding_freq
signal is legitimate and large (+47-92% VC@20 over formula), even after fixing the
temporal leakage. All previous v9-v10e results were inflated by 6-20%.
