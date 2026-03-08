# Stage 5 — Working Memory

## Status: v6b Best on VC@20, Holdout-Confirmed (2026-03-08)

Full audit: see `audit.md` (19/19 checks PASS, no leakage, no bugs).
Holdout test: see `holdout/` (2024-2025 one-time test, immutable results).

## VERIFICATION CHECKPOINTS

### Ground Truth = Realized DA, NOT shadow_price_da
- `shadow_price_da` in V6.2B parquet is a HISTORICAL 60-month lookback feature
- Ground truth = realized DA shadow prices: `abs(sum(shadow_price))` per constraint_id
- Fetched via `MisoApTools().tools.get_da_shadow_by_peaktype()`
- Cached in `data/realized_da/{YYYY-MM}.parquet` (79 months, 2019-06 to 2025-12)
- Spearman(shadow_price_da, realized_sp) = 0.22 — confirms NOT leaked

### da_rank_value Is NOT Leaky
- `da_rank_value = rank(shadow_price_da)` = rank of historical DA = legitimate signal
- Leaky features: `rank, rank_ori, tier, shadow_sign, shadow_price` — auto-stripped by LTRConfig
- `v62b_formula_score` (=rank_ori) is also NOT leaky — computed from 3 legitimate features

### Score Direction: Lower rank_value = More Binding
- V6.2B rank_values: LOWER = more binding
- evaluate_ltr: HIGHER score = better
- Formula evaluation: `scores = -v62b_score(...)` (negated)
- ML output: higher = more binding (no negation needed)

---

## Results: 36-Month Combined Eval (Realized DA Ground Truth)

| Metric | v0 (formula) | v5 (rank,12f) | v6a (reg,12f) | v6b (reg,13f) | v6c (rank,13f) |
|--------|-------------|---------------|---------------|---------------|----------------|
| VC@20 | 0.3336 | 0.3297 (-1%) | 0.3381 (+1%) | **0.3503** (+5%) | 0.3480 (+4%) |
| VC@100 | 0.6100 | 0.6191 (+1%) | 0.6166 (+1%) | 0.6241 (+2%) | **0.6313** (+3%) |
| Recall@20 | 0.2111 | 0.2375 (+13%) | 0.2278 (+8%) | 0.2375 (+13%) | **0.2389** (+13%) |
| Recall@100 | 0.2281 | 0.2514 (+10%) | **0.2597** (+14%) | 0.2586 (+13%) | 0.2511 (+10%) |
| NDCG | 0.4538 | 0.5470 (+21%) | 0.5538 (+22%) | **0.5567** (+23%) | 0.5366 (+18%) |
| Spearman | **0.1964** | 0.1902 (-3%) | 0.1705 (-13%) | 0.1712 (-13%) | 0.1881 (-4%) |

12-month subset (for reference):

| Metric | v0 (formula) | v5 (rank,12f) | v6a (reg,12f) | v6b (reg,13f) | v6c (rank,13f) |
|--------|-------------|---------------|---------------|---------------|----------------|
| VC@20 | 0.2817 | 0.2962 (+5%) | 0.3226 (+15%) | **0.3533** (+25%) | 0.3504 (+24%) |
| VC@100 | 0.6008 | 0.6051 (+1%) | 0.5808 (-3%) | 0.5953 (-1%) | **0.6247** (+4%) |

v6 legend: reg=LightGBM regression, rank=LightGBM lambdarank, 13f=12 features + formula score

## Key Findings
- **v6b best on VC@20 across 36 months** (0.3503, +5% over formula)
- **v6c best on VC@100** (0.6313, +3.5% over formula)
- **12-month eval overstated ML gains**: v6b was +25% on 12mo but only +5% on 36mo combined
- **Regression >> Ranking on VC@20**: v6a beats v5 with same features
- **Formula-as-feature helps**: v6b > v6a, v6c > v5
- **All ML variants beat formula** on Recall@20, Recall@100, NDCG
- **Spearman: formula still wins** — ML sacrifices global correlation for top-k precision
- **Tiered labels fix was massive**: v3→v4a = +36% VC@20 (same features, only label fix)
- **da_rank_value carries >90% of ranking signal**

## Bug Fixes
- **Label noise**: `_rank_transform_labels()` gave ~528 distinct labels to non-binding → fixed with `_tiered_labels()` (0/1/2/3)
- **LightGBM threading deadlock**: Container reports 64 CPUs → `num_threads=4` in all LightGBM params
- **Fork deadlock**: ProcessPoolExecutor fork copies broken LightGBM lock → removed, use sequential
- **NFS bottleneck**: In-memory `_MONTH_CACHE` in data_loader.py cuts NFS reads by ~87%

## V6.2B Formula (Verified Exact)
`rank_ori = 0.60 * da_rank_value + 0.30 * density_mix_rank_value + 0.10 * density_ori_rank_value`

## Timing
- 36-month comparison (5 variants × 24 months + v0): **28s** with cache
- 12-month eval (single variant): ~3s cached, ~10s cold NFS
- Sequential beats parallel (NFS contention + spawn overhead > compute savings)

## Ground Truth Stats
- ~68-81 V6.2B constraints match DA per month (out of ~550-780)
- ~200-250 DA binding constraints NOT in V6.2B signal universe (coverage gap)
- Average ~72 binding constraints per month (11-13% of V6.2B)

---

## Holdout Test: 2024-2025 (24 months, one-time, immutable)

Results in `holdout/{version}/metrics.json`. Dev eval period was 2020-2023; this is fully unseen.

| Metric | v0 (formula) | v5 (rank,12f) | v6b (reg,13f) | v6c (rank,13f) |
|--------|-------------|---------------|---------------|----------------|
| VC@20 | 0.1835 | 0.2160 (+18%) | **0.2709** (+48%) | 0.2100 (+14%) |
| VC@100 | 0.5924 | **0.6322** (+7%) | 0.5854 (-1%) | 0.6040 (+2%) |
| Recall@20 | 0.1500 | 0.1854 (+24%) | **0.1937** (+29%) | 0.1792 (+19%) |
| Recall@100 | 0.2421 | **0.2571** (+6%) | 0.2525 (+4%) | 0.2554 (+5%) |
| NDCG | 0.4224 | **0.4580** (+8%) | 0.4462 (+6%) | 0.4550 (+8%) |
| Spearman | **0.1946** | 0.1834 (-6%) | 0.1891 (-3%) | 0.1789 (-8%) |

### Holdout Key Findings
- **v6b VC@20 advantage holds on holdout**: +48% over formula (vs +5% on dev 36mo)
- **All metrics lower than dev eval** — expected: 2024-2025 is harder (more constraints per month)
- **v5 best on VC@100, NDCG, Recall@100** — ranking objective generalizes better for broader coverage
- **v6b best on VC@20 and Recall@20** — regression still best for top-k precision
- **Spearman: formula still wins** — consistent with dev eval finding
- **v6c underperforms on holdout** — formula-as-feature helps regression more than ranking on new data

---

## What's Next
- Investigate 5-tier labels (0-4) vs current 4-tier (0-3)
- Try different training window lengths (currently 8mo — try 12, 16)
- Multi-period extension (f1, f2, f3)
- Hybrid approach: regression for top-20, ranking for top-100
