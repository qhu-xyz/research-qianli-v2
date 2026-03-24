# Bucket_6_20 Model Report (2026-03-24)

## What is Bucket_6_20?

A single unified LambdaRank model trained on ALL branches with 5-tier severity labels and aggressive weighting on dangerous binders.

**Bound feature recipe**: `miso_annual_bucket_features_v1`

| Tier | Condition | Weight | Meaning |
|------|-----------|--------|---------|
| 0 | SP = 0 | 1 | Non-binder |
| 1 | 0 < SP ≤ $200 | 1 | Trivial binder |
| 2 | $200 < SP ≤ $5K | 2 | Moderate binder |
| 3 | $5K < SP ≤ $20K | 6 | Significant binder |
| 4 | SP > $20K | 20 | Dangerous binder |

**Features** (13): da_rank_value, shadow_price_da, bf (class-specific), count_active_cids, bin_80/90/100/110_cid_max, rt_max, top2_bin_80/90/100/110.

### Bound Feature Recipe: `miso_annual_bucket_features_v1`

This model is only well-defined together with the following feature recipe.

**Universe scope**:
- full annual branch universe
- separate training/eval per `class_type` (`onpeak`, `offpeak`)

**Feature columns**:
- `da_rank_value`
- `shadow_price_da`
- `bf` (class-specific alias from `bf_12` or `bfo_12`)
- `count_active_cids`
- `bin_80_max`
- `bin_90_max`
- `bin_100_max`
- `bin_110_max`
- `rt_max`
- `top2_bin_80`
- `top2_bin_90`
- `top2_bin_100`
- `top2_bin_110`

**Transforms / derivations**:
- `bf` is aliased from the class-specific BF column
- `bin_80/90/100/110_max` are aliases of `bin_*_cid_max`
- `rt_max = max(bin_80_max, bin_90_max, bin_100_max, bin_110_max)`
- `top2_bin_*` are joined from the cached top2 feature tables

**Fill rules**:
- `top2_bin_*` nulls are filled with `0.0`
- all features are loaded from cached class-specific tables and must remain ex-ante valid

**Current round behavior**:
- v1 is round-insensitive because current cached annual model tables are built from `market_round=1`
- a future `miso_annual_bucket_features_v2` should represent the round-aware version

**Training**: Class-specific tables from `build_class_model_table`, 2018-2025 expanding window, all branches (not just dormant).

**How it differs from previous models**:
- vs **v0c** (formula): ML model, not formula. Learns nonlinear feature interactions.
- vs **Opt3** (tiered tertile): 5 tiers with fixed SP boundaries instead of relative tertiles. 20x weight on dangerous tier vs 10x.
- vs **B_dang** (binary): Preserves ordering within binders. Binary threw away all information below threshold.
- vs **V4.4**: Fully reproducible. No SCADA deviation features. Larger universe (2,700 vs 1,200 branches).

## Results: Bucket_6_20 vs V4.4 — Native Top-K

Each model picks its own top-K from its own universe. SP and NB_SP are realized DA shadow price.

### K=200 (Tier 0) — Bucket_6_20 wins SP AND NB_SP in recent years

| Year | Ctype | Bkt SP | V4.4 SP | Delta SP | Bkt NB_SP | V4.4 NB_SP | Delta NB_SP |
|------|-------|---:|---:|---:|---:|---:|---:|
| 2022 | onpeak | $793K | $623K | +$171K | $19K | $32K | -$13K |
| 2022 | offpeak | $853K | $654K | +$200K | $65K | $54K | +$11K |
| 2023 | onpeak | $752K | $613K | +$139K | $43K | $94K | -$51K |
| 2023 | offpeak | $822K | $654K | +$167K | $29K | $69K | -$40K |
| 2024 | onpeak | $565K | $424K | +$141K | $10K | $3K | +$7K |
| 2024 | offpeak | $573K | $452K | +$121K | $6K | $3K | +$4K |
| 2025 | onpeak | $691K | $421K | +$270K | $31K | $24K | +$8K |
| 2025 | offpeak | $665K | $459K | +$205K | $29K | $26K | +$3K |

**Bucket_6_20 wins total SP in all 8 cells** (+$121K to +$270K).
**Bucket_6_20 wins NB_SP in 5/8 cells** (2022 offpeak, 2024 both, 2025 both). V4.4 wins NB_SP in 2022 onpeak and 2023 both.

### K=400 (Tier 0+1) — Bucket_6_20 wins SP, V4.4 wins NB_SP

| Year | Ctype | Bkt SP | V4.4 SP | Delta SP | Bkt NB_SP | V4.4 NB_SP | Delta NB_SP |
|------|-------|---:|---:|---:|---:|---:|---:|
| 2022 | onpeak | $1,009K | $995K | +$14K | $48K | $114K | -$66K |
| 2022 | offpeak | $1,042K | $953K | +$89K | $94K | $104K | -$9K |
| 2023 | onpeak | $1,014K | $779K | +$235K | $188K | $156K | +$32K |
| 2023 | offpeak | $962K | $833K | +$128K | $114K | $134K | -$20K |
| 2024 | onpeak | $683K | $565K | +$118K | $34K | $35K | -$1K |
| 2024 | offpeak | $640K | $552K | +$88K | $14K | $16K | -$2K |
| 2025 | onpeak | $817K | $703K | +$114K | $52K | $83K | -$30K |
| 2025 | offpeak | $853K | $700K | +$152K | $76K | $89K | -$12K |

**Bucket_6_20 wins total SP in all 8 cells** (+$14K to +$235K).
**V4.4 wins NB_SP in 7/8 cells** at K=400 (exception: 2023 onpeak +$32K). V4.4 packs ~20 dormant branches into its top-400 vs Bucket_6_20's ~12, sacrificing $14-235K of overall SP.

## Feature Importance (offpeak, 2025 eval window only)

These numbers are from the last trained model (offpeak, 2025 holdout window). Onpeak and other eval windows may differ.

| Feature | Weight |
|---------|---:|
| da_rank_value | 42.6% |
| shadow_price_da | 33.0% |
| bf (class-specific) | 5.1% |
| density bins + top2 | 15.0% |
| count_active_cids | 2.2% |
| rt_max | 1.3% |

History dominates (81%) in this window. Density contributes modestly (15%).

## How Bucket_6_20 evolved

| Stage | Model | Key change | Result |
|-------|-------|-----------|--------|
| V2 baseline | Per-ctype NB models | 2 models, 8 features, V4.4 benchmark | Baseline established |
| V3 ablation | 9 variants | +2020, +tiered_wt, +top2_mean | tiered_wt + top2 = best combo |
| Opt3 | Unified LambdaRank | Single model all branches, tiered [1,1,3,10] | Beats V4.4 on SP everywhere |
| Feature ablation | Density expansion | Tail probs, expected excess, extreme bins | All negative onpeak |
| Top-tail variants | A_extreme, B_dang, C_log1p | Heavier NB emphasis | B_dang best NB but costs SP |
| **Bucket_6_20** | **Danger-aware buckets** | **5-tier [1,1,2,6,20] on all branches** | **Best SP + improved NB** |

## Rank Type Clarification

All SP capture tables in this report use **`rank_native`** — each model's rank within its own full universe. Bucket_6_20 ranks out of ~2,700 branches, V4.4 out of ~1,200. These are the ranks that determine production top-K selection.

Earlier overlap-only analyses (reranking both models on the same shared branch set) showed Bucket_6_20 winning on `rank_overlap`. Those results are valid as a ranking-quality diagnostic but use a different denominator and should not be compared against the `rank_native` numbers in this report.

See `docs/metric-contract.md` for the full naming convention.

## Limitations

1. **V4.4 wins NB_SP at K=400** in most years — it packs 20+ dormant branches into top-400 by sacrificing overall SP.
2. **Top-5 NB binders still rank 100-800+** in all models — fundamental data limitation.
3. **2022-2023 NB_SP gap at K=200**: V4.4 captures more NB_SP in older years. Unclear why — possibly V4.4's deviation features were more predictive historically.
4. **Bucket boundaries are not tuned per ctype** — onpeak and offpeak have different SP distributions.

## Status: Candidate (not yet champion)

Bucket_6_20 beats V4.4 on total SP in all 16 cells. However:

1. **Not yet compared side-by-side with v0c or Opt3** in the same script. Earlier V3 reports suggest v0c may win some K=200 cells on total SP. A direct 3-way comparison (v0c vs Bucket_6_20 vs V4.4) is needed before declaring champion.

2. **V4.4 numbers are conservative**: V4.4 picks outside our universe (~5-10 per quarter) get zero credit because we can't resolve their GT. V4.4's true native SP could be slightly higher. Label coverage should be reported per cell.

3. **No deployment-style evaluation yet** (R30/R50 reserved slots, production shortlist logic).

4. **Feature recipe is now explicitly bound**: future comparisons and promotions should refer to `miso_annual_bucket_features_v1`, not just the informal column list.

**Next steps to confirm**:
- Run v0c + Bucket_6_20 + V4.4 in one comparison script
- Add V4.4 label coverage per cell
- Run R30/R50 deployment eval

## Artifacts

| Path | Contents |
|------|----------|
| `scripts/nb_bucket_model.py` | Training + evaluation + registry save |
| `registry/onpeak/bucket_6_20/config.json` | Model config (features, buckets, weights, params) |
| `registry/onpeak/bucket_6_20/metrics.json` | Per-(eval_py, aq) results vs V4.4 |
| `registry/offpeak/bucket_6_20/config.json` | Same for offpeak |
| `registry/offpeak/bucket_6_20/metrics.json` | Same |
