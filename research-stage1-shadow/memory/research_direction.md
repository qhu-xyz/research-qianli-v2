# Research Direction — Shadow Price Classification (Stage 1)

## Priority A: Aggressive Feature Engineering (PRIMARY FOCUS)

The AUC ceiling at ~0.836 with current features is NOT the problem ceiling — it's a feature ceiling.
The source data loader (`research-spice-shadow-price-pred-qianli/src/shadow_price_prediction/data_loader.py`)
produces **15+ features we don't use**. The pipeline has been operating on a narrow subset.

### Unused Feature Categories (ordered by expected impact)

**1. Shift Factor Features (network topology — completely new signal class)**
- `sf_max_abs` — peak node sensitivity to this constraint
- `sf_mean_abs` — average sensitivity across all nodes
- `sf_std` — sensitivity spread (wide = system-critical constraint)
- `sf_nonzero_frac` — constraint reach (what fraction of network is affected)
- WHY: These capture WHERE in the network a constraint sits. High-reach, high-sensitivity constraints may bind differently than isolated ones. This is fundamentally new information not in the density curve.
- MONOTONE: sf_max_abs → 1, sf_mean_abs → 1, sf_std → 0, sf_nonzero_frac → 0

**2. Constraint Metadata (structural features)**
- `is_interface` — 1 if flowgate/nomogram, 0 if line constraint
- `constraint_limit` — MW limit of the constraint (log-transformed)
- WHY: Flowgates behave differently from lines. High-limit vs low-limit constraints have different binding dynamics.
- MONOTONE: is_interface → 0, constraint_limit → 0

**3. Distribution Shape (unused density statistics)**
- `density_mean` — expected flow point (already computed, just not used)
- `density_variance` — flow uncertainty
- `density_entropy` — information content of the density
- WHY: density_mean tells you WHERE the distribution is centered. A constraint centered at 1.05 is more likely to bind than one at 0.80, BEYOND what exceedance probabilities capture.
- MONOTONE: density_mean → 1, density_variance → 0, density_entropy → 0

**4. Probability Band Features (near-boundary signal)**
- `tail_concentration` — prob_exceed_100 / prob_exceed_80 (how peaked the tail is)
- `prob_band_95_100` — mass in the near-binding band
- `prob_band_100_105` — mass in the mild-overload band
- WHY: Two constraints with the same prob_exceed_100 can have very different profiles — one with mass concentrated at 100-105, another spread to 130. Band features discriminate these.
- MONOTONE: tail_concentration → 1, prob_band_95_100 → 1, prob_band_100_105 → 1

**5. Temporal Features**
- `forecast_horizon` — months between auction and market
- WHY: Longer horizons = more uncertainty = different binding dynamics. Only matters for multi-period models (f1, f2, etc.) but should still be tested for f0.
- MONOTONE: forecast_horizon → 0

**6. Historical Signal Enrichment**
- `hist_da_max_season` — peak seasonal DA shadow price
- WHY: The max across seasons captures extreme events better than the mean (which `hist_da` already uses).
- MONOTONE: hist_da_max_season → 1

### Feature Engineering Opportunities (require code changes in features.py)
These DON'T exist in the source loader but can be derived from available data:
- `prob_exceed_ratio_110_90` = prob_exceed_110 / (prob_exceed_90 + 1e-6) — tail shape ratio
- `overload_severity` = expected_overload / (prob_exceed_100 + 1e-6) — conditional expected severity
- `sf_x_exceed` = sf_max_abs × prob_exceed_100 — network importance × binding risk interaction
- `hist_da_x_sf` = hist_da × sf_max_abs — historical + topology interaction

## Priority B: Minor Parameter Tweaks

- **train_months = 14 is the MAXIMUM** (human decision, non-negotiable)
- HP tuning is a dead end (5 experiments confirm)
- threshold_beta stays at 0.7

## Hard Constraints

- train_months ≤ 14 (HARD MAX — human override)
- v0 HP defaults are near-optimal, don't waste iterations
- XGBoost + monotone constraints is the correct architecture
- Precision > recall always
