"""
V2 Refinement: Market-wide seasonal factor — the winning approach.

From script 03, market-wide factor achieved:
  - 7.9% MAE reduction on trades (150.06 vs 162.97)
  - 62% win rate vs current

Now test:
1. Different training windows (expanding, rolling 3yr, rolling 5yr)
2. Blends with naive (10-90%, 20-80%, 30-70%, etc.)
3. Mean vs median for factor computation
4. Factor stability analysis
5. Per-period-type factor tables (the deliverable)
"""
import os
import gc
import resource
import numpy as np
import pandas as pd

os.environ["RAY_ADDRESS"] = "ray://10.8.0.36:10001"

def mem_mb():
    return resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024

from pbase.config.ray import init_ray
import pmodel
init_ray(extra_modules=[pmodel])

from pbase.data.dataset.ftr.mcp.pjm import PjmMcp
from pbase.data.dataset.ftr.cleared.pjm import PjmClearedFtrs

mcp_loader = PjmMcp()
cleared_loader = PjmClearedFtrs()

YEARS = list(range(2017, 2026))
CAL_MONTHS = ['Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May']

# ================================================================
# PHASE 1: Load path-level data
# ================================================================
print("PHASE 1: Loading data...")

annual_nodes = {}
monthly_nodes = {}

for year in YEARS:
    auction_month = f'{year}-06-01'
    annual = mcp_loader._load_annual(planning_year=year)
    annual_op = annual[
        (annual['market_round'] == 4) & (annual['class_type'] == 'onpeak')
    ][['node_name', 'mcp']].copy()
    annual_nodes[year] = annual_op.drop_duplicates('node_name').set_index('node_name')['mcp']

    monthly_nodes[year] = {}
    for fx in range(12):
        df = mcp_loader.load_data(
            auction_month=auction_month, market_round=1, period_type=f'f{fx}',
        )
        df_op = df[df['class_type'] == 'onpeak'][['node_name', 'mcp']].copy()
        monthly_nodes[year][fx] = df_op.drop_duplicates('node_name').set_index('node_name')['mcp']

    print(f"  PY{year}: loaded, mem={mem_mb():.0f} MB")
    del annual
    gc.collect()

# Build path-level data from cleared paths
print("\nBuilding path-level records...")
year_paths = {}

for year in YEARS:
    cleared = cleared_loader._load_annual(planning_year=year)
    cleared_filt = cleared[
        (cleared['class_type'].str.upper() == 'ONPEAK') & (cleared['hedge_type'] == 'Obligation')
    ]
    paths = cleared_filt[['source_name', 'sink_name']].drop_duplicates()
    ann = annual_nodes[year]

    records = []
    for _, row in paths.iterrows():
        src, snk = row['source_name'], row['sink_name']
        if src not in ann.index or snk not in ann.index:
            continue
        ann_path = float(ann[snk] - ann[src])

        fx_vals = []
        valid = True
        for fx in range(12):
            if src not in monthly_nodes[year][fx].index or snk not in monthly_nodes[year][fx].index:
                valid = False
                break
            fx_vals.append(float(monthly_nodes[year][fx][snk] - monthly_nodes[year][fx][src]))
        if not valid:
            continue

        rec = {'src': src, 'snk': snk, 'annual_mcp': ann_path}
        for fx in range(12):
            rec[f'f{fx}'] = fx_vals[fx]
        records.append(rec)

    year_paths[year] = pd.DataFrame(records)
    print(f"  PY{year}: {len(records)} paths")
    del cleared
    gc.collect()

# ================================================================
# PHASE 2: Compute market-wide factors with different methods
# ================================================================
print(f"\n{'=' * 70}")
print("PHASE 2: Factor computation variants")
print(f"{'=' * 70}")

def compute_mw_factors(train_years, method='median', min_ann=10):
    """Compute market-wide seasonal factors.

    For each month fx, compute the median (or mean) of fx_mcp / annual_mcp
    across all paths and training years, excluding paths with |annual| < min_ann.
    """
    all_ratios = {fx: [] for fx in range(12)}
    for ty in train_years:
        df = year_paths[ty]
        mask = df['annual_mcp'].abs() > min_ann
        for fx in range(12):
            ratios = df.loc[mask, f'f{fx}'] / df.loc[mask, 'annual_mcp']
            # Clip extreme ratios
            ratios_clipped = ratios.clip(-0.5, 0.5)
            all_ratios[fx].extend(ratios_clipped.tolist())

    factors = {}
    for fx in range(12):
        vals = np.array(all_ratios[fx])
        if method == 'median':
            factors[fx] = np.median(vals)
        elif method == 'mean':
            factors[fx] = np.mean(vals)
        elif method == 'trimmed_mean':
            # Trim top/bottom 5%
            lo, hi = np.percentile(vals, [5, 95])
            trimmed = vals[(vals >= lo) & (vals <= hi)]
            factors[fx] = np.mean(trimmed)
    return factors


# Show factor tables for different training windows
print("\nFactor tables:")
for label, train_yrs in [
    ("All (2017-2024)", list(range(2017, 2025))),
    ("Recent 3yr (2022-2024)", [2022, 2023, 2024]),
    ("Recent 5yr (2020-2024)", list(range(2020, 2025))),
]:
    for method in ['median', 'mean', 'trimmed_mean']:
        factors = compute_mw_factors(train_yrs, method=method)
        total = sum(factors.values())
        print(f"\n  {label} ({method})  total={total:.4f}")
        print(f"  {'fx':<4} {'Mon':<4} {'factor':>10} {'vs_flat':>10}")
        for fx in range(12):
            diff = (factors[fx] - 1/12) * 100
            print(f"  f{fx:<3} {CAL_MONTHS[fx]:<4} {factors[fx]:>10.6f} {diff:>+10.3f}%")


# ================================================================
# PHASE 3: Backtest all variants
# ================================================================
print(f"\n{'=' * 70}")
print("PHASE 3: Backtesting variants")
print(f"{'=' * 70}")

all_results = []

for test_year in range(2020, 2026):
    df_test = year_paths[test_year]

    # Different training windows
    variants = {
        'expanding': [y for y in YEARS if y < test_year],
        'rolling3': [y for y in range(test_year - 3, test_year) if y in year_paths],
        'rolling5': [y for y in range(test_year - 5, test_year) if y in year_paths],
    }

    # Compute factors for each variant
    factor_sets = {}
    for window, train_yrs in variants.items():
        if not train_yrs:
            continue
        for method in ['median', 'mean', 'trimmed_mean']:
            key = f'{window}_{method}'
            factor_sets[key] = compute_mw_factors(train_yrs, method=method)

    for _, row in df_test.iterrows():
        ann = row['annual_mcp']
        if abs(ann) < 10:
            continue

        for fx in range(12):
            actual = row[f'f{fx}']
            naive = ann / 12.0

            rec = {
                'year': test_year, 'fx': fx, 'actual': actual,
                'annual_mcp': ann, 'pred_naive': naive,
                'src': row['src'], 'snk': row['snk'],
            }

            for key, factors in factor_sets.items():
                pred_mw = ann * factors[fx]
                rec[f'pred_{key}'] = pred_mw
                # Blends with naive
                for w in [0.1, 0.2, 0.3, 0.4, 0.5, 0.7]:
                    rec[f'pred_blend{int(w*100)}_{key}'] = (1 - w) * naive + w * pred_mw

            all_results.append(rec)

    print(f"  PY{test_year}: {len([r for r in all_results if r['year'] == test_year])} samples")

results = pd.DataFrame(all_results)
print(f"\nTotal: {len(results)} samples")

# ================================================================
# EVALUATION
# ================================================================
print(f"\n{'=' * 70}")
print("EVALUATION: All variants")
print(f"{'=' * 70}")

# Find all prediction columns
pred_cols = [c for c in results.columns if c.startswith('pred_') and c != 'pred_naive']

# Compute MAE for each variant
variant_scores = {}
for col in pred_cols:
    err = np.abs(results['actual'] - results[col])
    err_naive = np.abs(results['actual'] - results['pred_naive'])
    mae = err.mean()
    win = (err < err_naive).mean() * 100
    variant_scores[col] = {'mae': mae, 'win': win}

naive_mae = np.abs(results['actual'] - results['pred_naive']).mean()
print(f"\n  Naive /12: MAE = {naive_mae:.2f}")
print(f"\n  Top 15 variants by MAE:")
sorted_variants = sorted(variant_scores.items(), key=lambda x: x[1]['mae'])
for i, (col, scores) in enumerate(sorted_variants[:15]):
    name = col.replace('pred_', '')
    pct_imp = (naive_mae - scores['mae']) / naive_mae * 100
    print(f"  {i+1:2d}. {name:40s}  MAE={scores['mae']:8.2f}  Win={scores['win']:5.1f}%  Imp={pct_imp:+5.2f}%")

# ================================================================
# Best variant: detailed per-year, per-month analysis
# ================================================================
best_col = sorted_variants[0][0]
best_name = best_col.replace('pred_', '')
print(f"\n{'=' * 70}")
print(f"DETAILED ANALYSIS: Best variant = {best_name}")
print(f"{'=' * 70}")

for year in sorted(results['year'].unique()):
    yr = results[results['year'] == year]
    naive_yr = np.abs(yr['actual'] - yr['pred_naive']).mean()
    best_yr = np.abs(yr['actual'] - yr[best_col]).mean()
    win = (np.abs(yr['actual'] - yr[best_col]) < np.abs(yr['actual'] - yr['pred_naive'])).mean() * 100
    pct = (naive_yr - best_yr) / naive_yr * 100
    print(f"  PY{year}: Naive MAE={naive_yr:.2f}, Best MAE={best_yr:.2f}, Win={win:.1f}%, Imp={pct:+.2f}%")

print(f"\n  Per month:")
print(f"  {'fx':<4} {'Mon':<4} {'Naive':>8} {'Best':>8} {'Imp%':>8} {'Win%':>8}")
for fx in range(12):
    fdf = results[results['fx'] == fx]
    m_naive = np.abs(fdf['actual'] - fdf['pred_naive']).mean()
    m_best = np.abs(fdf['actual'] - fdf[best_col]).mean()
    pct = (m_naive - m_best) / m_naive * 100
    win = (np.abs(fdf['actual'] - fdf[best_col]) < np.abs(fdf['actual'] - fdf['pred_naive'])).mean() * 100
    print(f"  f{fx:<3} {CAL_MONTHS[fx]:<4} {m_naive:>8.2f} {m_best:>8.2f} {pct:>+8.2f} {win:>8.1f}")

# Also show pure MW factors (no blend) vs best blend
pure_mw_cols = [c for c in pred_cols if 'blend' not in c]
print(f"\n  Pure MW factor variants (no blend):")
for col in sorted(pure_mw_cols, key=lambda c: variant_scores[c]['mae']):
    s = variant_scores[col]
    print(f"    {col.replace('pred_', ''):35s}  MAE={s['mae']:8.2f}  Win={s['win']:5.1f}%")

# ================================================================
# PHASE 4: Trades verification with best variant
# ================================================================
print(f"\n{'=' * 70}")
print("PHASE 4: Trades verification")
print(f"{'=' * 70}")

import polars as pl

# Build ID mapping
id_to_name = {}
for yr in range(2020, 2026):
    try:
        cdata = cleared_loader.load_data(auction_month=f'{yr}-06-01')
        for _, row in cdata[['source_id', 'source_name']].drop_duplicates().iterrows():
            if pd.notna(row['source_id']) and pd.notna(row['source_name']):
                id_to_name[str(row['source_id'])] = row['source_name']
        for _, row in cdata[['sink_id', 'sink_name']].drop_duplicates().iterrows():
            if pd.notna(row['sink_id']) and pd.notna(row['sink_name']):
                id_to_name[str(row['sink_id'])] = row['sink_name']
    except Exception as e:
        print(f"  Warning: {e}")
print(f"  ID mapping: {len(id_to_name)} entries")

trades = pl.read_parquet('/opt/temp/shiyi/trash/pjm_onpeak.parquet')
june_trades = trades.filter(pl.col('auction_month').dt.month() == 6)
june_pd = june_trades.select([
    'auction_month', 'period_type', 'source_id', 'sink_id', 'path',
    'mtm_1st_mean', 'mcp_mean'
]).to_pandas()
june_pd['year'] = june_pd['auction_month'].dt.year
june_pd['fx'] = june_pd['period_type'].str.replace('f', '').astype(int)
june_pd['src_name'] = june_pd['source_id'].map(id_to_name)
june_pd['snk_name'] = june_pd['sink_id'].map(id_to_name)
mapped = june_pd.dropna(subset=['src_name', 'snk_name'])
del trades, june_trades
gc.collect()

# Parse best variant to get window and method
# Format: blend{pct}_{window}_{method} or {window}_{method}
best_parts = best_name.split('_')
if 'blend' in best_parts[0]:
    blend_w = int(best_parts[0].replace('blend', '')) / 100.0
    window_name = best_parts[1]
    method_name = best_parts[2]
else:
    blend_w = 1.0  # pure MW
    window_name = best_parts[0]
    method_name = best_parts[1]

print(f"\n  Best variant: window={window_name}, method={method_name}, blend_w={blend_w}")

# Also test top 5 variants on trades
top5_cols = [sorted_variants[i][0] for i in range(min(5, len(sorted_variants)))]

matched_trades = []
for test_year in range(2020, 2026):
    # Compute factors for each top variant
    variant_factors = {}
    for col in top5_cols:
        parts = col.replace('pred_', '').split('_')
        if 'blend' in parts[0]:
            bw = int(parts[0].replace('blend', '')) / 100.0
            wname = parts[1]
            mname = parts[2]
        else:
            bw = 1.0
            wname = parts[0]
            mname = parts[1]

        if wname == 'expanding':
            train_yrs = [y for y in YEARS if y < test_year]
        elif wname == 'rolling3':
            train_yrs = [y for y in range(test_year - 3, test_year) if y in year_paths]
        elif wname == 'rolling5':
            train_yrs = [y for y in range(test_year - 5, test_year) if y in year_paths]
        else:
            train_yrs = [y for y in YEARS if y < test_year]

        factors = compute_mw_factors(train_yrs, method=mname)
        variant_factors[col] = (factors, bw)

    ann = annual_nodes[test_year]
    yr_trades = mapped[mapped['year'] == test_year]

    for _, trow in yr_trades.iterrows():
        src, snk = trow['src_name'], trow['snk_name']
        fx_val = trow['fx']

        if src not in ann.index or snk not in ann.index:
            continue
        ann_path = float(ann[snk] - ann[src])
        if abs(ann_path) < 1e-6:
            continue

        naive = ann_path / 12.0
        rec = {
            'year': test_year, 'fx': fx_val, 'path': trow['path'],
            'actual_mcp': trow['mcp_mean'],
            'current_mtm1st': trow['mtm_1st_mean'],
            'pred_naive': naive,
        }

        for col, (factors, bw) in variant_factors.items():
            pred_mw = ann_path * factors[fx_val]
            rec[col] = (1 - bw) * naive + bw * pred_mw if bw < 1.0 else pred_mw

        matched_trades.append(rec)

t = pd.DataFrame(matched_trades)
if len(t) > 0:
    t = t.drop_duplicates(subset=['path', 'year', 'fx'])
    print(f"\n  Matched trades (dedup): {len(t)}, unique paths: {t['path'].nunique()}")

    print(f"\n  --- Overall ---")
    err_curr = np.abs(t['actual_mcp'] - t['current_mtm1st'])
    print(f"    {'Current mtm_1st':30s}  MAE={err_curr.mean():8.2f}")
    err_naive = np.abs(t['actual_mcp'] - t['pred_naive'])
    print(f"    {'Naive /12':30s}  MAE={err_naive.mean():8.2f}  WinVsCurrent={(err_naive < err_curr).mean()*100:5.1f}%")

    for col in top5_cols:
        name = col.replace('pred_', '')
        err = np.abs(t['actual_mcp'] - t[col])
        mae = err.mean()
        win = (err < err_curr).mean() * 100
        pct_imp = (err_curr.mean() - mae) / err_curr.mean() * 100
        print(f"    {name:30s}  MAE={mae:8.2f}  WinVsCurrent={win:5.1f}%  Imp={pct_imp:+5.2f}%")

    # Per year for best variant
    best_trade_col = min(top5_cols, key=lambda c: np.abs(t['actual_mcp'] - t[c]).mean())
    best_trade_name = best_trade_col.replace('pred_', '')
    print(f"\n  Best on trades: {best_trade_name}")

    for year in sorted(t['year'].unique()):
        yr = t[t['year'] == year]
        ec = np.abs(yr['actual_mcp'] - yr['current_mtm1st']).mean()
        eb = np.abs(yr['actual_mcp'] - yr[best_trade_col]).mean()
        win = (np.abs(yr['actual_mcp'] - yr[best_trade_col]) < np.abs(yr['actual_mcp'] - yr['current_mtm1st'])).mean() * 100
        pct = (ec - eb) / ec * 100
        print(f"    PY{year} ({len(yr)} trades): Current MAE={ec:.2f}, Best MAE={eb:.2f}, Win={win:.1f}%, Imp={pct:+.1f}%")

    # Per month for best
    print(f"\n  --- Per month on trades ({best_trade_name}) ---")
    print(f"  {'fx':<4} {'Mon':<4} {'Current':>10} {'Naive':>10} {'Best':>10} {'Imp%':>8} {'Win%':>8}")
    for fx in range(12):
        fdf = t[t['fx'] == fx]
        if len(fdf) == 0:
            continue
        m_curr = np.abs(fdf['actual_mcp'] - fdf['current_mtm1st']).mean()
        m_naive = np.abs(fdf['actual_mcp'] - fdf['pred_naive']).mean()
        m_best = np.abs(fdf['actual_mcp'] - fdf[best_trade_col]).mean()
        pct = (m_curr - m_best) / m_curr * 100
        win = (np.abs(fdf['actual_mcp'] - fdf[best_trade_col]) < np.abs(fdf['actual_mcp'] - fdf['current_mtm1st'])).mean() * 100
        print(f"  f{fx:<3} {CAL_MONTHS[fx]:<4} {m_curr:>10.2f} {m_naive:>10.2f} {m_best:>10.2f} {pct:>+8.1f} {win:>8.1f}")

# ================================================================
# DELIVERABLE: Final factor table
# ================================================================
print(f"\n{'=' * 70}")
print("DELIVERABLE: Market-wide seasonal factor tables")
print(f"{'=' * 70}")

# Compute factors using all available data (2017-2025)
for method in ['median', 'trimmed_mean']:
    factors = compute_mw_factors(list(range(2017, 2026)), method=method)
    total = sum(factors.values())
    print(f"\n  All years (2017-2025), {method}:  sum={total:.6f}")
    print(f"  {'fx':<4} {'Mon':<4} {'factor':>12} {'pct':>8} {'vs_flat':>10}")
    for fx in range(12):
        pct = factors[fx] * 100
        diff = (factors[fx] - 1/12) * 100
        print(f"  f{fx:<3} {CAL_MONTHS[fx]:<4} {factors[fx]:>12.6f} {pct:>8.3f}% {diff:>+10.4f}%")

print(f"\nFinal memory: {mem_mb():.0f} MB")

import ray
ray.shutdown()
