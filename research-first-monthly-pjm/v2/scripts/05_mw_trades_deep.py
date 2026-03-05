"""
V2 Deep-dive: Market-wide factors on trades.

Script 03 found pure MW factor gives MAE 150 on trades (8% improvement),
but script 04's path backtest favored tiny blends (0.06% improvement).

Key question: why do paths and trades disagree? And are the MW factors
genuinely useful or is this an artifact?

Tests:
1. Pure MW (unnormalized, factors sum ~0.88)
2. Normalized MW (factors rescaled to sum to 1.0)
3. Different blend ratios specifically on trades
4. Is the improvement just from f0 (Jun) which dominates?
5. Excluding f0 from the evaluation
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
# Load data
# ================================================================
print("Loading data...")
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

def compute_mw_factors(train_years, method='median', min_ann=10):
    all_ratios = {fx: [] for fx in range(12)}
    for ty in train_years:
        df = year_paths[ty]
        mask = df['annual_mcp'].abs() > min_ann
        for fx in range(12):
            ratios = df.loc[mask, f'f{fx}'] / df.loc[mask, 'annual_mcp']
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
            lo, hi = np.percentile(vals, [5, 95])
            trimmed = vals[(vals >= lo) & (vals <= hi)]
            factors[fx] = np.mean(trimmed)
    return factors

# ================================================================
# Load trades
# ================================================================
print("\nLoading trades...")
import polars as pl

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

# ================================================================
# Test all MW variants on trades
# ================================================================
print(f"\n{'=' * 70}")
print("TRADES EVALUATION: All MW variants")
print(f"{'=' * 70}")

matched_trades = []
for test_year in range(2020, 2026):
    # Compute factors for different windows/methods
    factor_variants = {}
    for window_name, train_yrs in [
        ('expanding', [y for y in YEARS if y < test_year]),
        ('rolling3', [y for y in range(test_year-3, test_year) if y in year_paths]),
        ('rolling5', [y for y in range(test_year-5, test_year) if y in year_paths]),
    ]:
        if not train_yrs:
            continue
        for method in ['median', 'mean', 'trimmed_mean']:
            factors = compute_mw_factors(train_yrs, method=method)
            key = f'{window_name}_{method}'
            factor_variants[key] = factors
            # Normalized version (sum to 1.0)
            total = sum(factors.values())
            if abs(total) > 1e-6:
                norm_factors = {fx: v / total for fx, v in factors.items()}
                factor_variants[f'{key}_norm'] = norm_factors

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
            'annual_mcp': ann_path,
        }

        for key, factors in factor_variants.items():
            pred_mw = ann_path * factors[fx_val]
            rec[f'mw_{key}'] = pred_mw
            # Blends
            for blend_pct in [10, 20, 30, 40, 50, 70, 100]:
                w = blend_pct / 100.0
                rec[f'b{blend_pct}_{key}'] = (1 - w) * naive + w * pred_mw

        matched_trades.append(rec)

t = pd.DataFrame(matched_trades)
t = t.drop_duplicates(subset=['path', 'year', 'fx'])
print(f"\nMatched trades: {len(t)}, unique paths: {t['path'].nunique()}")

# Evaluate ALL variants
err_curr = np.abs(t['actual_mcp'] - t['current_mtm1st'])
err_naive = np.abs(t['actual_mcp'] - t['pred_naive'])
curr_mae = err_curr.mean()
naive_mae = err_naive.mean()

# Find all MW and blend columns
pred_cols = [c for c in t.columns if c.startswith('mw_') or c.startswith('b')]
variant_scores = {}
for col in pred_cols:
    err = np.abs(t['actual_mcp'] - t[col])
    mae = err.mean()
    win_vs_curr = (err < err_curr).mean() * 100
    variant_scores[col] = {'mae': mae, 'win': win_vs_curr}

print(f"\n  Current mtm_1st: MAE = {curr_mae:.2f}")
print(f"  Naive /12: MAE = {naive_mae:.2f}")

# Top 20 variants
sorted_v = sorted(variant_scores.items(), key=lambda x: x[1]['mae'])
print(f"\n  Top 20 variants on trades:")
for i, (col, scores) in enumerate(sorted_v[:20]):
    pct_imp = (curr_mae - scores['mae']) / curr_mae * 100
    print(f"  {i+1:2d}. {col:40s}  MAE={scores['mae']:8.2f}  Win={scores['win']:5.1f}%  Imp={pct_imp:+5.2f}%")

# ================================================================
# Deep analysis of top variant
# ================================================================
best_col = sorted_v[0][0]
print(f"\n{'=' * 70}")
print(f"DEEP ANALYSIS: Best = {best_col}")
print(f"{'=' * 70}")

err_best = np.abs(t['actual_mcp'] - t[best_col])

# Per year
print("\n  Per year:")
for year in sorted(t['year'].unique()):
    yr = t[t['year'] == year]
    ec = np.abs(yr['actual_mcp'] - yr['current_mtm1st']).mean()
    eb = np.abs(yr['actual_mcp'] - yr[best_col]).mean()
    win = (np.abs(yr['actual_mcp'] - yr[best_col]) < np.abs(yr['actual_mcp'] - yr['current_mtm1st'])).mean() * 100
    pct = (ec - eb) / ec * 100
    print(f"    PY{year} ({len(yr)} trades): Current={ec:.2f}, Best={eb:.2f}, Win={win:.1f}%, Imp={pct:+.1f}%")

# Per month
print(f"\n  Per month:")
print(f"  {'fx':<4} {'Mon':<4} {'Current':>10} {'Naive':>10} {'Best':>10} {'Imp%':>8} {'Win%':>8}")
for fx in range(12):
    fdf = t[t['fx'] == fx]
    if len(fdf) == 0:
        continue
    m_curr = np.abs(fdf['actual_mcp'] - fdf['current_mtm1st']).mean()
    m_naive = np.abs(fdf['actual_mcp'] - fdf['pred_naive']).mean()
    m_best = np.abs(fdf['actual_mcp'] - fdf[best_col]).mean()
    pct = (m_curr - m_best) / m_curr * 100
    win = (np.abs(fdf['actual_mcp'] - fdf[best_col]) < np.abs(fdf['actual_mcp'] - fdf['current_mtm1st'])).mean() * 100
    print(f"  f{fx:<3} {CAL_MONTHS[fx]:<4} {m_curr:>10.2f} {m_naive:>10.2f} {m_best:>10.2f} {pct:>+8.1f} {win:>8.1f}")

# ================================================================
# Excluding f0 (June is often the largest absolute MCP)
# ================================================================
print(f"\n{'=' * 70}")
print("ANALYSIS: Excluding f0 (Jun)")
print(f"{'=' * 70}")

t_no_f0 = t[t['fx'] != 0]
err_curr_no_f0 = np.abs(t_no_f0['actual_mcp'] - t_no_f0['current_mtm1st'])
print(f"\n  Current MAE (excl f0): {err_curr_no_f0.mean():.2f}")

# Re-rank variants on non-f0 trades
variant_scores_no_f0 = {}
for col in pred_cols:
    err = np.abs(t_no_f0['actual_mcp'] - t_no_f0[col])
    mae = err.mean()
    win = (err < err_curr_no_f0).mean() * 100
    variant_scores_no_f0[col] = {'mae': mae, 'win': win}

sorted_no_f0 = sorted(variant_scores_no_f0.items(), key=lambda x: x[1]['mae'])
print(f"\n  Top 10 variants (excl f0):")
for i, (col, scores) in enumerate(sorted_no_f0[:10]):
    pct_imp = (err_curr_no_f0.mean() - scores['mae']) / err_curr_no_f0.mean() * 100
    print(f"  {i+1:2d}. {col:40s}  MAE={scores['mae']:8.2f}  Win={scores['win']:5.1f}%  Imp={pct_imp:+5.2f}%")

# ================================================================
# Diagnostic: Why does MW help trades more than paths?
# ================================================================
print(f"\n{'=' * 70}")
print("DIAGNOSTIC: Trades vs paths characteristics")
print(f"{'=' * 70}")

# Compare annual_mcp distribution between trades and full paths
trade_annuals = t['annual_mcp'].abs()
print(f"\n  Trade annual_mcp (abs): mean={trade_annuals.mean():.1f}, median={trade_annuals.median():.1f}")
print(f"  Trade annual_mcp quartiles: "
      f"25%={trade_annuals.quantile(0.25):.1f}, "
      f"50%={trade_annuals.quantile(0.50):.1f}, "
      f"75%={trade_annuals.quantile(0.75):.1f}")

# Compare with actual vs naive error distribution
print(f"\n  Error analysis:")
print(f"  actual_mcp mean: {t['actual_mcp'].mean():.2f}")
print(f"  pred_naive mean: {t['pred_naive'].mean():.2f}")
print(f"  current_mtm1st mean: {t['current_mtm1st'].mean():.2f}")
print(f"  best pred mean: {t[best_col].mean():.2f}")

# Signed error analysis: is naive systematically biased?
naive_error = t['actual_mcp'] - t['pred_naive']
curr_error = t['actual_mcp'] - t['current_mtm1st']
best_error = t['actual_mcp'] - t[best_col]

print(f"\n  Signed error (actual - pred):")
print(f"    Naive: mean={naive_error.mean():.2f}, median={naive_error.median():.2f}")
print(f"    Current: mean={curr_error.mean():.2f}, median={curr_error.median():.2f}")
print(f"    Best MW: mean={best_error.mean():.2f}, median={best_error.median():.2f}")

# Per-month signed error
print(f"\n  Per-month signed error (actual - naive):")
for fx in range(12):
    fdf = t[t['fx'] == fx]
    e = fdf['actual_mcp'] - fdf['pred_naive']
    print(f"  f{fx:<3} {CAL_MONTHS[fx]:<4} mean={e.mean():>8.2f} median={e.median():>8.2f}")

print(f"\nFinal memory: {mem_mb():.0f} MB")

import ray
ray.shutdown()
