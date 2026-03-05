"""
Diagnose why path-level distribution prediction fails, and test improvements:
1. Filter out paths with extreme ratio (monthly_sum / annual)
2. Clip predictions
3. Use aggregate (average) distributions instead of per-path
4. Blend: weighted average of naive and distribution
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

YEARS = list(range(2019, 2026))
cal_months = ['Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May']

# ================================================================
# PHASE 1: Build path-level data (same as script 20)
# ================================================================
print("PHASE 1: Loading data...")
year_data = {}

for year in YEARS:
    auction_month = f'{year}-06-01'
    annual = mcp_loader._load_annual(planning_year=year)
    annual_op = annual[
        (annual['market_round'] == 4) & (annual['class_type'] == 'onpeak')
    ][['node_name', 'mcp']].copy()
    annual_node = annual_op.drop_duplicates('node_name').set_index('node_name')['mcp']

    monthly_node = {}
    for fx in range(12):
        df = mcp_loader.load_data(
            auction_month=auction_month, market_round=1, period_type=f'f{fx}',
        )
        df_op = df[df['class_type'] == 'onpeak'][['node_name', 'mcp']].copy()
        monthly_node[fx] = df_op.drop_duplicates('node_name').set_index('node_name')['mcp']

    cleared = cleared_loader._load_annual(planning_year=year)
    cleared_filt = cleared[
        (cleared['class_type'].str.upper() == 'ONPEAK') & (cleared['hedge_type'] == 'Obligation')
    ]
    paths = cleared_filt[['source_name', 'sink_name']].drop_duplicates()

    records = []
    for _, row in paths.iterrows():
        src, snk = row['source_name'], row['sink_name']
        if src not in annual_node.index or snk not in annual_node.index:
            continue
        ann_path = float(annual_node[snk] - annual_node[src])

        fx_vals = []
        valid = True
        for fx in range(12):
            if src not in monthly_node[fx].index or snk not in monthly_node[fx].index:
                valid = False
                break
            fx_vals.append(float(monthly_node[fx][snk] - monthly_node[fx][src]))
        if not valid:
            continue

        monthly_sum = sum(fx_vals)
        rec = {'src': src, 'snk': snk, 'annual_mcp': ann_path, 'monthly_sum': monthly_sum}
        for fx in range(12):
            rec[f'f{fx}'] = fx_vals[fx]
        if abs(monthly_sum) > 1e-6:
            for fx in range(12):
                rec[f'pct_f{fx}'] = fx_vals[fx] / monthly_sum
        else:
            for fx in range(12):
                rec[f'pct_f{fx}'] = 1.0 / 12
        records.append(rec)

    year_data[year] = pd.DataFrame(records)
    print(f"  PY{year}: {len(records)} paths, mem={mem_mb():.0f} MB")

    del annual, cleared
    gc.collect()

# ================================================================
# PHASE 2: Diagnose distribution instability
# ================================================================
print(f"\n{'=' * 70}")
print("PHASE 2: Diagnose instability")
print(f"{'=' * 70}")

# Look at ratio distribution (monthly_sum / annual_mcp)
for year in range(2020, 2026):
    prev = year - 1
    df_prev = year_data[prev]
    df_curr = year_data[year]

    common = set(zip(df_prev['src'], df_prev['snk'])) & set(zip(df_curr['src'], df_curr['snk']))

    prev_idx = df_prev.set_index(['src', 'snk'])
    ratios = []
    pct_changes = []  # How much does pct_f0 change year-over-year?
    for src, snk in common:
        p = prev_idx.loc[(src, snk)]
        if abs(p['annual_mcp']) > 10:
            r = p['monthly_sum'] / p['annual_mcp']
            ratios.append(r)
            # Track pct stability
            c = year_data[year].set_index(['src', 'snk'])
            if (src, snk) in c.index:
                curr_row = c.loc[(src, snk)]
                for fx in range(12):
                    if abs(curr_row['monthly_sum']) > 10:
                        pct_changes.append(abs(p[f'pct_f{fx}'] - curr_row[f'pct_f{fx}']))

    ratios = np.array(ratios)
    print(f"\nPY{year} ratio (prev monthly_sum / prev annual):")
    print(f"  Mean={ratios.mean():.2f}, Median={np.median(ratios):.2f}, "
          f"Std={ratios.std():.2f}")
    print(f"  P5={np.percentile(ratios, 5):.2f}, P25={np.percentile(ratios, 25):.2f}, "
          f"P75={np.percentile(ratios, 75):.2f}, P95={np.percentile(ratios, 95):.2f}")
    print(f"  |ratio| > 3: {(np.abs(ratios) > 3).sum()} ({(np.abs(ratios) > 3).mean()*100:.1f}%)")
    print(f"  ratio < 0: {(ratios < 0).sum()} ({(ratios < 0).mean()*100:.1f}%)")

    if pct_changes:
        pct_arr = np.array(pct_changes)
        print(f"  Pct change YoY (mean abs diff in pct_fx): {pct_arr.mean():.4f} "
              f"(i.e. {pct_arr.mean()*100:.2f} pct points)")

# ================================================================
# PHASE 3: Test improved methods
# ================================================================
print(f"\n{'=' * 70}")
print("PHASE 3: Test improved prediction methods")
print(f"{'=' * 70}")

def eval_metrics(actual, predicted):
    err = actual - predicted
    mae = np.abs(err).mean()
    rmse = np.sqrt((err ** 2).mean())
    mask = np.abs(actual) > 10
    mape = (np.abs(err[mask]) / np.abs(actual[mask])).mean() * 100 if mask.sum() > 0 else float('nan')
    return mae, rmse, mape

# Build the aggregate distribution from previous years (9-year avg from findings)
AGG_DIST = {
    0: 0.079, 1: 0.084, 2: 0.080, 3: 0.092, 4: 0.094,
    5: 0.084, 6: 0.084, 7: 0.091, 8: 0.080, 9: 0.078,
    10: 0.076, 11: 0.078
}
# These are from 24h - let me compute onpeak-specific aggregate distributions
# from the data we just loaded
print("\nComputing onpeak aggregate distributions from loaded data...")

# Compute value-weighted aggregate distribution per year
onpeak_agg_dist_by_year = {}
for year in YEARS:
    df = year_data[year]
    pos = df[df['annual_mcp'] > 100]
    if len(pos) > 0:
        totals = [pos[f'f{fx}'].sum() for fx in range(12)]
        grand = sum(totals)
        if abs(grand) > 1:
            onpeak_agg_dist_by_year[year] = [t / grand for t in totals]

for year in sorted(onpeak_agg_dist_by_year):
    d = onpeak_agg_dist_by_year[year]
    print(f"  PY{year}: " + " ".join(f"{v*100:5.1f}%" for v in d))

# Compute rolling-average aggregate distribution (using all prior years)
def get_rolling_agg_dist(target_year):
    """Average of all years before target_year."""
    prior_dists = [onpeak_agg_dist_by_year[y] for y in onpeak_agg_dist_by_year if y < target_year]
    if not prior_dists:
        return [1/12] * 12
    return [np.mean([d[fx] for d in prior_dists]) for fx in range(12)]

# Build predictions for all methods
all_results = []
for year in range(2020, 2026):
    prev_year = year - 1
    if prev_year not in year_data or year not in year_data:
        continue

    df_prev = year_data[prev_year].set_index(['src', 'snk'])
    df_curr = year_data[year].set_index(['src', 'snk'])

    common = set(df_prev.index) & set(df_curr.index)

    rolling_dist = get_rolling_agg_dist(year)
    prev_year_dist = onpeak_agg_dist_by_year.get(prev_year, [1/12]*12)

    for src, snk in common:
        prev = df_prev.loc[(src, snk)]
        curr = df_curr.loc[(src, snk)]

        curr_annual = curr['annual_mcp']
        prev_annual = prev['annual_mcp']
        prev_monthly_sum = prev['monthly_sum']

        if abs(curr_annual) < 10:
            continue

        ratio = prev_monthly_sum / prev_annual if abs(prev_annual) > 1e-6 else 1.0

        for fx in range(12):
            actual = curr[f'f{fx}']
            prev_pct = prev[f'pct_f{fx}']

            # Method 1: Naive /12
            pred_naive = curr_annual / 12.0

            # Method 2: Per-path distribution (raw)
            pred_dist = curr_annual * prev_pct

            # Method 3: Per-path dist + ratio
            pred_dist_ratio = curr_annual * prev_pct * ratio

            # Method 4: Per-path distribution, ratio CLIPPED to [0.5, 2.0]
            clipped_ratio = np.clip(ratio, 0.5, 2.0)
            pred_dist_clipped = curr_annual * prev_pct * clipped_ratio

            # Method 5: Aggregate distribution (rolling avg of prior years)
            pred_agg = curr_annual * rolling_dist[fx] * 12  # *12 because rolling_dist sums to 1

            # Method 6: Aggregate distribution (just last year's aggregate)
            pred_agg_prev = curr_annual * prev_year_dist[fx] * 12

            # Method 7: Blend 50/50 naive + per-path distribution
            pred_blend50 = 0.5 * pred_naive + 0.5 * pred_dist

            # Method 8: Blend 70/30 naive + per-path distribution
            pred_blend70 = 0.7 * pred_naive + 0.3 * pred_dist

            # Method 9: Blend 50/50 naive + aggregate distribution
            pred_blend_agg = 0.5 * pred_naive + 0.5 * pred_agg

            all_results.append({
                'year': year, 'fx': fx, 'actual': actual,
                'pred_naive': pred_naive,
                'pred_dist': pred_dist,
                'pred_dist_ratio': pred_dist_ratio,
                'pred_dist_clipped': pred_dist_clipped,
                'pred_agg': pred_agg,
                'pred_agg_prev': pred_agg_prev,
                'pred_blend50': pred_blend50,
                'pred_blend70': pred_blend70,
                'pred_blend_agg': pred_blend_agg,
                'annual_mcp': curr_annual,
                'ratio': ratio,
            })

results = pd.DataFrame(all_results)
print(f"\nTotal samples: {len(results)}")

# Overall comparison
print(f"\n{'=' * 70}")
print("OVERALL COMPARISON")
print(f"{'=' * 70}")
methods = [
    ('Naive /12', 'pred_naive'),
    ('Per-path dist', 'pred_dist'),
    ('Per-path d+r', 'pred_dist_ratio'),
    ('Per-path clipped', 'pred_dist_clipped'),
    ('Agg dist (rolling)', 'pred_agg'),
    ('Agg dist (prev yr)', 'pred_agg_prev'),
    ('Blend 50 naive+dist', 'pred_blend50'),
    ('Blend 70 naive+dist', 'pred_blend70'),
    ('Blend 50 naive+agg', 'pred_blend_agg'),
]

for label, col in methods:
    mae, rmse, mape = eval_metrics(results['actual'].values, results[col].values)
    err_naive = np.abs(results['actual'] - results['pred_naive'])
    err_this = np.abs(results['actual'] - results[col])
    win_rate = (err_this < err_naive).mean() * 100
    print(f"  {label:22s}  MAE={mae:10.2f}  RMSE={rmse:12.2f}  MAPE={mape:7.1f}%  WinVsNaive={win_rate:5.1f}%")

# Per-year breakdown for top methods
print(f"\n--- Per-year: Top methods ---")
top_methods = [
    ('Naive /12', 'pred_naive'),
    ('Agg dist (rolling)', 'pred_agg'),
    ('Blend 50 naive+agg', 'pred_blend_agg'),
    ('Blend 70 naive+dist', 'pred_blend70'),
]

for year in sorted(results['year'].unique()):
    yr = results[results['year'] == year]
    print(f"\n  PY{year} ({len(yr)} samples):")
    for label, col in top_methods:
        mae, rmse, mape = eval_metrics(yr['actual'].values, yr[col].values)
        print(f"    {label:22s}  MAE={mae:8.2f}  RMSE={rmse:10.2f}  MAPE={mape:6.1f}%")

# Per month breakdown for top methods
print(f"\n--- Per month: Naive vs Agg dist ---")
print(f"  {'fx':<4} {'Mon':<4} {'Naive_MAE':>10} {'Agg_MAE':>10} {'Blend_MAE':>10} {'Winner':>8}")
for fx in range(12):
    fx_df = results[results['fx'] == fx]
    mae_naive = np.abs(fx_df['actual'] - fx_df['pred_naive']).mean()
    mae_agg = np.abs(fx_df['actual'] - fx_df['pred_agg']).mean()
    mae_blend = np.abs(fx_df['actual'] - fx_df['pred_blend_agg']).mean()
    winner = 'Naive' if mae_naive <= min(mae_agg, mae_blend) else ('Agg' if mae_agg <= mae_blend else 'Blend')
    print(f"  f{fx:<3} {cal_months[fx]:<4} {mae_naive:>10.2f} {mae_agg:>10.2f} {mae_blend:>10.2f} {winner:>8}")

# ================================================================
# PHASE 4: Filter analysis — only paths with stable ratio
# ================================================================
print(f"\n{'=' * 70}")
print("PHASE 4: Filtered analysis (stable paths only)")
print(f"{'=' * 70}")

for ratio_bound in [2.0, 1.5, 1.2]:
    stable = results[(results['ratio'] > 1/ratio_bound) & (results['ratio'] < ratio_bound)]
    print(f"\n  Ratio within [{1/ratio_bound:.2f}, {ratio_bound:.2f}]: {len(stable)} samples ({len(stable)/len(results)*100:.0f}%)")
    for label, col in [('Naive /12', 'pred_naive'), ('Per-path dist', 'pred_dist'),
                         ('Blend 70 naive+dist', 'pred_blend70')]:
        mae, _, mape = eval_metrics(stable['actual'].values, stable[col].values)
        err_naive = np.abs(stable['actual'] - stable['pred_naive'])
        err_this = np.abs(stable['actual'] - stable[col])
        win = (err_this < err_naive).mean() * 100
        print(f"    {label:22s}  MAE={mae:8.2f}  MAPE={mape:6.1f}%  WinVsNaive={win:5.1f}%")

# ================================================================
# PHASE 5: Verify on trades
# ================================================================
print(f"\n{'=' * 70}")
print("PHASE 5: Trades verification")
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

# Build prediction lookup with all methods
pred_lookup = {}
for year in range(2020, 2026):
    prev_year = year - 1
    if prev_year not in year_data or year not in year_data:
        continue
    df_prev = year_data[prev_year].set_index(['src', 'snk'])
    df_curr = year_data[year].set_index(['src', 'snk'])
    rolling_dist = get_rolling_agg_dist(year)

    common = set(df_prev.index) & set(df_curr.index)
    for src, snk in common:
        prev = df_prev.loc[(src, snk)]
        curr = df_curr.loc[(src, snk)]
        curr_annual = curr['annual_mcp']
        prev_annual = prev['annual_mcp']

        if abs(curr_annual) < 1e-6:
            continue

        for fx in range(12):
            prev_pct = prev[f'pct_f{fx}']
            pred_lookup[(year, src, snk, fx)] = {
                'pred_naive': curr_annual / 12.0,
                'pred_dist': curr_annual * prev_pct,
                'pred_agg': curr_annual * rolling_dist[fx] * 12,
                'pred_blend_agg': 0.5 * curr_annual / 12.0 + 0.5 * curr_annual * rolling_dist[fx] * 12,
                'pred_blend70': 0.7 * curr_annual / 12.0 + 0.3 * curr_annual * prev_pct,
            }

matched = []
for _, row in mapped.iterrows():
    key = (row['year'], row['src_name'], row['snk_name'], row['fx'])
    if key in pred_lookup:
        preds = pred_lookup[key]
        matched.append({
            'year': row['year'], 'fx': row['fx'],
            'actual_mcp': row['mcp_mean'],
            'current_mtm1st': row['mtm_1st_mean'],
            **preds,
            'path': row['path'],
        })

t = pd.DataFrame(matched)
if len(t) > 0:
    t = t.drop_duplicates(subset=['path', 'year', 'fx'])
    print(f"\n  Matched trades (dedup): {len(t)}")

    print(f"\n  --- Overall ---")
    for label, col in [('Current mtm_1st', 'current_mtm1st'),
                         ('Our Naive /12', 'pred_naive'),
                         ('Per-path dist', 'pred_dist'),
                         ('Agg dist', 'pred_agg'),
                         ('Blend 50 naive+agg', 'pred_blend_agg'),
                         ('Blend 70 naive+dist', 'pred_blend70')]:
        mae, rmse, mape = eval_metrics(t['actual_mcp'].values, t[col].values)
        err_curr = np.abs(t['actual_mcp'] - t['current_mtm1st'])
        err_this = np.abs(t['actual_mcp'] - t[col])
        win = (err_this < err_curr).mean() * 100
        print(f"    {label:22s}  MAE={mae:8.2f}  RMSE={rmse:10.2f}  MAPE={mape:6.1f}%  WinVsCurrent={win:5.1f}%")

    # Per-year
    print(f"\n  --- Per year ---")
    for year in sorted(t['year'].unique()):
        yr = t[t['year'] == year]
        print(f"\n    PY{year} ({len(yr)} trade-ptypes):")
        for label, col in [('Current mtm_1st', 'current_mtm1st'),
                             ('Agg dist', 'pred_agg'),
                             ('Blend 50 naive+agg', 'pred_blend_agg')]:
            mae, rmse, mape = eval_metrics(yr['actual_mcp'].values, yr[col].values)
            print(f"      {label:22s}  MAE={mae:8.2f}  RMSE={rmse:10.2f}  MAPE={mape:6.1f}%")

print(f"\nFinal memory: {mem_mb():.0f} MB")

import ray
ray.shutdown()
