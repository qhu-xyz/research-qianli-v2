"""
Clean backtesting: fix the aggregate distribution bug (* 12 removed),
focus on what actually works, and verify on trades.

Key finding from script 21:
- Path-level distributions are very unstable YoY (30-56 pct points change)
- Raw per-path distribution is WORSE than naive /12
- BUT for stable paths (ratio~1), a 70/30 blend beats naive
- Aggregate distribution should be close to naive (deviations ~1%)
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
# PHASE 1: Load data
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

# Compute onpeak aggregate distributions per year
onpeak_agg = {}
for year in YEARS:
    df = year_data[year]
    pos = df[df['annual_mcp'] > 100]
    if len(pos) > 0:
        totals = [pos[f'f{fx}'].sum() for fx in range(12)]
        grand = sum(totals)
        if abs(grand) > 1:
            onpeak_agg[year] = [t / grand for t in totals]

def get_rolling_agg(target_year):
    prior = [onpeak_agg[y] for y in onpeak_agg if y < target_year]
    if not prior:
        return [1/12] * 12
    return [np.mean([d[fx] for d in prior]) for fx in range(12)]

# ================================================================
# PHASE 2: Build predictions
# ================================================================
print(f"\n{'=' * 70}")
print("PHASE 2: Build predictions")
print(f"{'=' * 70}")

all_results = []
for year in range(2020, 2026):
    prev_year = year - 1
    if prev_year not in year_data or year not in year_data:
        continue

    df_prev = year_data[prev_year].set_index(['src', 'snk'])
    df_curr = year_data[year].set_index(['src', 'snk'])
    common = set(df_prev.index) & set(df_curr.index)

    rolling_dist = get_rolling_agg(year)
    prev_dist = onpeak_agg.get(prev_year, [1/12]*12)

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
            naive = curr_annual / 12.0

            # Method 1: Naive /12
            pred_naive = naive

            # Method 2: Aggregate distribution (FIXED: no * 12)
            # pred = annual * agg_pct[fx] -> distributes annual across months
            pred_agg = curr_annual * rolling_dist[fx]

            # Method 3: Per-path distribution
            pred_dist = curr_annual * prev_pct

            # Method 4: Per-path dist with clipped ratio
            clipped_ratio = np.clip(ratio, 0.5, 2.0)
            pred_dist_clipped = curr_annual * prev_pct * clipped_ratio

            # Method 5: Blend 70/30 naive + per-path dist
            pred_blend70 = 0.7 * naive + 0.3 * pred_dist

            # Method 6: Blend 50/50 naive + agg dist
            pred_blend_nagg = 0.5 * naive + 0.5 * pred_agg

            # Method 7: Blend 80/20 naive + per-path dist
            pred_blend80 = 0.8 * naive + 0.2 * pred_dist

            # Method 8: Agg dist (previous year only)
            pred_agg_prev = curr_annual * prev_dist[fx]

            all_results.append({
                'year': year, 'fx': fx, 'actual': actual,
                'pred_naive': pred_naive,
                'pred_agg': pred_agg,
                'pred_agg_prev': pred_agg_prev,
                'pred_dist': pred_dist,
                'pred_dist_clipped': pred_dist_clipped,
                'pred_blend70': pred_blend70,
                'pred_blend80': pred_blend80,
                'pred_blend_nagg': pred_blend_nagg,
                'annual_mcp': curr_annual,
                'ratio': ratio,
            })

results = pd.DataFrame(all_results)
print(f"Total samples: {len(results)}")

# ================================================================
# PHASE 3: Evaluate
# ================================================================
print(f"\n{'=' * 70}")
print("PHASE 3: Evaluation")
print(f"{'=' * 70}")

def eval_all(df, label=""):
    if label:
        print(f"\n  {label} ({len(df)} samples)")
    methods = [
        ('Naive /12', 'pred_naive'),
        ('Agg dist (rolling)', 'pred_agg'),
        ('Agg dist (prev yr)', 'pred_agg_prev'),
        ('Per-path dist', 'pred_dist'),
        ('Per-path clipped', 'pred_dist_clipped'),
        ('Blend 70 N+30 D', 'pred_blend70'),
        ('Blend 80 N+20 D', 'pred_blend80'),
        ('Blend 50 N+50 Agg', 'pred_blend_nagg'),
    ]
    for name, col in methods:
        err = np.abs(df['actual'] - df[col])
        err_naive = np.abs(df['actual'] - df['pred_naive'])
        mae = err.mean()
        rmse = np.sqrt(((df['actual'] - df[col]) ** 2).mean())
        win = (err < err_naive).mean() * 100
        # Weighted MAE (by |annual_mcp|)
        weights = df['annual_mcp'].abs()
        wmae = (err * weights).sum() / weights.sum()
        print(f"    {name:22s}  MAE={mae:8.2f}  wMAE={wmae:8.2f}  RMSE={rmse:10.2f}  WinVsNaive={win:5.1f}%")

eval_all(results, "Overall")

# By year
for year in sorted(results['year'].unique()):
    eval_all(results[results['year'] == year], f"PY{year}")

# By sign
for sign, mask_fn in [('Positive annual', lambda d: d['annual_mcp'] > 0),
                       ('Negative annual', lambda d: d['annual_mcp'] < 0)]:
    eval_all(results[mask_fn(results)], sign)

# By month
print(f"\n  --- Per month: Key methods ---")
print(f"  {'fx':<4} {'Mon':<4} {'Naive':>8} {'AggDist':>8} {'PPathD':>8} {'B70N30D':>8} {'Winner':>8}")
for fx in range(12):
    fdf = results[results['fx'] == fx]
    maes = {}
    for col in ['pred_naive', 'pred_agg', 'pred_dist', 'pred_blend70']:
        maes[col] = np.abs(fdf['actual'] - fdf[col]).mean()
    best = min(maes, key=maes.get)
    labels = {'pred_naive': 'Naive', 'pred_agg': 'Agg', 'pred_dist': 'PPath', 'pred_blend70': 'Blend'}
    print(f"  f{fx:<3} {cal_months[fx]:<4} {maes['pred_naive']:>8.2f} {maes['pred_agg']:>8.2f} "
          f"{maes['pred_dist']:>8.2f} {maes['pred_blend70']:>8.2f} {labels[best]:>8}")

# ================================================================
# PHASE 4: Filtered analysis (stable paths)
# ================================================================
print(f"\n{'=' * 70}")
print("PHASE 4: Filtered analysis")
print(f"{'=' * 70}")

for lo, hi, label in [(0.5, 2.0, "0.5-2.0"), (0.67, 1.5, "0.67-1.5"),
                        (0.8, 1.25, "0.8-1.25"), (0.9, 1.1, "0.9-1.1")]:
    stable = results[(results['ratio'] >= lo) & (results['ratio'] <= hi)]
    eval_all(stable, f"Ratio [{lo:.2f}, {hi:.2f}]")

# ================================================================
# PHASE 5: Trades verification
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

# Build prediction lookup
pred_lookup = {}
for year in range(2020, 2026):
    prev_year = year - 1
    if prev_year not in year_data or year not in year_data:
        continue
    df_prev = year_data[prev_year].set_index(['src', 'snk'])
    df_curr = year_data[year].set_index(['src', 'snk'])
    rolling_dist = get_rolling_agg(year)

    common = set(df_prev.index) & set(df_curr.index)
    for src, snk in common:
        prev = df_prev.loc[(src, snk)]
        curr = df_curr.loc[(src, snk)]
        curr_annual = curr['annual_mcp']

        if abs(curr_annual) < 1e-6:
            continue

        for fx in range(12):
            prev_pct = prev[f'pct_f{fx}']
            naive = curr_annual / 12.0
            pred_lookup[(year, src, snk, fx)] = {
                'pred_naive': naive,
                'pred_agg': curr_annual * rolling_dist[fx],
                'pred_dist': curr_annual * prev_pct,
                'pred_blend70': 0.7 * naive + 0.3 * curr_annual * prev_pct,
                'pred_blend80': 0.8 * naive + 0.2 * curr_annual * prev_pct,
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
    print(f"  Unique paths: {t['path'].nunique()}")
    print(f"  Years: {sorted(t['year'].unique())}")

    print(f"\n  --- Overall ---")
    for label, col in [('Current mtm_1st', 'current_mtm1st'),
                         ('Our Naive /12', 'pred_naive'),
                         ('Agg dist', 'pred_agg'),
                         ('Per-path dist', 'pred_dist'),
                         ('Blend 70 N+30 D', 'pred_blend70'),
                         ('Blend 80 N+20 D', 'pred_blend80')]:
        err = np.abs(t['actual_mcp'] - t[col])
        err_curr = np.abs(t['actual_mcp'] - t['current_mtm1st'])
        mae = err.mean()
        rmse = np.sqrt(((t['actual_mcp'] - t[col]) ** 2).mean())
        win = (err < err_curr).mean() * 100
        print(f"    {label:22s}  MAE={mae:8.2f}  RMSE={rmse:10.2f}  WinVsCurrent={win:5.1f}%")

    # Per year for trades
    for year in sorted(t['year'].unique()):
        yr = t[t['year'] == year]
        print(f"\n    PY{year} ({len(yr)} trade-ptypes):")
        for label, col in [('Current mtm_1st', 'current_mtm1st'),
                             ('Agg dist', 'pred_agg'),
                             ('Blend 70 N+30 D', 'pred_blend70')]:
            err = np.abs(yr['actual_mcp'] - yr[col])
            mae = err.mean()
            rmse = np.sqrt(((yr['actual_mcp'] - yr[col]) ** 2).mean())
            print(f"      {label:22s}  MAE={mae:8.2f}  RMSE={rmse:10.2f}")

    # Per-month for trades
    print(f"\n  --- Per month on trades ---")
    print(f"  {'fx':<4} {'Mon':<4} {'Current':>10} {'Naive':>10} {'AggDist':>10} {'Blend70':>10}")
    for fx in range(12):
        fdf = t[t['fx'] == fx]
        if len(fdf) == 0:
            continue
        maes = {}
        for col in ['current_mtm1st', 'pred_naive', 'pred_agg', 'pred_blend70']:
            maes[col] = np.abs(fdf['actual_mcp'] - fdf[col]).mean()
        print(f"  f{fx:<3} {cal_months[fx]:<4} {maes['current_mtm1st']:>10.2f} {maes['pred_naive']:>10.2f} "
              f"{maes['pred_agg']:>10.2f} {maes['pred_blend70']:>10.2f}")

print(f"\nFinal memory: {mem_mb():.0f} MB")

import ray
ray.shutdown()
