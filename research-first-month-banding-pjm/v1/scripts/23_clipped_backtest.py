"""
Fixed backtesting with proper percentage clipping.

Bug found: 2% of paths have |pct| > 1 (due to monthly_sum ≈ 0 while annual ≠ 0),
producing predictions 100-1000x the actual value. These outliers dominate the mean.

Fix: clip percentages to reasonable range, renormalize, then apply.
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

        # RAW percentages (will clip later)
        if abs(monthly_sum) > 1e-6:
            for fx in range(12):
                rec[f'raw_pct_f{fx}'] = fx_vals[fx] / monthly_sum
        else:
            for fx in range(12):
                rec[f'raw_pct_f{fx}'] = 1.0 / 12

        records.append(rec)

    year_data[year] = pd.DataFrame(records)
    print(f"  PY{year}: {len(records)} paths, mem={mem_mb():.0f} MB")
    del annual, cleared
    gc.collect()

# ================================================================
# Helper: clip and renormalize percentages
# ================================================================
def clip_pcts(raw_pcts, lo=0.0, hi=0.20):
    """Clip raw percentages and renormalize to sum to 1.0."""
    clipped = [np.clip(p, lo, hi) for p in raw_pcts]
    total = sum(clipped)
    if abs(total) > 1e-10:
        return [c / total for c in clipped]
    return [1.0/12] * 12


def safe_pcts(raw_pcts, annual, monthly_sum):
    """Return reasonable percentages, falling back to 1/12 for pathological cases."""
    # If monthly_sum and annual have opposite signs or ratio is extreme, use flat
    if abs(annual) < 1e-6:
        return [1.0/12] * 12
    ratio = monthly_sum / annual
    if ratio < 0.2 or ratio > 5.0:
        return [1.0/12] * 12
    # If any individual pct is extreme, clip
    if any(abs(p) > 0.5 for p in raw_pcts):
        return clip_pcts(raw_pcts, lo=-0.02, hi=0.25)
    return raw_pcts

# ================================================================
# PHASE 2: Build predictions with all methods
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

    for src, snk in common:
        prev = df_prev.loc[(src, snk)]
        curr = df_curr.loc[(src, snk)]

        curr_annual = curr['annual_mcp']
        prev_annual = prev['annual_mcp']
        prev_monthly_sum = prev['monthly_sum']

        if abs(curr_annual) < 10:
            continue

        # Raw percentages from last year
        raw_pcts = [prev[f'raw_pct_f{fx}'] for fx in range(12)]
        ratio = prev_monthly_sum / prev_annual if abs(prev_annual) > 1e-6 else 1.0

        # Clipped percentages
        clipped_pcts = clip_pcts(raw_pcts, lo=0.0, hi=0.20)

        # Safe percentages (falls back to flat for pathological cases)
        safe_pcts_vals = safe_pcts(raw_pcts, prev_annual, prev_monthly_sum)

        for fx in range(12):
            actual = curr[f'f{fx}']
            naive = curr_annual / 12.0

            # Method 1: Naive /12
            pred_naive = naive

            # Method 2: Raw per-path distribution (BUGGY for pathological paths)
            pred_raw_dist = curr_annual * raw_pcts[fx]

            # Method 3: Clipped + renormalized distribution
            pred_clipped = curr_annual * clipped_pcts[fx]

            # Method 4: Safe distribution (flat for bad paths)
            pred_safe = curr_annual * safe_pcts_vals[fx]

            # Method 5: Blend 50/50 naive + clipped dist
            pred_blend50_clip = 0.5 * naive + 0.5 * pred_clipped

            # Method 6: Blend 70/30 naive + clipped dist
            pred_blend70_clip = 0.7 * naive + 0.3 * pred_clipped

            # Method 7: Raw dist + ratio (user's method 3), but clipped
            clipped_ratio = np.clip(ratio, 0.5, 2.0)
            pred_clip_dist_ratio = curr_annual * clipped_pcts[fx] * clipped_ratio

            all_results.append({
                'year': year, 'fx': fx, 'actual': actual,
                'annual_mcp': curr_annual, 'ratio': ratio,
                'pred_naive': pred_naive,
                'pred_raw_dist': pred_raw_dist,
                'pred_clipped': pred_clipped,
                'pred_safe': pred_safe,
                'pred_blend50_clip': pred_blend50_clip,
                'pred_blend70_clip': pred_blend70_clip,
                'pred_clip_dist_ratio': pred_clip_dist_ratio,
            })

results = pd.DataFrame(all_results)
print(f"Total samples: {len(results)}, paths: {len(results)//12}")

# ================================================================
# PHASE 3: Evaluate
# ================================================================
print(f"\n{'=' * 70}")
print("PHASE 3: Evaluation")
print(f"{'=' * 70}")

def eval_methods(df, label=""):
    if label:
        print(f"\n  {label} ({len(df)} samples)")
    methods = [
        ('Naive /12', 'pred_naive'),
        ('Raw per-path dist', 'pred_raw_dist'),
        ('Clipped dist', 'pred_clipped'),
        ('Safe dist', 'pred_safe'),
        ('Blend 50 N+50 clip', 'pred_blend50_clip'),
        ('Blend 70 N+30 clip', 'pred_blend70_clip'),
        ('Clip dist * ratio', 'pred_clip_dist_ratio'),
    ]
    for name, col in methods:
        err = np.abs(df['actual'] - df[col])
        err_naive = np.abs(df['actual'] - df['pred_naive'])
        mae = err.mean()
        rmse = np.sqrt(((df['actual'] - df[col]) ** 2).mean())
        win = (err < err_naive).mean() * 100
        median_ae = err.median()
        print(f"    {name:22s}  MAE={mae:8.2f}  MedAE={median_ae:8.2f}  RMSE={rmse:10.2f}  WinVsNaive={win:5.1f}%")

eval_methods(results, "Overall")

# Per year
for year in sorted(results['year'].unique()):
    eval_methods(results[results['year'] == year], f"PY{year}")

# Per month
print(f"\n  --- Per month ---")
print(f"  {'fx':<4} {'Mon':<4} {'Naive':>8} {'ClipDist':>8} {'SafeDist':>8} {'B50clip':>8} {'Best':>8}")
for fx in range(12):
    fdf = results[results['fx'] == fx]
    m = {}
    for col in ['pred_naive', 'pred_clipped', 'pred_safe', 'pred_blend50_clip']:
        m[col] = np.abs(fdf['actual'] - fdf[col]).mean()
    best = min(m, key=m.get)
    labels = {'pred_naive': 'Naive', 'pred_clipped': 'Clip', 'pred_safe': 'Safe', 'pred_blend50_clip': 'Blend'}
    print(f"  f{fx:<3} {cal_months[fx]:<4} {m['pred_naive']:>8.2f} {m['pred_clipped']:>8.2f} "
          f"{m['pred_safe']:>8.2f} {m['pred_blend50_clip']:>8.2f} {labels[best]:>8}")

# By sign
for sign_label, mask_fn in [('Positive annual', lambda d: d['annual_mcp'] > 0),
                              ('Negative annual', lambda d: d['annual_mcp'] < 0)]:
    eval_methods(results[mask_fn(results)], sign_label)

# ================================================================
# PHASE 4: Trades verification
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

# Build prediction lookup with clipped methods
pred_lookup = {}
for year in range(2020, 2026):
    prev_year = year - 1
    if prev_year not in year_data or year not in year_data:
        continue
    df_prev = year_data[prev_year].set_index(['src', 'snk'])
    df_curr = year_data[year].set_index(['src', 'snk'])

    common = set(df_prev.index) & set(df_curr.index)
    for src, snk in common:
        prev = df_prev.loc[(src, snk)]
        curr = df_curr.loc[(src, snk)]
        curr_annual = curr['annual_mcp']
        prev_annual = prev['annual_mcp']
        prev_monthly_sum = prev['monthly_sum']

        if abs(curr_annual) < 1e-6:
            continue

        raw_pcts = [prev[f'raw_pct_f{fx}'] for fx in range(12)]
        ratio = prev_monthly_sum / prev_annual if abs(prev_annual) > 1e-6 else 1.0
        clipped_pcts = clip_pcts(raw_pcts, lo=0.0, hi=0.20)
        safe_pcts_vals = safe_pcts(raw_pcts, prev_annual, prev_monthly_sum)

        for fx in range(12):
            naive = curr_annual / 12.0
            pred_lookup[(year, src, snk, fx)] = {
                'pred_naive': naive,
                'pred_clipped': curr_annual * clipped_pcts[fx],
                'pred_safe': curr_annual * safe_pcts_vals[fx],
                'pred_blend50_clip': 0.5 * naive + 0.5 * curr_annual * clipped_pcts[fx],
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
    print(f"\n  Matched trades (dedup): {len(t)}, unique paths: {t['path'].nunique()}")

    print(f"\n  --- Overall ---")
    for label, col in [('Current mtm_1st', 'current_mtm1st'),
                         ('Our Naive /12', 'pred_naive'),
                         ('Clipped dist', 'pred_clipped'),
                         ('Safe dist', 'pred_safe'),
                         ('Blend 50 N+50 clip', 'pred_blend50_clip')]:
        err = np.abs(t['actual_mcp'] - t[col])
        err_curr = np.abs(t['actual_mcp'] - t['current_mtm1st'])
        mae = err.mean()
        rmse = np.sqrt(((t['actual_mcp'] - t[col]) ** 2).mean())
        win = (err < err_curr).mean() * 100
        median_ae = err.median()
        print(f"    {label:22s}  MAE={mae:8.2f}  MedAE={median_ae:8.2f}  RMSE={rmse:10.2f}  WinVsCurrent={win:5.1f}%")

    # Per year
    for year in sorted(t['year'].unique()):
        yr = t[t['year'] == year]
        print(f"\n    PY{year} ({len(yr)} trades):")
        for label, col in [('Current mtm_1st', 'current_mtm1st'),
                             ('Clipped dist', 'pred_clipped'),
                             ('Safe dist', 'pred_safe'),
                             ('Blend 50 N+50 clip', 'pred_blend50_clip')]:
            mae = np.abs(yr['actual_mcp'] - yr[col]).mean()
            rmse = np.sqrt(((yr['actual_mcp'] - yr[col]) ** 2).mean())
            print(f"      {label:22s}  MAE={mae:8.2f}  RMSE={rmse:10.2f}")

    # Per month
    print(f"\n  --- Per month on trades ---")
    print(f"  {'fx':<4} {'Mon':<4} {'Current':>10} {'Naive':>10} {'ClipDist':>10} {'SafeDist':>10} {'Blend50':>10}")
    for fx in range(12):
        fdf = t[t['fx'] == fx]
        if len(fdf) == 0:
            continue
        m_curr = np.abs(fdf['actual_mcp'] - fdf['current_mtm1st']).mean()
        m_naive = np.abs(fdf['actual_mcp'] - fdf['pred_naive']).mean()
        m_clip = np.abs(fdf['actual_mcp'] - fdf['pred_clipped']).mean()
        m_safe = np.abs(fdf['actual_mcp'] - fdf['pred_safe']).mean()
        m_blend = np.abs(fdf['actual_mcp'] - fdf['pred_blend50_clip']).mean()
        print(f"  f{fx:<3} {cal_months[fx]:<4} {m_curr:>10.2f} {m_naive:>10.2f} {m_clip:>10.2f} {m_safe:>10.2f} {m_blend:>10.2f}")

print(f"\nFinal memory: {mem_mb():.0f} MB")

import ray
ray.shutdown()
