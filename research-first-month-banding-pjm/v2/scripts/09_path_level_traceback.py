"""
V2 Path-level traceback: use prev year's path-level f0-f11/annual ratio.

Instead of node-level (which amplifies noise through subtraction),
directly compute each path's monthly/annual ratio from the previous year:
  factor[path][fx] = prev_year_path_fx_mcp / prev_year_path_annual_mcp

Then predict:
  pred_fx = this_year_annual_path_mcp * factor[path][fx]

Coverage will be lower (path must exist in both years), but avoids
the subtraction-amplification problem entirely.

Also test: expanding window (all prior years for same path), and
hybrid (path-level where available, MW elsewhere).
"""
import os
import gc
import resource
import numpy as np
import pandas as pd
import polars as pl
from scipy import stats

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
# Load data (reuse from previous scripts)
# ================================================================
print("Loading MCP data...")
annual_nodes = {}
monthly_nodes = {}

for year in YEARS:
    annual = mcp_loader._load_annual(planning_year=year)
    annual_op = annual[
        (annual['market_round'] == 4) & (annual['class_type'] == 'onpeak')
    ][['node_name', 'mcp']].copy()
    annual_nodes[year] = annual_op.drop_duplicates('node_name').set_index('node_name')['mcp']

    monthly_nodes[year] = {}
    for fx in range(12):
        df = mcp_loader.load_data(auction_month=f'{year}-06-01', market_round=1, period_type=f'f{fx}')
        df_op = df[df['class_type'] == 'onpeak'][['node_name', 'mcp']].copy()
        monthly_nodes[year][fx] = df_op.drop_duplicates('node_name').set_index('node_name')['mcp']

    del annual
    gc.collect()
    print(f"  PY{year}, mem={mem_mb():.0f} MB")

# Build path-level data (path_key -> {annual, f0..f11} per year)
print("\nBuilding path-level data...")
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
        rec = {'src': src, 'snk': snk, 'annual_mcp': ann_path,
               'path_key': f"{src}|{snk}"}
        for fx in range(12):
            rec[f'f{fx}'] = fx_vals[fx]
        # Compute per-path ratios
        if abs(ann_path) > 1:
            for fx in range(12):
                rec[f'ratio_f{fx}'] = fx_vals[fx] / ann_path
        records.append(rec)
    year_paths[year] = pd.DataFrame(records)
    del cleared
    gc.collect()
    print(f"  PY{year}: {len(records)} paths")

# ================================================================
# Compute MW factors (for comparison)
# ================================================================
def compute_mw_factors(train_years, min_ann=10):
    all_ratios = {fx: [] for fx in range(12)}
    for ty in train_years:
        if ty not in year_paths:
            continue
        df = year_paths[ty]
        mask = df['annual_mcp'].abs() > min_ann
        for fx in range(12):
            ratios = df.loc[mask, f'f{fx}'] / df.loc[mask, 'annual_mcp']
            all_ratios[fx].extend(ratios.clip(-0.5, 0.5).tolist())
    factors = {}
    for fx in range(12):
        vals = np.array(all_ratios[fx])
        factors[fx] = np.median(vals)
    return factors

# ================================================================
# Load trades
# ================================================================
print("\nLoading trades...")
id_to_name = {}
for yr in range(2020, 2026):
    cdata = cleared_loader.load_data(auction_month=f'{yr}-06-01')
    for _, row in cdata[['source_id', 'source_name']].drop_duplicates().iterrows():
        if pd.notna(row['source_id']) and pd.notna(row['source_name']):
            id_to_name[str(row['source_id'])] = row['source_name']
    for _, row in cdata[['sink_id', 'sink_name']].drop_duplicates().iterrows():
        if pd.notna(row['sink_id']) and pd.notna(row['sink_name']):
            id_to_name[str(row['sink_id'])] = row['sink_name']

trades = pl.scan_parquet('/opt/temp/shiyi/trash/pjm_onpeak.parquet').filter(
    pl.col('auction_month').dt.month() == 6
).select([
    'auction_month', 'period_type', 'source_id', 'sink_id', 'path',
    'mtm_1st_mean', 'mcp_mean'
]).collect()

tp = trades.to_pandas()
tp['year'] = tp['auction_month'].dt.year
tp['fx'] = tp['period_type'].str.replace('f', '').astype(int)
tp['src_name'] = tp['source_id'].map(id_to_name)
tp['snk_name'] = tp['sink_id'].map(id_to_name)
tp = tp.drop_duplicates(subset=['path', 'year', 'fx'])
del trades
gc.collect()

# ================================================================
# Build predictions for each trade
# ================================================================
print("\nBuilding predictions...")

# Pre-index year_paths by path_key for fast lookup
path_ratios = {}  # path_ratios[year][path_key] = {fx: ratio}
for year in YEARS:
    yp = year_paths[year]
    pr = {}
    for _, row in yp.iterrows():
        pk = row['path_key']
        if f'ratio_f0' in row and pd.notna(row.get('ratio_f0')):
            ratios = {}
            valid = True
            for fx in range(12):
                k = f'ratio_f{fx}'
                if k in row and pd.notna(row[k]):
                    ratios[fx] = row[k]
                else:
                    valid = False
                    break
            if valid:
                pr[pk] = ratios
    path_ratios[year] = pr
    print(f"  PY{year}: {len(pr)} paths with ratios")

# Pre-compute MW factors per year (avoid recomputing per trade)
year_mw_factors = {}
for test_year in range(2020, 2026):
    train_yrs = [y for y in YEARS if y < test_year]
    year_mw_factors[test_year] = compute_mw_factors(train_yrs)

matched = []
for _, trow in tp.iterrows():
    src = trow['src_name']
    snk = trow['snk_name']
    if pd.isna(src) or pd.isna(snk):
        continue

    yr = trow['year']
    fx_val = trow['fx']
    ann = annual_nodes.get(yr)
    if ann is None or src not in ann.index or snk not in ann.index:
        continue
    ann_path = float(ann[snk] - ann[src])
    if abs(ann_path) < 1e-6:
        continue

    # MW factors (pre-computed)
    mw_factors = year_mw_factors.get(yr)
    if mw_factors is None:
        continue

    naive = ann_path / 12.0
    pred_mw = ann_path * mw_factors[fx_val]

    path_key = f"{src}|{snk}"

    # Path-level traceback: prev year only
    pred_path_prev = np.nan
    prev_year = yr - 1
    if prev_year in path_ratios and path_key in path_ratios[prev_year]:
        ratio = path_ratios[prev_year][path_key][fx_val]
        if abs(ratio) < 0.5:  # same clip as MW
            pred_path_prev = ann_path * ratio

    # Path-level traceback: expanding median (all prior years)
    pred_path_expand = np.nan
    prior_ratios = []
    for ty in train_yrs:
        if ty in path_ratios and path_key in path_ratios[ty]:
            r = path_ratios[ty][path_key][fx_val]
            if abs(r) < 0.5:
                prior_ratios.append(r)
    if prior_ratios:
        pred_path_expand = ann_path * np.median(prior_ratios)

    # Path-level expanding mean
    pred_path_expand_mean = np.nan
    if prior_ratios:
        pred_path_expand_mean = ann_path * np.mean(prior_ratios)

    matched.append({
        'year': yr, 'fx': fx_val, 'path': trow['path'],
        'path_key': path_key,
        'actual': trow['mcp_mean'],
        'current': trow['mtm_1st_mean'],
        'naive': naive,
        'mw': pred_mw,
        'path_prev': pred_path_prev,
        'path_expand': pred_path_expand,
        'path_expand_mean': pred_path_expand_mean,
        'annual_mcp': ann_path,
        'n_prior_years': len(prior_ratios),
    })

t = pd.DataFrame(matched).drop_duplicates(subset=['path', 'year', 'fx'])
print(f"Total trades: {len(t)}")

# Errors
for col in ['current', 'naive', 'mw', 'path_prev', 'path_expand', 'path_expand_mean']:
    t[f'err_{col}'] = np.abs(t['actual'] - t[col])

# ================================================================
# Coverage analysis
# ================================================================
print(f"\n{'=' * 70}")
print("COVERAGE")
print(f"{'=' * 70}")

has_prev = t['path_prev'].notna()
has_expand = t['path_expand'].notna()

print(f"  Path prev-year: {has_prev.sum()}/{len(t)} ({has_prev.mean()*100:.1f}%)")
print(f"  Path expanding: {has_expand.sum()}/{len(t)} ({has_expand.mean()*100:.1f}%)")

# Per year
print(f"\n  Per year:")
print(f"  {'Year':>6} {'N':>6} {'Prev %':>8} {'Expand %':>8} {'Avg yrs':>8}")
for year in sorted(t['year'].unique()):
    yr = t[t['year'] == year]
    prev_pct = yr['path_prev'].notna().mean() * 100
    exp_pct = yr['path_expand'].notna().mean() * 100
    avg_yrs = yr.loc[yr['n_prior_years'] > 0, 'n_prior_years'].mean()
    print(f"  PY{year} {len(yr):>6} {prev_pct:>8.1f} {exp_pct:>8.1f} {avg_yrs:>8.1f}")

# ================================================================
# Comparison on covered subset
# ================================================================
print(f"\n{'=' * 70}")
print("PATH-LEVEL RESULTS (on covered trades)")
print(f"{'=' * 70}")

# Expanding median coverage
tc = t[has_expand].copy()
print(f"\n  Trades with path-level expanding history: {len(tc)} ({len(tc)/len(t)*100:.1f}%)")

print(f"\n  {'Method':30s} {'MAE':>10} {'vs Current':>12} {'Win%':>8}")
for label, col in [
    ('Current mtm_1st', 'err_current'),
    ('Naive /12', 'err_naive'),
    ('MW raw', 'err_mw'),
    ('Path prev-year', 'err_path_prev'),
    ('Path expand (median)', 'err_path_expand'),
    ('Path expand (mean)', 'err_path_expand_mean'),
]:
    valid = tc[col].notna()
    if valid.sum() == 0:
        continue
    subset = tc[valid]
    mae = subset[col].mean()
    if col == 'err_current':
        base_mae = mae
        print(f"  {label:30s} {mae:>10.2f} {'baseline':>12}")
    else:
        imp = (base_mae - mae) / base_mae * 100
        win = (subset[col] < subset['err_current']).mean() * 100
        print(f"  {label:30s} {mae:>10.2f} {imp:>+11.1f}% {win:>8.1f}")

# Per year
print(f"\n  Per year (path expand median vs MW):")
print(f"  {'Year':>6} {'N':>6} {'Curr':>10} {'MW':>10} {'Path':>10} {'MW Imp%':>8} {'Path Imp%':>8}")
for year in sorted(tc['year'].unique()):
    yr = tc[tc['year'] == year]
    mc = yr['err_current'].mean()
    mm = yr['err_mw'].mean()
    mp = yr['err_path_expand'].mean()
    mw_imp = (mc - mm) / mc * 100
    p_imp = (mc - mp) / mc * 100
    print(f"  PY{year} {len(yr):>6} {mc:>10.2f} {mm:>10.2f} {mp:>10.2f} {mw_imp:>+8.1f} {p_imp:>+8.1f}")

# Per month
print(f"\n  Per month (path expand median):")
print(f"  {'fx':<4} {'Mon':<4} {'Curr':>10} {'MW':>10} {'Path':>10} {'MW Imp%':>8} {'Path Imp%':>8}")
for fx in range(12):
    fdf = tc[tc['fx'] == fx]
    if len(fdf) < 10:
        continue
    mc = fdf['err_current'].mean()
    mm = fdf['err_mw'].mean()
    mp = fdf['err_path_expand'].mean()
    mw_imp = (mc - mm) / mc * 100
    p_imp = (mc - mp) / mc * 100
    print(f"  f{fx:<3} {CAL_MONTHS[fx]:<4} {mc:>10.2f} {mm:>10.2f} {mp:>10.2f} {mw_imp:>+8.1f} {p_imp:>+8.1f}")

# ================================================================
# By risk bucket
# ================================================================
print(f"\n{'=' * 70}")
print("PATH-LEVEL BY RISK (error size)")
print(f"{'=' * 70}")

tc['risk_bucket'] = pd.cut(
    tc['err_current'], bins=[0, 50, 100, 200, 500, 1000, float('inf')],
    labels=['0-50', '50-100', '100-200', '200-500', '500-1K', '1000+']
)
print(f"\n  {'Bucket':>12} {'N':>6} {'Curr':>10} {'MW':>10} {'Path':>10} {'MW Imp%':>8} {'Path Imp%':>8}")
for bucket in ['0-50', '50-100', '100-200', '200-500', '500-1K', '1000+']:
    b = tc[tc['risk_bucket'] == bucket]
    if len(b) < 10:
        continue
    mc = b['err_current'].mean()
    mm = b['err_mw'].mean()
    mp = b['err_path_expand'].mean()
    mw_imp = (mc - mm) / mc * 100
    p_imp = (mc - mp) / mc * 100
    print(f"  {bucket:>12} {len(b):>6} {mc:>10.2f} {mm:>10.2f} {mp:>10.2f} {mw_imp:>+8.1f} {p_imp:>+8.1f}")

# ================================================================
# HYBRID: path-level where available, MW elsewhere
# ================================================================
print(f"\n{'=' * 70}")
print("HYBRID: path expand where available, MW elsewhere")
print(f"{'=' * 70}")

t['pred_hybrid'] = t['mw'].copy()
mask_path = t['path_expand'].notna()
t.loc[mask_path, 'pred_hybrid'] = t.loc[mask_path, 'path_expand']
t['err_hybrid'] = np.abs(t['actual'] - t['pred_hybrid'])

mae_curr_all = t['err_current'].mean()
mae_mw_all = t['err_mw'].mean()
mae_hybrid_all = t['err_hybrid'].mean()

print(f"\n  All {len(t)} trades:")
print(f"  {'Method':30s} {'MAE':>10} {'vs Current':>12} {'Win%':>8}")
print(f"  {'Current':30s} {mae_curr_all:>10.2f} {'baseline':>12}")
print(f"  {'MW raw':30s} {mae_mw_all:>10.2f} {(mae_curr_all-mae_mw_all)/mae_curr_all*100:>+11.1f}% {(t['err_mw']<t['err_current']).mean()*100:>8.1f}")
print(f"  {'Hybrid (path+MW)':30s} {mae_hybrid_all:>10.2f} {(mae_curr_all-mae_hybrid_all)/mae_curr_all*100:>+11.1f}% {(t['err_hybrid']<t['err_current']).mean()*100:>8.1f}")

# Per year
print(f"\n  Per year:")
print(f"  {'Year':>6} {'Curr':>10} {'MW':>10} {'Hybrid':>10} {'MW%':>8} {'Hyb%':>8}")
for year in sorted(t['year'].unique()):
    yr = t[t['year'] == year]
    mc = yr['err_current'].mean()
    mm = yr['err_mw'].mean()
    mh = yr['err_hybrid'].mean()
    print(f"  PY{year} {mc:>10.2f} {mm:>10.2f} {mh:>10.2f} {(mc-mm)/mc*100:>+8.1f} {(mc-mh)/mc*100:>+8.1f}")

# ================================================================
# How many years of history helps?
# ================================================================
print(f"\n{'=' * 70}")
print("PATH HISTORY DEPTH: does more history help?")
print(f"{'=' * 70}")

print(f"\n  {'N prior yrs':>12} {'Count':>8} {'Curr MAE':>10} {'MW MAE':>10} {'Path MAE':>10} {'Path Imp%':>10}")
for n_yrs in sorted(tc['n_prior_years'].unique()):
    subset = tc[tc['n_prior_years'] == n_yrs]
    if len(subset) < 50:
        continue
    mc = subset['err_current'].mean()
    mm = subset['err_mw'].mean()
    mp = subset['err_path_expand'].mean()
    imp = (mc - mp) / mc * 100
    print(f"  {n_yrs:>12} {len(subset):>8} {mc:>10.2f} {mm:>10.2f} {mp:>10.2f} {imp:>+10.1f}")

print(f"\nFinal memory: {mem_mb():.0f} MB")

import ray
ray.shutdown()
