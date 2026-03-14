"""
V2 End-to-end: Does MW-adjusted mtm_1st_mean improve the actual baseline?

The V2 baseline formula:
  baseline = 0.65 * avg(mtm_1st, mtm_2nd, mtm_3rd) + 0.35 * avg(rev1, rev2, rev3)

MW factor only changes mtm_1st_mean. So the impact is diluted:
  ~0.65 * (1/3) ≈ 22% of baseline weight comes from mtm_1st.

Tests on TWO datasets:
  1. Pool: /opt/temp/qianli/pjm_mcp_pred_training2 (all paths, June auction only)
  2. Trades: /opt/temp/shiyi/trash/pjm_onpeak.parquet (actual traded paths, June auction only)

CRITICAL: We only update June auction prices (auction_month = YYYY-06).
"""
import os
import gc
import resource
import numpy as np
import polars as pl

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

# ================================================================
# PHASE 1: Compute MW factors (same as previous scripts)
# ================================================================
print("PHASE 1: Computing MW seasonal factors...")
print(f"  Memory: {mem_mb():.0f} MB")

import pandas as pd

annual_nodes = {}
for year in YEARS:
    annual = mcp_loader._load_annual(planning_year=year)
    annual_op = annual[
        (annual['market_round'] == 4) & (annual['class_type'] == 'onpeak')
    ][['node_name', 'mcp']].copy()
    annual_nodes[year] = annual_op.drop_duplicates('node_name').set_index('node_name')['mcp']
    del annual
    gc.collect()

# Build path-level data for ratio computation
year_paths = {}
for year in YEARS:
    cleared = cleared_loader._load_annual(planning_year=year)
    cleared_filt = cleared[
        (cleared['class_type'].str.upper() == 'ONPEAK') & (cleared['hedge_type'] == 'Obligation')
    ]
    paths = cleared_filt[['source_name', 'sink_name']].drop_duplicates()
    ann = annual_nodes[year]

    monthly = {}
    for fx in range(12):
        df = mcp_loader.load_data(auction_month=f'{year}-06-01', market_round=1, period_type=f'f{fx}')
        df_op = df[df['class_type'] == 'onpeak'][['node_name', 'mcp']].copy()
        monthly[fx] = df_op.drop_duplicates('node_name').set_index('node_name')['mcp']

    records = []
    for _, row in paths.iterrows():
        src, snk = row['source_name'], row['sink_name']
        if src not in ann.index or snk not in ann.index:
            continue
        ann_path = float(ann[snk] - ann[src])
        fx_vals = []
        valid = True
        for fx in range(12):
            if src not in monthly[fx].index or snk not in monthly[fx].index:
                valid = False
                break
            fx_vals.append(float(monthly[fx][snk] - monthly[fx][src]))
        if not valid:
            continue
        rec = {'src': src, 'snk': snk, 'annual_mcp': ann_path}
        for fx in range(12):
            rec[f'f{fx}'] = fx_vals[fx]
        records.append(rec)
    year_paths[year] = pd.DataFrame(records)
    del cleared, monthly
    gc.collect()
    print(f"  PY{year}: {len(records)} paths, mem={mem_mb():.0f} MB")

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

# Pre-compute factors for each test year
year_factors = {}
for test_year in range(2020, 2026):
    train_yrs = [y for y in YEARS if y < test_year]
    year_factors[test_year] = compute_mw_factors(train_yrs)

print("\n  MW factors (expanding median):")
# Show factors for the latest year as reference
latest_factors = year_factors[2025]
CAL_MONTHS = ['Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May']
for fx in range(12):
    adj = latest_factors[fx] * 12
    print(f"    f{fx} ({CAL_MONTHS[fx]}): factor={latest_factors[fx]:.6f}, adj_ratio={adj:.4f}")

# ================================================================
# PHASE 2: Test on POOL data (training2, June auction only)
# ================================================================
print(f"\n{'=' * 70}")
print("PHASE 2: Pool data — June auction partitions only")
print(f"{'=' * 70}")

POOL_BASE = '/opt/temp/qianli/pjm_mcp_pred_training2'

# Read only June auction + onpeak parquets directly (avoids schema mismatch)
POOL_COLS = ['mtm_1st_mean', 'mtm_2nd_mean', 'mtm_3rd_mean', '1(rev)', '2(rev)', '3(rev)', 'mcp_mean', 'path']
import glob as globmod
june_onpeak_files = sorted(globmod.glob(os.path.join(POOL_BASE, 'auction_month=*-06/period_type=*/class_type=onpeak/training.parquet')))
print(f"  Found {len(june_onpeak_files)} June onpeak parquet files")
pool_dfs = []
for f in june_onpeak_files:
    # Extract partition values from path
    parts = f.split('/')
    am = [p for p in parts if p.startswith('auction_month=')][0].replace('auction_month=', '')
    pt = [p for p in parts if p.startswith('period_type=')][0].replace('period_type=', '')
    df = pl.read_parquet(f, columns=POOL_COLS)
    df = df.with_columns([
        pl.lit(am).alias('auction_month'),
        pl.lit(pt).alias('period_type'),
    ])
    pool_dfs.append(df)
pool = pl.concat(pool_dfs)
del pool_dfs

pool = pool.with_columns([
    pl.col('auction_month').str.slice(0, 4).cast(pl.Int32).alias('year'),
    pl.col('period_type').str.replace('f', '').cast(pl.Int32).alias('fx'),
])
print(f"  Pool records: {len(pool)}, years: {sorted(pool['year'].unique().to_list())}")
print(f"  Period types: {sorted(pool['period_type'].unique().to_list())}")
gc.collect()

# Convert to pandas for easier manipulation
pp = pool.to_pandas()
del pool
gc.collect()

# Apply MW adjustment to mtm_1st_mean
pp['mtm_1st_mw'] = pp['mtm_1st_mean'].copy()
for yr in range(2020, 2026):
    if yr not in year_factors:
        continue
    for fx in range(12):
        mask = (pp['year'] == yr) & (pp['fx'] == fx)
        if mask.sum() == 0:
            continue
        adj = year_factors[yr][fx] * 12  # ratio vs naive 1/12
        pp.loc[mask, 'mtm_1st_mw'] = pp.loc[mask, 'mtm_1st_mean'] * adj

# Compute V2 baselines
def compute_v2_baseline(df, mtm1_col):
    """V2 baseline: 0.65 * avg(mtm_1st, mtm_2nd, mtm_3rd) + 0.35 * avg(rev1, rev2, rev3)"""
    avg_mtm = df[[mtm1_col, 'mtm_2nd_mean', 'mtm_3rd_mean']].mean(axis=1)
    rev_cols = ['1(rev)', '2(rev)', '3(rev)']
    has_rev = df[rev_cols].notna().all(axis=1)
    avg_rev = df[rev_cols].mean(axis=1)
    return np.where(has_rev, 0.65 * avg_mtm + 0.35 * avg_rev, avg_mtm)

pp['baseline_orig'] = compute_v2_baseline(pp, 'mtm_1st_mean')
pp['baseline_mw'] = compute_v2_baseline(pp, 'mtm_1st_mw')

# Evaluate
eval_mask = pp['year'].between(2020, 2025) & pp['mcp_mean'].notna()
ep = pp[eval_mask].copy()
print(f"  Eval pool: {len(ep)} records")

def eval_mae(actual, pred):
    err = np.abs(actual - pred)
    return err.mean(), np.median(err)

def print_comparison(df, label):
    print(f"\n  --- {label} ---")
    methods = [
        ('mtm_1st (orig)', 'mtm_1st_mean'),
        ('mtm_1st (MW)', 'mtm_1st_mw'),
        ('V2 baseline (orig)', 'baseline_orig'),
        ('V2 baseline (MW)', 'baseline_mw'),
    ]
    print(f"  {'Method':30s} {'MAE':>10} {'MedAE':>10} {'vs mtm_orig':>12}")
    ref_mae = None
    for label_m, col in methods:
        mae, medae = eval_mae(df['mcp_mean'], df[col])
        if ref_mae is None:
            ref_mae = mae
        pct = (ref_mae - mae) / ref_mae * 100
        print(f"  {label_m:30s} {mae:>10.2f} {medae:>10.2f} {pct:>+12.2f}%")

print_comparison(ep, "Pool overall")

# Per year
print(f"\n  --- Pool per year ---")
print(f"  {'Year':>6} {'Orig BL MAE':>12} {'MW BL MAE':>12} {'Imp%':>8} {'Win%':>8} {'N':>8}")
for year in sorted(ep['year'].unique()):
    yr = ep[ep['year'] == year]
    eo = np.abs(yr['mcp_mean'] - yr['baseline_orig'])
    em = np.abs(yr['mcp_mean'] - yr['baseline_mw'])
    mae_o = eo.mean()
    mae_m = em.mean()
    imp = (mae_o - mae_m) / mae_o * 100
    win = (em < eo).mean() * 100
    print(f"  PY{year} {mae_o:>12.2f} {mae_m:>12.2f} {imp:>+8.1f} {win:>8.1f} {len(yr):>8}")

# Per period type
print(f"\n  --- Pool per period type ---")
print(f"  {'fx':<4} {'Mon':<4} {'Orig BL':>10} {'MW BL':>10} {'mtm1 Orig':>10} {'mtm1 MW':>10} {'BL Imp%':>8}")
for fx in range(12):
    fdf = ep[ep['fx'] == fx]
    if len(fdf) == 0:
        continue
    bl_o = np.abs(fdf['mcp_mean'] - fdf['baseline_orig']).mean()
    bl_m = np.abs(fdf['mcp_mean'] - fdf['baseline_mw']).mean()
    m1_o = np.abs(fdf['mcp_mean'] - fdf['mtm_1st_mean']).mean()
    m1_m = np.abs(fdf['mcp_mean'] - fdf['mtm_1st_mw']).mean()
    imp = (bl_o - bl_m) / bl_o * 100
    print(f"  f{fx:<3} {CAL_MONTHS[fx]:<4} {bl_o:>10.2f} {bl_m:>10.2f} {m1_o:>10.2f} {m1_m:>10.2f} {imp:>+8.1f}")

del pp, ep
gc.collect()
print(f"\n  Memory after pool: {mem_mb():.0f} MB")

# ================================================================
# PHASE 3: Test on TRADES (June auction only)
# ================================================================
print(f"\n{'=' * 70}")
print("PHASE 3: Trades — June auction only")
print(f"{'=' * 70}")

# Load trades, filter June auction
trades = pl.scan_parquet('/opt/temp/shiyi/trash/pjm_onpeak.parquet').filter(
    pl.col('auction_month').dt.month() == 6
).select([
    'auction_month', 'period_type', 'source_id', 'sink_id', 'path',
    'mtm_1st_mean', 'mtm_2nd_mean', 'mtm_3rd_mean',
    '1(rev)', '2(rev)', '3(rev)',
    'mcp_mean', 'baseline',
]).collect()

tp = trades.to_pandas()
tp['year'] = tp['auction_month'].dt.year
tp['fx'] = tp['period_type'].str.replace('f', '').astype(int)
del trades
gc.collect()

print(f"  June trades: {len(tp)}, years: {sorted(tp['year'].unique())}")
print(f"  Period types: {sorted(tp['period_type'].unique())}")

# Dedup by path + year + fx
tp = tp.drop_duplicates(subset=['path', 'year', 'fx'])
print(f"  After dedup: {len(tp)}")

# Apply MW adjustment per year and period type
tp['mtm_1st_mw'] = tp['mtm_1st_mean'].copy()
for yr in range(2020, 2026):
    if yr not in year_factors:
        continue
    for fx in range(12):
        mask = (tp['year'] == yr) & (tp['fx'] == fx)
        if mask.sum() == 0:
            continue
        adj = year_factors[yr][fx] * 12
        tp.loc[mask, 'mtm_1st_mw'] = tp.loc[mask, 'mtm_1st_mean'] * adj

# Compute V2 baselines
tp['baseline_orig'] = compute_v2_baseline(tp, 'mtm_1st_mean')
tp['baseline_mw'] = compute_v2_baseline(tp, 'mtm_1st_mw')

# Evaluate
eval_mask = tp['year'].between(2020, 2025) & tp['mcp_mean'].notna()
et = tp[eval_mask].copy()
print(f"  Eval trades: {len(et)}")

# Check: does our recomputed baseline match the stored one?
if 'baseline' in et.columns and et['baseline'].notna().sum() > 0:
    diff = np.abs(et['baseline'] - et['baseline_orig'])
    print(f"\n  Sanity check: recomputed vs stored baseline")
    print(f"    Mean diff: {diff.mean():.4f}, Max diff: {diff.max():.4f}")
    print(f"    Match (diff < 0.01): {(diff < 0.01).mean()*100:.1f}%")

print_comparison(et, "Trades overall")

# Also show the existing stored baseline for reference
if 'baseline' in et.columns and et['baseline'].notna().sum() > 0:
    mae_stored, _ = eval_mae(et['mcp_mean'], et['baseline'])
    print(f"\n  Stored baseline MAE (for reference): {mae_stored:.2f}")

# Per year
print(f"\n  --- Trades per year ---")
print(f"  {'Year':>6} {'Orig BL MAE':>12} {'MW BL MAE':>12} {'Imp%':>8} {'Win%':>8} {'N':>8}")
for year in sorted(et['year'].unique()):
    yr = et[et['year'] == year]
    eo = np.abs(yr['mcp_mean'] - yr['baseline_orig'])
    em = np.abs(yr['mcp_mean'] - yr['baseline_mw'])
    mae_o = eo.mean()
    mae_m = em.mean()
    imp = (mae_o - mae_m) / mae_o * 100
    win = (em < eo).mean() * 100
    print(f"  PY{year} {mae_o:>12.2f} {mae_m:>12.2f} {imp:>+8.1f} {win:>8.1f} {len(yr):>8}")

# Per period type
print(f"\n  --- Trades per period type ---")
print(f"  {'fx':<4} {'Mon':<4} {'Orig BL':>10} {'MW BL':>10} {'Stored BL':>10} {'BL Imp%':>8} {'N':>8}")
for fx in range(12):
    fdf = et[et['fx'] == fx]
    if len(fdf) == 0:
        continue
    bl_o = np.abs(fdf['mcp_mean'] - fdf['baseline_orig']).mean()
    bl_m = np.abs(fdf['mcp_mean'] - fdf['baseline_mw']).mean()
    bl_s = np.abs(fdf['mcp_mean'] - fdf['baseline']).mean() if fdf['baseline'].notna().sum() > 0 else float('nan')
    imp = (bl_o - bl_m) / bl_o * 100
    print(f"  f{fx:<3} {CAL_MONTHS[fx]:<4} {bl_o:>10.2f} {bl_m:>10.2f} {bl_s:>10.2f} {imp:>+8.1f} {len(fdf):>8}")

# ================================================================
# PHASE 4: Summary (reuses ep from Phase 2, saved before cleanup)
# ================================================================
print(f"\n{'=' * 70}")
print("SUMMARY")
print(f"{'=' * 70}")

# Reload pool data (was freed after Phase 2)
pool_dfs2 = []
for f in june_onpeak_files:
    parts = f.split('/')
    am = [p for p in parts if p.startswith('auction_month=')][0].replace('auction_month=', '')
    pt = [p for p in parts if p.startswith('period_type=')][0].replace('period_type=', '')
    df = pl.read_parquet(f, columns=POOL_COLS)
    df = df.with_columns([pl.lit(am).alias('auction_month'), pl.lit(pt).alias('period_type')])
    pool_dfs2.append(df)
pool_reload = pl.concat(pool_dfs2).with_columns([
    pl.col('auction_month').str.slice(0, 4).cast(pl.Int32).alias('year'),
    pl.col('period_type').str.replace('f', '').cast(pl.Int32).alias('fx'),
])
pp2 = pool_reload.to_pandas()
del pool_reload, pool_dfs2
gc.collect()

pp2['mtm_1st_mw'] = pp2['mtm_1st_mean'].copy()
for yr in range(2020, 2026):
    if yr not in year_factors:
        continue
    for fx in range(12):
        mask = (pp2['year'] == yr) & (pp2['fx'] == fx)
        if mask.sum() == 0:
            continue
        pp2.loc[mask, 'mtm_1st_mw'] = pp2.loc[mask, 'mtm_1st_mean'] * year_factors[yr][fx] * 12

pp2['baseline_orig'] = compute_v2_baseline(pp2, 'mtm_1st_mean')
pp2['baseline_mw'] = compute_v2_baseline(pp2, 'mtm_1st_mw')
ep2 = pp2[pp2['year'].between(2020, 2025) & pp2['mcp_mean'].notna()]

pool_mtm_orig = np.abs(ep2['mcp_mean'] - ep2['mtm_1st_mean']).mean()
pool_mtm_mw = np.abs(ep2['mcp_mean'] - ep2['mtm_1st_mw']).mean()
pool_bl_orig = np.abs(ep2['mcp_mean'] - ep2['baseline_orig']).mean()
pool_bl_mw = np.abs(ep2['mcp_mean'] - ep2['baseline_mw']).mean()

trade_mtm_orig = np.abs(et['mcp_mean'] - et['mtm_1st_mean']).mean()
trade_mtm_mw = np.abs(et['mcp_mean'] - et['mtm_1st_mw']).mean()
trade_bl_orig = np.abs(et['mcp_mean'] - et['baseline_orig']).mean()
trade_bl_mw = np.abs(et['mcp_mean'] - et['baseline_mw']).mean()

print(f"\n  {'Dataset':>10} {'Metric':>20} {'Orig MAE':>10} {'MW MAE':>10} {'Imp%':>8}")
print(f"  {'Pool':>10} {'mtm_1st_mean':>20} {pool_mtm_orig:>10.2f} {pool_mtm_mw:>10.2f} {(pool_mtm_orig-pool_mtm_mw)/pool_mtm_orig*100:>+8.1f}")
print(f"  {'Pool':>10} {'V2 baseline':>20} {pool_bl_orig:>10.2f} {pool_bl_mw:>10.2f} {(pool_bl_orig-pool_bl_mw)/pool_bl_orig*100:>+8.1f}")
print(f"  {'Trades':>10} {'mtm_1st_mean':>20} {trade_mtm_orig:>10.2f} {trade_mtm_mw:>10.2f} {(trade_mtm_orig-trade_mtm_mw)/trade_mtm_orig*100:>+8.1f}")
print(f"  {'Trades':>10} {'V2 baseline':>20} {trade_bl_orig:>10.2f} {trade_bl_mw:>10.2f} {(trade_bl_orig-trade_bl_mw)/trade_bl_orig*100:>+8.1f}")

if 'baseline' in et.columns and et['baseline'].notna().sum() > 0:
    trade_bl_stored = np.abs(et['mcp_mean'] - et['baseline']).mean()
    print(f"  {'Trades':>10} {'Stored baseline':>20} {trade_bl_stored:>10.2f} {'':>10} {'(ref)':>8}")

print(f"\n  Note: MW only changes mtm_1st (1/3 of mtm avg). mtm_1st is ~65% * 1/3 = ~22% of baseline.")
print(f"  Expected dilution: ~7.9% * 0.22 = ~1.7% baseline improvement")

print(f"\nFinal memory: {mem_mb():.0f} MB")

import ray
ray.shutdown()
