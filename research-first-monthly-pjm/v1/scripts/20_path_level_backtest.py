"""
Path-level backtesting: Can last year's monthly distribution predict this year's
June auction MCPs better than naive annual/12?

Three methods compared:
1. Naive: annual_mcp / 12 (hour-weighted, i.e. current mtm_1st_mean approach)
2. Distribution: annual_mcp * last_year_pct[fx]
3. Distribution + ratio: annual_mcp * last_year_pct[fx] * (last_year_monthly_sum / last_year_annual)

Uses onpeak class_type to match the trades data.
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

# ================================================================
# PHASE 1: Build path-level distributions from PjmMcp (onpeak)
# ================================================================
print("=" * 70)
print("PHASE 1: Build path-level distributions")
print("=" * 70)

YEARS = list(range(2019, 2026))  # Need 2019 for predicting 2020
cal_months = ['Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May']

# Store per-year data: {year: DataFrame with columns [src, snk, annual_mcp, f0..f11, monthly_sum, pct_f0..pct_f11]}
year_data = {}

for year in YEARS:
    print(f"\n--- PY {year} ---")
    auction_month = f'{year}-06-01'

    # Annual R4 onpeak node MCPs
    annual = mcp_loader._load_annual(planning_year=year)
    annual_op = annual[
        (annual['market_round'] == 4) & (annual['class_type'] == 'onpeak')
    ][['node_name', 'mcp']].copy()
    annual_node = annual_op.drop_duplicates('node_name').set_index('node_name')['mcp']
    print(f"  Annual R4 onpeak nodes: {len(annual_node)}")

    # Monthly f0-f11 onpeak node MCPs
    monthly_node = {}
    for fx in range(12):
        df = mcp_loader.load_data(
            auction_month=auction_month, market_round=1, period_type=f'f{fx}',
        )
        df_op = df[df['class_type'] == 'onpeak'][['node_name', 'mcp']].copy()
        monthly_node[fx] = df_op.drop_duplicates('node_name').set_index('node_name')['mcp']
    print(f"  Monthly onpeak nodes (f0): {len(monthly_node[0])}")

    # Annual cleared Obligation onpeak paths (all rounds)
    cleared = cleared_loader._load_annual(planning_year=year)
    cleared_filt = cleared[
        (cleared['class_type'] == 'ONPEAK') & (cleared['hedge_type'] == 'Obligation')
    ]
    if len(cleared_filt) == 0:
        # Try different case
        cleared_filt = cleared[
            (cleared['class_type'].str.upper() == 'ONPEAK') & (cleared['hedge_type'] == 'Obligation')
        ]
    paths = cleared_filt[['source_name', 'sink_name']].drop_duplicates()
    print(f"  Cleared onpeak obligation paths: {len(paths)}")

    # If no onpeak paths, try OnPeak or other variants
    if len(paths) == 0:
        print(f"  WARNING: No onpeak paths found. class_types in cleared: {cleared['class_type'].unique().tolist()}")
        # Fall back to 24H paths
        cleared_filt = cleared[
            (cleared['class_type'] == '24H') & (cleared['hedge_type'] == 'Obligation')
        ]
        paths = cleared_filt[['source_name', 'sink_name']].drop_duplicates()
        print(f"  Fallback to 24H paths: {len(paths)}")

    # Compute path MCPs
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
        rec = {
            'src': src, 'snk': snk,
            'annual_mcp': ann_path,
            'monthly_sum': monthly_sum,
        }
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
    print(f"  Valid paths with both annual & monthly data: {len(records)}")
    print(f"  Memory: {mem_mb():.0f} MB")

    del annual, cleared
    gc.collect()

# ================================================================
# PHASE 2: Path-level backtesting
# ================================================================
print(f"\n{'=' * 70}")
print("PHASE 2: Path-level backtesting")
print(f"{'=' * 70}")

# For each year Y (2020-2025), predict using Y-1's distribution
all_results = []

for year in range(2020, 2026):
    prev_year = year - 1
    if prev_year not in year_data or year not in year_data:
        print(f"  Skipping PY{year}: missing data")
        continue

    df_prev = year_data[prev_year]
    df_curr = year_data[year]

    # Find paths that appear in both years
    prev_paths = set(zip(df_prev['src'], df_prev['snk']))
    curr_paths = set(zip(df_curr['src'], df_curr['snk']))
    common = prev_paths & curr_paths
    print(f"\nPY{year}: {len(common)} common paths (prev={len(prev_paths)}, curr={len(curr_paths)})")

    if len(common) == 0:
        continue

    # Index previous year data by (src, snk)
    df_prev_idx = df_prev.set_index(['src', 'snk'])
    df_curr_idx = df_curr.set_index(['src', 'snk'])

    for src, snk in common:
        prev = df_prev_idx.loc[(src, snk)]
        curr = df_curr_idx.loc[(src, snk)]

        curr_annual = curr['annual_mcp']
        prev_annual = prev['annual_mcp']
        prev_monthly_sum = prev['monthly_sum']

        # Skip paths with tiny annual MCP (noise-dominated)
        if abs(curr_annual) < 10:
            continue

        ratio = prev_monthly_sum / prev_annual if abs(prev_annual) > 1e-6 else 1.0

        for fx in range(12):
            actual = curr[f'f{fx}']
            prev_pct = prev[f'pct_f{fx}']

            # Method 1: Naive (flat annual/12)
            pred_naive = curr_annual / 12.0

            # Method 2: Distribution only
            pred_dist = curr_annual * prev_pct

            # Method 3: Distribution + ratio
            pred_dist_ratio = curr_annual * prev_pct * ratio

            all_results.append({
                'year': year, 'src': src, 'snk': snk, 'fx': fx,
                'actual': actual,
                'pred_naive': pred_naive,
                'pred_dist': pred_dist,
                'pred_dist_ratio': pred_dist_ratio,
                'annual_mcp': curr_annual,
                'prev_pct': prev_pct,
                'ratio': ratio,
            })

results_df = pd.DataFrame(all_results)
print(f"\nTotal prediction samples: {len(results_df)}")
print(f"Unique paths: {results_df.groupby(['src', 'snk']).ngroups}")
print(f"Years covered: {sorted(results_df['year'].unique())}")

# ================================================================
# PHASE 3: Evaluate prediction accuracy
# ================================================================
print(f"\n{'=' * 70}")
print("PHASE 3: Evaluate prediction accuracy")
print(f"{'=' * 70}")

def eval_metrics(actual, predicted, label):
    err = actual - predicted
    mae = np.abs(err).mean()
    rmse = np.sqrt((err ** 2).mean())
    # Directional accuracy: does the prediction have the right sign?
    sign_match = (np.sign(actual) == np.sign(predicted)).mean() * 100
    # Mean absolute percentage error (for non-zero actuals)
    mask = np.abs(actual) > 10
    if mask.sum() > 0:
        mape = (np.abs(err[mask]) / np.abs(actual[mask])).mean() * 100
    else:
        mape = float('nan')
    return {'method': label, 'MAE': mae, 'RMSE': rmse, 'Sign%': sign_match, 'MAPE%': mape}

# Overall metrics
print("\n--- Overall (all years, all months) ---")
metrics = []
for method, col in [('Naive /12', 'pred_naive'), ('Distribution', 'pred_dist'), ('Dist+Ratio', 'pred_dist_ratio')]:
    m = eval_metrics(results_df['actual'].values, results_df[col].values, method)
    metrics.append(m)
    print(f"  {method:20s}  MAE={m['MAE']:8.2f}  RMSE={m['RMSE']:10.2f}  Sign={m['Sign%']:5.1f}%  MAPE={m['MAPE%']:6.1f}%")

# Per-year metrics
print("\n--- Per year ---")
for year in sorted(results_df['year'].unique()):
    yr_df = results_df[results_df['year'] == year]
    print(f"\n  PY{year} ({len(yr_df)} samples):")
    for method, col in [('Naive /12', 'pred_naive'), ('Distribution', 'pred_dist'), ('Dist+Ratio', 'pred_dist_ratio')]:
        m = eval_metrics(yr_df['actual'].values, yr_df[col].values, method)
        print(f"    {method:20s}  MAE={m['MAE']:8.2f}  RMSE={m['RMSE']:10.2f}  Sign={m['Sign%']:5.1f}%  MAPE={m['MAPE%']:6.1f}%")

# Per fx (month) metrics
print("\n--- Per month (aggregated across years) ---")
print(f"  {'fx':<4} {'Mon':<4}", end='')
for method in ['Naive /12', 'Distribution', 'Dist+Ratio']:
    print(f"  {method:>14s}_MAE", end='')
print()
for fx in range(12):
    fx_df = results_df[results_df['fx'] == fx]
    print(f"  f{fx:<3} {cal_months[fx]:<4}", end='')
    for col in ['pred_naive', 'pred_dist', 'pred_dist_ratio']:
        mae = np.abs(fx_df['actual'] - fx_df[col]).mean()
        print(f"  {mae:>18.2f}", end='')
    print()

# Improvement analysis: how often does distribution beat naive?
print("\n--- Win rate: Distribution vs Naive ---")
results_df['err_naive'] = np.abs(results_df['actual'] - results_df['pred_naive'])
results_df['err_dist'] = np.abs(results_df['actual'] - results_df['pred_dist'])
results_df['err_dist_ratio'] = np.abs(results_df['actual'] - results_df['pred_dist_ratio'])
results_df['dist_wins'] = results_df['err_dist'] < results_df['err_naive']
results_df['dist_ratio_wins'] = results_df['err_dist_ratio'] < results_df['err_naive']

print(f"  Distribution wins: {results_df['dist_wins'].mean()*100:.1f}% of samples")
print(f"  Dist+Ratio wins:  {results_df['dist_ratio_wins'].mean()*100:.1f}% of samples")

# Split by positive vs negative annual MCP
for sign_label, mask in [('Positive annual', results_df['annual_mcp'] > 0),
                          ('Negative annual', results_df['annual_mcp'] < 0)]:
    sub = results_df[mask]
    if len(sub) == 0:
        continue
    print(f"\n  {sign_label} ({len(sub)} samples):")
    for method, col in [('Naive /12', 'pred_naive'), ('Distribution', 'pred_dist'), ('Dist+Ratio', 'pred_dist_ratio')]:
        m = eval_metrics(sub['actual'].values, sub[col].values, method)
        print(f"    {method:20s}  MAE={m['MAE']:8.2f}  RMSE={m['RMSE']:10.2f}  MAPE={m['MAPE%']:6.1f}%")
    print(f"    Dist win rate: {sub['dist_wins'].mean()*100:.1f}%, Dist+Ratio win rate: {sub['dist_ratio_wins'].mean()*100:.1f}%")

# ================================================================
# PHASE 4: Verify on trades data
# ================================================================
print(f"\n{'=' * 70}")
print("PHASE 4: Verify on actual trades (pjm_onpeak.parquet)")
print(f"{'=' * 70}")

import polars as pl

# Build node_name -> pnode_id mapping from MCP data
# We need: source_id/sink_id (in trades) -> node_name (in MCP)
# Monthly cleared data has both source_id and source_name
print("\nBuilding ID-to-name mapping from monthly cleared data...")
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
        print(f"  Warning: Could not load cleared data for {yr}: {e}")
print(f"  ID-to-name mapping: {len(id_to_name)} entries")

# Load trades
trades = pl.read_parquet('/opt/temp/shiyi/trash/pjm_onpeak.parquet')
june_trades = trades.filter(pl.col('auction_month').dt.month() == 6)
print(f"  June trades: {len(june_trades)}")

# Extract year, map IDs to names
june_pd = june_trades.select([
    'auction_month', 'period_type', 'source_id', 'sink_id', 'path',
    'mtm_1st_mean', 'mcp_mean'
]).to_pandas()

june_pd['year'] = june_pd['auction_month'].dt.year
june_pd['fx'] = june_pd['period_type'].str.replace('f', '').astype(int)
june_pd['src_name'] = june_pd['source_id'].map(id_to_name)
june_pd['snk_name'] = june_pd['sink_id'].map(id_to_name)

# Drop rows where we couldn't map names
mapped = june_pd.dropna(subset=['src_name', 'snk_name'])
print(f"  Trades with mapped names: {len(mapped)} / {len(june_pd)}")

# For each trade, look up our predictions
# Build lookup from year_data
pred_lookup = {}
for year in range(2020, 2026):
    prev_year = year - 1
    if prev_year not in year_data or year not in year_data:
        continue
    df_prev = year_data[prev_year].set_index(['src', 'snk'])
    df_curr = year_data[year].set_index(['src', 'snk'])

    for (src, snk) in df_curr.index:
        if (src, snk) not in df_prev.index:
            continue
        prev = df_prev.loc[(src, snk)]
        curr = df_curr.loc[(src, snk)]

        curr_annual = curr['annual_mcp']
        prev_annual = prev['annual_mcp']
        prev_monthly_sum = prev['monthly_sum']

        if abs(curr_annual) < 1e-6:
            continue

        ratio = prev_monthly_sum / prev_annual if abs(prev_annual) > 1e-6 else 1.0

        for fx in range(12):
            prev_pct = prev[f'pct_f{fx}']
            pred_lookup[(year, src, snk, fx)] = {
                'pred_naive': curr_annual / 12.0,
                'pred_dist': curr_annual * prev_pct,
                'pred_dist_ratio': curr_annual * prev_pct * ratio,
            }

print(f"  Prediction lookup entries: {len(pred_lookup)}")

# Match trades to predictions
matched_trades = []
for _, row in mapped.iterrows():
    key = (row['year'], row['src_name'], row['snk_name'], row['fx'])
    if key in pred_lookup:
        preds = pred_lookup[key]
        matched_trades.append({
            'year': row['year'],
            'fx': row['fx'],
            'actual_mcp': row['mcp_mean'],
            'current_mtm1st': row['mtm_1st_mean'],
            'pred_naive': preds['pred_naive'],
            'pred_dist': preds['pred_dist'],
            'pred_dist_ratio': preds['pred_dist_ratio'],
            'path': row['path'],
        })

trades_results = pd.DataFrame(matched_trades)
print(f"\n  Matched trades: {len(trades_results)} (unique paths: {trades_results['path'].nunique() if len(trades_results) > 0 else 0})")

if len(trades_results) > 0:
    # De-duplicate: same (path, year, fx) might appear multiple times (multiple bids)
    trades_dedup = trades_results.drop_duplicates(subset=['path', 'year', 'fx'])
    print(f"  After dedup: {len(trades_dedup)}")

    print("\n  --- Trades verification: overall ---")
    for method, col in [('Current mtm_1st', 'current_mtm1st'),
                         ('Our Naive /12', 'pred_naive'),
                         ('Distribution', 'pred_dist'),
                         ('Dist+Ratio', 'pred_dist_ratio')]:
        m = eval_metrics(trades_dedup['actual_mcp'].values, trades_dedup[col].values, method)
        print(f"    {method:20s}  MAE={m['MAE']:8.2f}  RMSE={m['RMSE']:10.2f}  Sign={m['Sign%']:5.1f}%  MAPE={m['MAPE%']:6.1f}%")

    # Per-year
    print("\n  --- Trades verification: per year ---")
    for year in sorted(trades_dedup['year'].unique()):
        yr = trades_dedup[trades_dedup['year'] == year]
        print(f"\n    PY{year} ({len(yr)} trade-ptypes):")
        for method, col in [('Current mtm_1st', 'current_mtm1st'),
                             ('Distribution', 'pred_dist'),
                             ('Dist+Ratio', 'pred_dist_ratio')]:
            m = eval_metrics(yr['actual_mcp'].values, yr[col].values, method)
            print(f"      {method:20s}  MAE={m['MAE']:8.2f}  RMSE={m['RMSE']:10.2f}  MAPE={m['MAPE%']:6.1f}%")

    # Win rate vs current mtm_1st_mean
    trades_dedup = trades_dedup.copy()
    trades_dedup['err_current'] = np.abs(trades_dedup['actual_mcp'] - trades_dedup['current_mtm1st'])
    trades_dedup['err_dist'] = np.abs(trades_dedup['actual_mcp'] - trades_dedup['pred_dist'])
    trades_dedup['err_dist_ratio'] = np.abs(trades_dedup['actual_mcp'] - trades_dedup['pred_dist_ratio'])

    print(f"\n  --- Win rate vs current mtm_1st_mean ---")
    print(f"    Distribution beats current: {(trades_dedup['err_dist'] < trades_dedup['err_current']).mean()*100:.1f}%")
    print(f"    Dist+Ratio beats current:   {(trades_dedup['err_dist_ratio'] < trades_dedup['err_current']).mean()*100:.1f}%")
else:
    print("  No matched trades found!")

# ================================================================
# PHASE 5: Also try using MCP data to build ID mapping from PjmMcp pnode_id
# ================================================================
# If Phase 4 had low match rate, try alternative mapping
if len(trades_results) < 1000:
    print(f"\n{'=' * 70}")
    print("PHASE 5: Alternative mapping via PjmMcp pnode_id")
    print(f"{'=' * 70}")

    # Build pnode_id -> node_name from MCP data
    pnode_to_name = {}
    for yr in range(2020, 2026):
        data = mcp_loader.load_data(
            auction_month=f'{yr}-06-01', market_round=1, period_type='f0'
        )
        for _, row in data[['pnode_id', 'node_name']].drop_duplicates().iterrows():
            if pd.notna(row['pnode_id']):
                pnode_to_name[str(row['pnode_id'])] = row['node_name']
    print(f"  pnode_id mapping: {len(pnode_to_name)} entries")

    # Try matching with pnode_id
    june_pd['src_name2'] = june_pd['source_id'].map(pnode_to_name)
    june_pd['snk_name2'] = june_pd['sink_id'].map(pnode_to_name)
    mapped2 = june_pd.dropna(subset=['src_name2', 'snk_name2'])
    print(f"  Trades with pnode mapping: {len(mapped2)} / {len(june_pd)}")

    matched_trades2 = []
    for _, row in mapped2.iterrows():
        key = (row['year'], row['src_name2'], row['snk_name2'], row['fx'])
        if key in pred_lookup:
            preds = pred_lookup[key]
            matched_trades2.append({
                'year': row['year'],
                'fx': row['fx'],
                'actual_mcp': row['mcp_mean'],
                'current_mtm1st': row['mtm_1st_mean'],
                'pred_dist': preds['pred_dist'],
                'pred_dist_ratio': preds['pred_dist_ratio'],
                'path': row['path'],
            })

    t2 = pd.DataFrame(matched_trades2)
    print(f"  Matched via pnode: {len(t2)}")

    if len(t2) > 0:
        t2_dedup = t2.drop_duplicates(subset=['path', 'year', 'fx'])
        print(f"  After dedup: {len(t2_dedup)}")
        for method, col in [('Current mtm_1st', 'current_mtm1st'),
                             ('Distribution', 'pred_dist'),
                             ('Dist+Ratio', 'pred_dist_ratio')]:
            m = eval_metrics(t2_dedup['actual_mcp'].values, t2_dedup[col].values, method)
            print(f"    {method:20s}  MAE={m['MAE']:8.2f}  RMSE={m['RMSE']:10.2f}  MAPE={m['MAPE%']:6.1f}%")

print(f"\nFinal memory: {mem_mb():.0f} MB")

import ray
ray.shutdown()
