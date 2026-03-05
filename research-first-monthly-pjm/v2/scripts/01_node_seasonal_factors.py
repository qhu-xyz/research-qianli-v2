"""
V2: Node-level seasonal factors for PJM June first-monthly MCP prediction.

Approach:
  For each node, compute historical fx_ratio = avg(fx_node_mcp / annual_node_mcp)
  across multiple years. Then predict:
    pred_fx_path = annual_sink * sink_fx_ratio - annual_src * src_fx_ratio

This addresses v1's core failure: node-level ratios are stable (single value, no
subtraction noise), while path-level ratios are noisy (small denominator from
sink - source near zero).

Phases:
  1. Load node-level annual + monthly MCP data (PY 2017-2025, onpeak)
  2. Compute per-node seasonal factors (rolling averages)
  3. Analyze factor stability (year-over-year std)
  4. Backtest against naive /12 on cleared paths (PY 2020-2025)
  5. Verify on trades parquet
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

YEARS = list(range(2017, 2026))  # PY 2017-2025
CAL_MONTHS = ['Jun', 'Jul', 'Aug', 'Sep', 'Oct', 'Nov', 'Dec', 'Jan', 'Feb', 'Mar', 'Apr', 'May']

# ================================================================
# PHASE 1: Load node-level data
# ================================================================
print("PHASE 1: Loading node-level annual + monthly MCP data...")
print(f"  Years: {YEARS}")

# annual_nodes[year] = Series(node_name -> annual_mcp)
# monthly_nodes[year][fx] = Series(node_name -> fx_mcp)
annual_nodes = {}
monthly_nodes = {}

for year in YEARS:
    auction_month = f'{year}-06-01'

    # Annual R4 onpeak
    annual = mcp_loader._load_annual(planning_year=year)
    annual_op = annual[
        (annual['market_round'] == 4) & (annual['class_type'] == 'onpeak')
    ][['node_name', 'mcp']].copy()
    annual_nodes[year] = annual_op.drop_duplicates('node_name').set_index('node_name')['mcp']

    # Monthly R1 onpeak for f0-f11
    monthly_nodes[year] = {}
    for fx in range(12):
        df = mcp_loader.load_data(
            auction_month=auction_month, market_round=1, period_type=f'f{fx}',
        )
        df_op = df[df['class_type'] == 'onpeak'][['node_name', 'mcp']].copy()
        monthly_nodes[year][fx] = df_op.drop_duplicates('node_name').set_index('node_name')['mcp']

    n_ann = len(annual_nodes[year])
    n_mon = len(monthly_nodes[year][0])
    print(f"  PY{year}: {n_ann} annual nodes, {n_mon} monthly nodes, mem={mem_mb():.0f} MB")
    del annual
    gc.collect()

# ================================================================
# PHASE 2: Compute per-node seasonal factors
# ================================================================
print(f"\n{'=' * 70}")
print("PHASE 2: Compute per-node seasonal factors")
print(f"{'=' * 70}")

# For each node and year: fx_ratio = fx_mcp / annual_mcp
# Then average across available years (rolling window for backtest)
# node_factors[year] = DataFrame with columns: node_name, f0_ratio, ..., f11_ratio
# Uses all years < `year` as training data

def compute_node_factors(train_years, annual_nodes, monthly_nodes, min_years=2):
    """Compute average fx_ratio for each node across train_years.

    Returns dict: node_name -> [f0_ratio, ..., f11_ratio]
    Only includes nodes present in ALL train_years with |annual| > threshold.
    """
    # Collect per-node, per-year ratios
    node_year_ratios = {}  # node -> list of [f0_r, ..., f11_r]

    for yr in train_years:
        ann = annual_nodes[yr]
        for node in ann.index:
            ann_val = ann[node]
            if abs(ann_val) < 50:  # skip near-zero annual nodes
                continue

            # Check all monthly values exist
            fx_vals = []
            valid = True
            for fx in range(12):
                if node not in monthly_nodes[yr][fx].index:
                    valid = False
                    break
                fx_vals.append(monthly_nodes[yr][fx][node])
            if not valid:
                continue

            ratios = [fv / ann_val for fv in fx_vals]
            if node not in node_year_ratios:
                node_year_ratios[node] = []
            node_year_ratios[node].append(ratios)

    # Average across years, require min_years
    result = {}
    for node, ratio_list in node_year_ratios.items():
        if len(ratio_list) < min_years:
            continue
        avg_ratios = np.mean(ratio_list, axis=0).tolist()
        result[node] = avg_ratios

    return result


# Compute factors for each backtest year using expanding window
factors_by_year = {}
for year in range(2020, 2026):
    train_years = [y for y in YEARS if y < year]
    factors = compute_node_factors(train_years, annual_nodes, monthly_nodes, min_years=2)
    factors_by_year[year] = factors
    print(f"  PY{year}: train on {train_years}, {len(factors)} nodes with factors")

# ================================================================
# PHASE 3: Analyze factor stability
# ================================================================
print(f"\n{'=' * 70}")
print("PHASE 3: Factor stability analysis")
print(f"{'=' * 70}")

# For nodes in factors_by_year[2025] (most training data), check std across years
# Compare node-level vs path-level stability
factors_2025 = factors_by_year[2025]
train_years_2025 = [y for y in YEARS if y < 2025]

# Gather per-year ratios for each node to compute std
node_stds = {}
for node in list(factors_2025.keys())[:5000]:  # sample
    year_ratios = []
    for yr in train_years_2025:
        ann = annual_nodes[yr]
        if node not in ann.index or abs(ann[node]) < 50:
            continue
        fx_vals = []
        valid = True
        for fx in range(12):
            if node not in monthly_nodes[yr][fx].index:
                valid = False
                break
            fx_vals.append(monthly_nodes[yr][fx][node])
        if not valid:
            continue
        year_ratios.append([fv / ann[node] for fv in fx_vals])

    if len(year_ratios) >= 3:
        stds = np.std(year_ratios, axis=0)
        node_stds[node] = stds

if node_stds:
    all_stds = np.array(list(node_stds.values()))
    print(f"  Nodes with >=3 years of data: {len(node_stds)}")
    print(f"  Mean per-month std of node ratios:")
    print(f"  {'fx':<4} {'Mon':<4} {'Mean_std':>10} {'Med_std':>10} {'p90_std':>10}")
    for fx in range(12):
        col = all_stds[:, fx]
        print(f"  f{fx:<3} {CAL_MONTHS[fx]:<4} {col.mean():>10.4f} {np.median(col):>10.4f} {np.percentile(col, 90):>10.4f}")
    print(f"\n  Overall mean node-level std: {all_stds.mean():.4f}")
    print(f"  Compare with v1 path-level: 30-56 pct points avg change")

# ================================================================
# PHASE 4: Backtest on cleared paths
# ================================================================
print(f"\n{'=' * 70}")
print("PHASE 4: Backtest on cleared paths (PY 2020-2025, onpeak)")
print(f"{'=' * 70}")

all_results = []

for year in range(2020, 2026):
    factors = factors_by_year[year]

    # Get cleared paths for this year
    cleared = cleared_loader._load_annual(planning_year=year)
    cleared_filt = cleared[
        (cleared['class_type'].str.upper() == 'ONPEAK') & (cleared['hedge_type'] == 'Obligation')
    ]
    paths = cleared_filt[['source_name', 'sink_name']].drop_duplicates()

    ann = annual_nodes[year]
    matched = 0
    skipped = 0

    for _, row in paths.iterrows():
        src, snk = row['source_name'], row['sink_name']

        # Need both nodes in annual and factors
        if src not in ann.index or snk not in ann.index:
            skipped += 1
            continue
        if src not in factors or snk not in factors:
            skipped += 1
            continue

        ann_src = float(ann[src])
        ann_snk = float(ann[snk])
        ann_path = ann_snk - ann_src

        # Skip tiny paths
        if abs(ann_path) < 10:
            skipped += 1
            continue

        src_ratios = factors[src]
        snk_ratios = factors[snk]

        for fx in range(12):
            # Actual monthly MCP
            if src not in monthly_nodes[year][fx].index or snk not in monthly_nodes[year][fx].index:
                continue
            actual = float(monthly_nodes[year][fx][snk] - monthly_nodes[year][fx][src])

            # Method 1: Naive /12
            pred_naive = ann_path / 12.0

            # Method 2: Node-level seasonal factors
            pred_node = ann_snk * snk_ratios[fx] - ann_src * src_ratios[fx]

            # Method 3: Blend 50/50 naive + node-level
            pred_blend50 = 0.5 * pred_naive + 0.5 * pred_node

            # Method 4: Blend 70/30 naive + node-level
            pred_blend70 = 0.7 * pred_naive + 0.3 * pred_node

            # Method 5: Blend 30/70 naive + node-level
            pred_blend30 = 0.3 * pred_naive + 0.7 * pred_node

            all_results.append({
                'year': year, 'fx': fx, 'src': src, 'snk': snk,
                'actual': actual, 'annual_path': ann_path,
                'pred_naive': pred_naive,
                'pred_node': pred_node,
                'pred_blend50': pred_blend50,
                'pred_blend70': pred_blend70,
                'pred_blend30': pred_blend30,
            })

        matched += 1

    print(f"  PY{year}: {matched} paths matched, {skipped} skipped, mem={mem_mb():.0f} MB")
    del cleared
    gc.collect()

results = pd.DataFrame(all_results)
print(f"\n  Total samples: {len(results)}, unique paths: {results.groupby(['src', 'snk']).ngroups}")

# ================================================================
# Evaluation
# ================================================================
print(f"\n{'=' * 70}")
print("EVALUATION")
print(f"{'=' * 70}")

def eval_methods(df, label=""):
    if label:
        print(f"\n  {label} ({len(df)} samples)")
    methods = [
        ('Naive /12', 'pred_naive'),
        ('Node seasonal', 'pred_node'),
        ('Blend 50 N+50 node', 'pred_blend50'),
        ('Blend 70 N+30 node', 'pred_blend70'),
        ('Blend 30 N+70 node', 'pred_blend30'),
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

for year in sorted(results['year'].unique()):
    eval_methods(results[results['year'] == year], f"PY{year}")

# Per month
print(f"\n  --- Per month ---")
print(f"  {'fx':<4} {'Mon':<4} {'Naive':>8} {'Node':>8} {'B50':>8} {'B30':>8} {'B70':>8} {'Best':>8}")
for fx in range(12):
    fdf = results[results['fx'] == fx]
    m = {}
    for col in ['pred_naive', 'pred_node', 'pred_blend50', 'pred_blend30', 'pred_blend70']:
        m[col] = np.abs(fdf['actual'] - fdf[col]).mean()
    best = min(m, key=m.get)
    labels = {'pred_naive': 'Naive', 'pred_node': 'Node', 'pred_blend50': 'B50',
              'pred_blend30': 'B30', 'pred_blend70': 'B70'}
    print(f"  f{fx:<3} {CAL_MONTHS[fx]:<4} {m['pred_naive']:>8.2f} {m['pred_node']:>8.2f} "
          f"{m['pred_blend50']:>8.2f} {m['pred_blend30']:>8.2f} {m['pred_blend70']:>8.2f} {labels[best]:>8}")

# By sign
for sign_label, mask_fn in [('Positive annual', lambda d: d['annual_path'] > 0),
                              ('Negative annual', lambda d: d['annual_path'] < 0)]:
    eval_methods(results[mask_fn(results)], sign_label)

# ================================================================
# PHASE 5: Trades verification
# ================================================================
print(f"\n{'=' * 70}")
print("PHASE 5: Trades verification")
print(f"{'=' * 70}")

import polars as pl

# Build ID -> name mapping from monthly cleared data
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

# Load trades
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

# Build prediction lookup
pred_lookup = {}
for year in range(2020, 2026):
    factors = factors_by_year[year]
    ann = annual_nodes[year]

    for _, row in mapped[mapped['year'] == year][['src_name', 'snk_name']].drop_duplicates().iterrows():
        src, snk = row['src_name'], row['snk_name']
        if src not in ann.index or snk not in ann.index:
            continue
        if src not in factors or snk not in factors:
            continue

        ann_src = float(ann[src])
        ann_snk = float(ann[snk])
        ann_path = ann_snk - ann_src
        if abs(ann_path) < 1e-6:
            continue

        src_ratios = factors[src]
        snk_ratios = factors[snk]

        for fx in range(12):
            naive = ann_path / 12.0
            pred_node = ann_snk * snk_ratios[fx] - ann_src * src_ratios[fx]
            pred_lookup[(year, src, snk, fx)] = {
                'pred_naive': naive,
                'pred_node': pred_node,
                'pred_blend50': 0.5 * naive + 0.5 * pred_node,
                'pred_blend30': 0.3 * naive + 0.7 * pred_node,
                'pred_blend70': 0.7 * naive + 0.3 * pred_node,
            }

matched_trades = []
for _, row in mapped.iterrows():
    key = (row['year'], row['src_name'], row['snk_name'], row['fx'])
    if key in pred_lookup:
        preds = pred_lookup[key]
        matched_trades.append({
            'year': row['year'], 'fx': row['fx'],
            'actual_mcp': row['mcp_mean'],
            'current_mtm1st': row['mtm_1st_mean'],
            **preds,
            'path': row['path'],
        })

t = pd.DataFrame(matched_trades)
if len(t) > 0:
    t = t.drop_duplicates(subset=['path', 'year', 'fx'])
    print(f"\n  Matched trades (dedup): {len(t)}, unique paths: {t['path'].nunique()}")

    print(f"\n  --- Overall ---")
    for label, col in [('Current mtm_1st', 'current_mtm1st'),
                         ('Our Naive /12', 'pred_naive'),
                         ('Node seasonal', 'pred_node'),
                         ('Blend 50 N+50 node', 'pred_blend50'),
                         ('Blend 30 N+70 node', 'pred_blend30'),
                         ('Blend 70 N+30 node', 'pred_blend70')]:
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
                             ('Node seasonal', 'pred_node'),
                             ('Blend 50 N+50 node', 'pred_blend50'),
                             ('Blend 30 N+70 node', 'pred_blend30')]:
            err = np.abs(yr['actual_mcp'] - yr[col])
            err_curr = np.abs(yr['actual_mcp'] - yr['current_mtm1st'])
            mae = err.mean()
            win = (err < err_curr).mean() * 100
            print(f"      {label:22s}  MAE={mae:8.2f}  WinVsCurrent={win:5.1f}%")

    # Per month on trades
    print(f"\n  --- Per month on trades ---")
    print(f"  {'fx':<4} {'Mon':<4} {'Current':>10} {'Naive':>10} {'Node':>10} {'B50':>10} {'B30':>10}")
    for fx in range(12):
        fdf = t[t['fx'] == fx]
        if len(fdf) == 0:
            continue
        m_curr = np.abs(fdf['actual_mcp'] - fdf['current_mtm1st']).mean()
        m_naive = np.abs(fdf['actual_mcp'] - fdf['pred_naive']).mean()
        m_node = np.abs(fdf['actual_mcp'] - fdf['pred_node']).mean()
        m_b50 = np.abs(fdf['actual_mcp'] - fdf['pred_blend50']).mean()
        m_b30 = np.abs(fdf['actual_mcp'] - fdf['pred_blend30']).mean()
        print(f"  f{fx:<3} {CAL_MONTHS[fx]:<4} {m_curr:>10.2f} {m_naive:>10.2f} {m_node:>10.2f} {m_b50:>10.2f} {m_b30:>10.2f}")

print(f"\nFinal memory: {mem_mb():.0f} MB")

import ray
ray.shutdown()
