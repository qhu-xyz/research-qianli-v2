"""
V2 Follow-up analysis:
1. Statistical significance of MW improvement on mtm_1st_mean
2. Normalized vs unnormalized MW factors (sum=1.0 vs sum=0.88)
3. Node-level traceback: use prev year's June auction f0-f11 distribution per node
4. Coverage of node-level traceback
5. Does node-level traceback help on high-risk paths?

All tests on June auction only.
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
# Load MCP data
# ================================================================
print("Loading MCP data...")
annual_nodes = {}
monthly_nodes = {}  # monthly_nodes[year][fx] = Series(node_name -> mcp)

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

# Build path-level data
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
    del cleared
    gc.collect()

def compute_mw_factors(train_years, min_ann=10, normalize=False):
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
    if normalize:
        total = sum(factors.values())
        if abs(total) > 1e-6:
            factors = {fx: v / total for fx, v in factors.items()}
    return factors

# Load trades
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

# Build matched trades with predictions
matched = []
for test_year in range(2020, 2026):
    train_yrs = [y for y in YEARS if y < test_year]
    factors_raw = compute_mw_factors(train_yrs, normalize=False)
    factors_norm = compute_mw_factors(train_yrs, normalize=True)

    ann = annual_nodes[test_year]
    yr_trades = tp[tp['year'] == test_year]

    for _, trow in yr_trades.iterrows():
        src, snk = trow['src_name'], trow['snk_name']
        if pd.isna(src) or pd.isna(snk):
            continue
        fx_val = trow['fx']
        if src not in ann.index or snk not in ann.index:
            continue
        ann_path = float(ann[snk] - ann[src])
        if abs(ann_path) < 1e-6:
            continue

        naive = ann_path / 12.0
        pred_mw_raw = ann_path * factors_raw[fx_val]
        pred_mw_norm = ann_path * factors_norm[fx_val]

        matched.append({
            'year': test_year, 'fx': fx_val, 'path': trow['path'],
            'src': src, 'snk': snk,
            'actual': trow['mcp_mean'],
            'current': trow['mtm_1st_mean'],
            'naive': naive,
            'mw_raw': pred_mw_raw,
            'mw_norm': pred_mw_norm,
            'annual_mcp': ann_path,
        })

t = pd.DataFrame(matched).drop_duplicates(subset=['path', 'year', 'fx'])
t['err_current'] = np.abs(t['actual'] - t['current'])
t['err_naive'] = np.abs(t['actual'] - t['naive'])
t['err_mw_raw'] = np.abs(t['actual'] - t['mw_raw'])
t['err_mw_norm'] = np.abs(t['actual'] - t['mw_norm'])
print(f"Matched trades: {len(t)}")

# ================================================================
# Q1: Statistical significance
# ================================================================
print(f"\n{'=' * 70}")
print("Q1: STATISTICAL SIGNIFICANCE of MW improvement")
print(f"{'=' * 70}")

# Paired test: for each trade, is err_mw < err_current?
diff = t['err_current'] - t['err_mw_raw']  # positive means MW is better
print(f"\n  Per-trade error difference (current_err - mw_err):")
print(f"    Mean: {diff.mean():.4f}")
print(f"    Std:  {diff.std():.4f}")
print(f"    N:    {len(diff)}")

# Paired t-test
t_stat, p_value = stats.ttest_rel(t['err_current'], t['err_mw_raw'])
print(f"\n  Paired t-test (H0: mean(err_current) = mean(err_mw)):")
print(f"    t-statistic: {t_stat:.4f}")
print(f"    p-value:     {p_value:.2e}")
print(f"    Significant at 0.01: {'YES' if p_value < 0.01 else 'NO'}")

# Wilcoxon signed-rank (non-parametric, more robust)
w_stat, w_pvalue = stats.wilcoxon(t['err_current'], t['err_mw_raw'], alternative='greater')
print(f"\n  Wilcoxon signed-rank (H0: err_current ≤ err_mw):")
print(f"    statistic: {w_stat:.0f}")
print(f"    p-value:   {w_pvalue:.2e}")
print(f"    Significant at 0.01: {'YES' if w_pvalue < 0.01 else 'NO'}")

# Bootstrap confidence interval for MAE difference
np.random.seed(42)
n_boot = 10000
mae_diffs = []
n = len(t)
for _ in range(n_boot):
    idx = np.random.randint(0, n, n)
    mae_curr = t['err_current'].values[idx].mean()
    mae_mw = t['err_mw_raw'].values[idx].mean()
    mae_diffs.append(mae_curr - mae_mw)
mae_diffs = np.array(mae_diffs)
ci_lo, ci_hi = np.percentile(mae_diffs, [2.5, 97.5])
print(f"\n  Bootstrap 95% CI for MAE improvement (current - MW):")
print(f"    Mean: {np.mean(mae_diffs):.2f}")
print(f"    95% CI: [{ci_lo:.2f}, {ci_hi:.2f}]")
print(f"    Entirely positive (MW always better): {'YES' if ci_lo > 0 else 'NO'}")

# Per-year significance
print(f"\n  Per-year paired t-test:")
print(f"  {'Year':>6} {'MAE diff':>10} {'t-stat':>10} {'p-value':>12} {'Sig?':>6}")
for year in sorted(t['year'].unique()):
    yr = t[t['year'] == year]
    d = yr['err_current'] - yr['err_mw_raw']
    ts, pv = stats.ttest_rel(yr['err_current'], yr['err_mw_raw'])
    sig = 'YES' if pv < 0.01 else 'no'
    print(f"  PY{year} {d.mean():>10.2f} {ts:>10.2f} {pv:>12.2e} {sig:>6}")

# ================================================================
# Q2: Normalized vs unnormalized (sum=1.0 vs sum≈0.88)
# ================================================================
print(f"\n{'=' * 70}")
print("Q2: NORMALIZED (sum=1.0) vs UNNORMALIZED (sum≈0.88) MW factors")
print(f"{'=' * 70}")

mae_curr = t['err_current'].mean()
mae_naive = t['err_naive'].mean()
mae_raw = t['err_mw_raw'].mean()
mae_norm = t['err_mw_norm'].mean()

print(f"\n  {'Method':30s} {'MAE':>10} {'vs Current':>12} {'Win%':>8}")
print(f"  {'Current mtm_1st':30s} {mae_curr:>10.2f} {'baseline':>12} {'':>8}")
print(f"  {'Naive /12':30s} {mae_naive:>10.2f} {(mae_curr-mae_naive)/mae_curr*100:>+11.1f}% {(t['err_naive']<t['err_current']).mean()*100:>8.1f}")
print(f"  {'MW raw (sum≈0.88)':30s} {mae_raw:>10.2f} {(mae_curr-mae_raw)/mae_curr*100:>+11.1f}% {(t['err_mw_raw']<t['err_current']).mean()*100:>8.1f}")
print(f"  {'MW norm (sum=1.0)':30s} {mae_norm:>10.2f} {(mae_curr-mae_norm)/mae_curr*100:>+11.1f}% {(t['err_mw_norm']<t['err_current']).mean()*100:>8.1f}")

# Show factor comparison
print(f"\n  Factor comparison (PY2025 training set):")
train_2025 = [y for y in YEARS if y < 2025]
f_raw = compute_mw_factors(train_2025, normalize=False)
f_norm = compute_mw_factors(train_2025, normalize=True)
print(f"  {'fx':<4} {'Mon':<4} {'Raw':>10} {'Norm':>10} {'1/12':>10}")
for fx in range(12):
    print(f"  f{fx:<3} {CAL_MONTHS[fx]:<4} {f_raw[fx]:>10.6f} {f_norm[fx]:>10.6f} {1/12:>10.6f}")
print(f"  Sum: raw={sum(f_raw.values()):.4f}, norm={sum(f_norm.values()):.4f}")

# Per-month comparison
print(f"\n  Per-month: raw vs norm")
print(f"  {'fx':<4} {'Mon':<4} {'Curr MAE':>10} {'Raw MAE':>10} {'Norm MAE':>10} {'Raw Imp%':>10} {'Norm Imp%':>10}")
for fx in range(12):
    fdf = t[t['fx'] == fx]
    if len(fdf) == 0:
        continue
    mc = fdf['err_current'].mean()
    mr = fdf['err_mw_raw'].mean()
    mn = fdf['err_mw_norm'].mean()
    print(f"  f{fx:<3} {CAL_MONTHS[fx]:<4} {mc:>10.2f} {mr:>10.2f} {mn:>10.2f} {(mc-mr)/mc*100:>+10.1f} {(mc-mn)/mc*100:>+10.1f}")

# ================================================================
# Q3: Node-level traceback — use prev year's June f0-f11 per node
# ================================================================
print(f"\n{'=' * 70}")
print("Q3: NODE-LEVEL TRACEBACK — use prev year's per-node f0-f11 distribution")
print(f"{'=' * 70}")
print("  Idea: for each node, use its actual monthly MCP distribution from")
print("  the previous year's June auction to predict this year's distribution.")
print("  pred_node_fx = annual_node_mcp * (prev_year_node_fx / prev_year_annual_node)")

# For each test year, compute node-level factors from prev year
node_factors = {}  # node_factors[year][node_name][fx] = factor
coverage_stats = {}

for test_year in range(2020, 2026):
    prev_year = test_year - 1
    if prev_year not in annual_nodes or prev_year not in monthly_nodes:
        continue

    prev_ann = annual_nodes[prev_year]
    prev_monthly = monthly_nodes[prev_year]
    curr_ann = annual_nodes[test_year]

    # Build per-node factors from previous year
    nf = {}
    all_nodes = set(curr_ann.index)
    covered = 0
    for node in all_nodes:
        if node not in prev_ann.index:
            continue
        prev_ann_val = prev_ann[node]
        if abs(prev_ann_val) < 1:  # skip near-zero annual
            continue
        valid = True
        factors = {}
        for fx in range(12):
            if node not in prev_monthly[fx].index:
                valid = False
                break
            factors[fx] = prev_monthly[fx][node] / prev_ann_val
        if valid:
            nf[node] = factors
            covered += 1

    node_factors[test_year] = nf
    coverage_stats[test_year] = {
        'total_nodes': len(all_nodes),
        'covered': covered,
        'pct': covered / len(all_nodes) * 100 if all_nodes else 0,
    }
    print(f"  PY{test_year}: {covered}/{len(all_nodes)} nodes covered ({coverage_stats[test_year]['pct']:.1f}%)")

# Apply to trades
print(f"\n  Applying node-level traceback to trades...")
node_preds = []
for _, row in t.iterrows():
    yr = row['year']
    src, snk = row['src'], row['snk']
    fx_val = row['fx']

    if yr not in node_factors:
        continue

    nf = node_factors[yr]
    ann = annual_nodes[yr]

    # Both nodes need factors
    if src not in nf or snk not in nf:
        continue
    if src not in ann.index or snk not in ann.index:
        continue

    # Node-level prediction: apply each node's own seasonal factor
    pred_src = float(ann[src]) * nf[src][fx_val]
    pred_snk = float(ann[snk]) * nf[snk][fx_val]
    pred_path = pred_snk - pred_src

    node_preds.append({
        'idx': row.name,
        'pred_node': pred_path,
    })

node_df = pd.DataFrame(node_preds).set_index('idx')
t_node = t.join(node_df, how='inner')
t_node['err_node'] = np.abs(t_node['actual'] - t_node['pred_node'])

node_coverage = len(t_node) / len(t) * 100
print(f"  Trade coverage: {len(t_node)}/{len(t)} ({node_coverage:.1f}%)")

if len(t_node) > 100:
    # Compare on covered subset
    print(f"\n  On covered trades ({len(t_node)} trades):")
    for label, col in [('Current', 'err_current'), ('MW raw', 'err_mw_raw'), ('Node traceback', 'err_node')]:
        mae = t_node[col].mean()
        win = (t_node[col] < t_node['err_current']).mean() * 100 if col != 'err_current' else 0
        print(f"    {label:20s}: MAE={mae:.2f}, Win%={win:.1f}")

    # Per-year
    print(f"\n  Per-year (covered trades only):")
    print(f"  {'Year':>6} {'N':>6} {'Curr':>10} {'MW':>10} {'Node':>10} {'Node Imp%':>10}")
    for year in sorted(t_node['year'].unique()):
        yr = t_node[t_node['year'] == year]
        mc = yr['err_current'].mean()
        mm = yr['err_mw_raw'].mean()
        mn = yr['err_node'].mean()
        imp = (mc - mn) / mc * 100
        print(f"  PY{year} {len(yr):>6} {mc:>10.2f} {mm:>10.2f} {mn:>10.2f} {imp:>+10.1f}")

    # Node traceback + scaling: apply MW sum-correction to node predictions
    # Idea: node factors sum to ~X for each node. We can also try scaling.
    print(f"\n  Node traceback variants:")

    # V1: raw node factors (no scaling)
    # Already computed above as pred_node

    # V2: node factors scaled so sum matches MW global sum
    # For each trade, we already have pred_node. The MW global factor sum ≈ 0.88.
    # Node factor sums vary per node. Scaling would be: pred_node * (mw_sum / node_sum)
    node_scaled_preds = []
    for _, row in t_node.iterrows():
        yr = row['year']
        src, snk = row['src'], row['snk']
        nf = node_factors[yr]
        # Compute each node's factor sum
        src_sum = sum(nf[src][fx] for fx in range(12))
        snk_sum = sum(nf[snk][fx] for fx in range(12))
        # Don't scale individual nodes, just check distribution
        node_scaled_preds.append({
            'src_factor_sum': src_sum,
            'snk_factor_sum': snk_sum,
        })

    ns_df = pd.DataFrame(node_scaled_preds, index=t_node.index)
    print(f"    Node factor sums — src: mean={ns_df['src_factor_sum'].mean():.4f}, std={ns_df['src_factor_sum'].std():.4f}")
    print(f"    Node factor sums — snk: mean={ns_df['snk_factor_sum'].mean():.4f}, std={ns_df['snk_factor_sum'].std():.4f}")
    print(f"    (MW global sum ≈ 0.88 for reference)")

# ================================================================
# Q4: Does node-level traceback help on HIGH-RISK paths?
# ================================================================
print(f"\n{'=' * 70}")
print("Q4: NODE-LEVEL TRACEBACK on HIGH-RISK paths")
print(f"{'=' * 70}")

if len(t_node) > 100:
    # Define high-risk: paths with large current prediction error
    t_node['risk_bucket'] = pd.cut(
        t_node['err_current'],
        bins=[0, 50, 100, 200, 500, 1000, float('inf')],
        labels=['0-50', '50-100', '100-200', '200-500', '500-1000', '1000+']
    )

    print(f"\n  {'Bucket':>12} {'N':>6} {'Curr':>10} {'MW':>10} {'Node':>10} {'MW Imp%':>8} {'Node Imp%':>8}")
    for bucket in ['0-50', '50-100', '100-200', '200-500', '500-1000', '1000+']:
        b = t_node[t_node['risk_bucket'] == bucket]
        if len(b) < 10:
            continue
        mc = b['err_current'].mean()
        mm = b['err_mw_raw'].mean()
        mn = b['err_node'].mean()
        mw_imp = (mc - mm) / mc * 100
        n_imp = (mc - mn) / mc * 100
        print(f"  {bucket:>12} {len(b):>6} {mc:>10.2f} {mm:>10.2f} {mn:>10.2f} {mw_imp:>+8.1f} {n_imp:>+8.1f}")

    # Also by annual_mcp magnitude (larger paths = more risk)
    t_node['ann_bucket'] = pd.cut(
        t_node['annual_mcp'].abs(),
        bins=[0, 100, 500, 1000, 5000, float('inf')],
        labels=['0-100', '100-500', '500-1K', '1K-5K', '5K+']
    )
    print(f"\n  By annual MCP magnitude:")
    print(f"  {'|Ann MCP|':>12} {'N':>6} {'Curr':>10} {'MW':>10} {'Node':>10} {'MW Imp%':>8} {'Node Imp%':>8}")
    for bucket in ['0-100', '100-500', '500-1K', '1K-5K', '5K+']:
        b = t_node[t_node['ann_bucket'] == bucket]
        if len(b) < 10:
            continue
        mc = b['err_current'].mean()
        mm = b['err_mw_raw'].mean()
        mn = b['err_node'].mean()
        mw_imp = (mc - mm) / mc * 100
        n_imp = (mc - mn) / mc * 100
        print(f"  {bucket:>12} {len(b):>6} {mc:>10.2f} {mm:>10.2f} {mn:>10.2f} {mw_imp:>+8.1f} {n_imp:>+8.1f}")

# ================================================================
# Q5: Hybrid — MW for all, node traceback where available
# ================================================================
print(f"\n{'=' * 70}")
print("Q5: HYBRID — node traceback where available, MW elsewhere")
print(f"{'=' * 70}")

# For the full trade set, use node traceback if both nodes have factors,
# otherwise fall back to MW
t['pred_hybrid'] = t['mw_raw'].copy()
if len(t_node) > 0:
    t.loc[t_node.index, 'pred_hybrid'] = t_node['pred_node']
t['err_hybrid'] = np.abs(t['actual'] - t['pred_hybrid'])

mae_hybrid = t['err_hybrid'].mean()
print(f"\n  {'Method':30s} {'MAE':>10} {'vs Current':>12} {'Win%':>8}")
print(f"  {'Current':30s} {mae_curr:>10.2f} {'baseline':>12}")
print(f"  {'MW raw':30s} {mae_raw:>10.2f} {(mae_curr-mae_raw)/mae_curr*100:>+11.1f}% {(t['err_mw_raw']<t['err_current']).mean()*100:>8.1f}")
print(f"  {'Hybrid (node+MW)':30s} {mae_hybrid:>10.2f} {(mae_curr-mae_hybrid)/mae_curr*100:>+11.1f}% {(t['err_hybrid']<t['err_current']).mean()*100:>8.1f}")

print(f"\nFinal memory: {mem_mb():.0f} MB")

import ray
ray.shutdown()
