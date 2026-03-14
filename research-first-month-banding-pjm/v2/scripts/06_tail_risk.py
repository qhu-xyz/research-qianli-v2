"""
V2 Tail risk analysis: Does MW factor improve or worsen the worst predictions?

Metrics:
- P90, P95, P99 absolute error (tail risk)
- Max absolute error
- % of trades where |error| > 2x median (outlier rate)
- Worst-case per year and per month
- Error distribution comparison (current vs MW)
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

# Load data (same as before)
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

    del annual
    gc.collect()

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

def compute_mw_factors(train_years, method='median', min_ann=10):
    all_ratios = {fx: [] for fx in range(12)}
    for ty in train_years:
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

# Load trades
print("Loading trades...")
import polars as pl

id_to_name = {}
for yr in range(2020, 2026):
    cdata = cleared_loader.load_data(auction_month=f'{yr}-06-01')
    for _, row in cdata[['source_id', 'source_name']].drop_duplicates().iterrows():
        if pd.notna(row['source_id']) and pd.notna(row['source_name']):
            id_to_name[str(row['source_id'])] = row['source_name']
    for _, row in cdata[['sink_id', 'sink_name']].drop_duplicates().iterrows():
        if pd.notna(row['sink_id']) and pd.notna(row['sink_name']):
            id_to_name[str(row['sink_id'])] = row['sink_name']

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

# Build predictions
matched = []
for test_year in range(2020, 2026):
    train_yrs = [y for y in YEARS if y < test_year]
    factors = compute_mw_factors(train_yrs, method='median')
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
        pred_mw = ann_path * factors[fx_val]

        matched.append({
            'year': test_year, 'fx': fx_val, 'path': trow['path'],
            'actual': trow['mcp_mean'],
            'current': trow['mtm_1st_mean'],
            'naive': naive,
            'mw': pred_mw,
            'annual_mcp': ann_path,
        })

t = pd.DataFrame(matched).drop_duplicates(subset=['path', 'year', 'fx'])
t['err_current'] = np.abs(t['actual'] - t['current'])
t['err_naive'] = np.abs(t['actual'] - t['naive'])
t['err_mw'] = np.abs(t['actual'] - t['mw'])

print(f"\nTrades: {len(t)}, paths: {t['path'].nunique()}")

# ================================================================
# TAIL RISK: Percentile comparison
# ================================================================
print(f"\n{'=' * 70}")
print("TAIL RISK: Error distribution comparison")
print(f"{'=' * 70}")

percentiles = [50, 75, 90, 95, 99, 99.5]
print(f"\n  {'Percentile':>12} {'Current':>10} {'Naive':>10} {'MW':>10} {'MW vs Curr':>12}")
for p in percentiles:
    ec = np.percentile(t['err_current'], p)
    en = np.percentile(t['err_naive'], p)
    em = np.percentile(t['err_mw'], p)
    diff_pct = (ec - em) / ec * 100
    print(f"  {'P'+str(p):>12} {ec:>10.2f} {en:>10.2f} {em:>10.2f} {diff_pct:>+11.1f}%")

print(f"\n  {'Max':>12} {t['err_current'].max():>10.2f} {t['err_naive'].max():>10.2f} {t['err_mw'].max():>10.2f}")
print(f"  {'Mean':>12} {t['err_current'].mean():>10.2f} {t['err_naive'].mean():>10.2f} {t['err_mw'].mean():>10.2f}")

# ================================================================
# OUTLIER RATE
# ================================================================
print(f"\n{'=' * 70}")
print("OUTLIER RATES")
print(f"{'=' * 70}")

for threshold_label, threshold in [('> 500', 500), ('> 1000', 1000), ('> 2000', 2000), ('> 5000', 5000)]:
    rate_curr = (t['err_current'] > threshold).mean() * 100
    rate_mw = (t['err_mw'] > threshold).mean() * 100
    print(f"  |error| {threshold_label:>6}: Current={rate_curr:5.2f}%  MW={rate_mw:5.2f}%  diff={rate_mw - rate_curr:+.2f}%")

# ================================================================
# WIN RATE BY ERROR BUCKET
# ================================================================
print(f"\n{'=' * 70}")
print("WIN RATE BY CURRENT ERROR SIZE (does MW help more on large errors?)")
print(f"{'=' * 70}")

t['err_bucket'] = pd.cut(t['err_current'], bins=[0, 50, 100, 200, 500, 1000, float('inf')],
                          labels=['0-50', '50-100', '100-200', '200-500', '500-1000', '1000+'])
print(f"\n  {'Bucket':>12} {'Count':>8} {'Curr MAE':>10} {'MW MAE':>10} {'Win%':>8} {'Imp%':>8}")
for bucket in ['0-50', '50-100', '100-200', '200-500', '500-1000', '1000+']:
    b = t[t['err_bucket'] == bucket]
    if len(b) == 0:
        continue
    ec = b['err_current'].mean()
    em = b['err_mw'].mean()
    win = (b['err_mw'] < b['err_current']).mean() * 100
    imp = (ec - em) / ec * 100
    print(f"  {bucket:>12} {len(b):>8} {ec:>10.2f} {em:>10.2f} {win:>8.1f} {imp:>+8.1f}")

# ================================================================
# TAIL CASES: Where MW is WORSE than current
# ================================================================
print(f"\n{'=' * 70}")
print("CASES WHERE MW IS WORSE (MW error > current error)")
print(f"{'=' * 70}")

t['mw_worse_by'] = t['err_mw'] - t['err_current']
worse = t[t['mw_worse_by'] > 0].copy()
print(f"\n  MW worse on {len(worse)}/{len(t)} trades ({len(worse)/len(t)*100:.1f}%)")
print(f"  When MW is worse, avg extra error: {worse['mw_worse_by'].mean():.2f}")

better = t[t['mw_worse_by'] < 0].copy()
print(f"  When MW is better, avg saved error: {(-better['mw_worse_by']).mean():.2f}")

print(f"\n  Worst MW degradations (top 20):")
worst20 = worse.nlargest(20, 'mw_worse_by')
print(f"  {'Year':>4} {'fx':>3} {'Actual':>10} {'Current':>10} {'MW':>10} {'Curr_err':>10} {'MW_err':>10} {'Worse_by':>10}")
for _, r in worst20.iterrows():
    print(f"  {r['year']:>4} f{r['fx']:<2} {r['actual']:>10.1f} {r['current']:>10.1f} {r['mw']:>10.1f} "
          f"{r['err_current']:>10.1f} {r['err_mw']:>10.1f} {r['mw_worse_by']:>10.1f}")

# ================================================================
# PER-YEAR TAIL RISK
# ================================================================
print(f"\n{'=' * 70}")
print("PER-YEAR TAIL RISK (P95)")
print(f"{'=' * 70}")

print(f"\n  {'Year':>6} {'Curr P95':>10} {'MW P95':>10} {'Imp%':>8} {'Curr P99':>10} {'MW P99':>10} {'Imp%':>8}")
for year in sorted(t['year'].unique()):
    yr = t[t['year'] == year]
    c95 = np.percentile(yr['err_current'], 95)
    m95 = np.percentile(yr['err_mw'], 95)
    c99 = np.percentile(yr['err_current'], 99)
    m99 = np.percentile(yr['err_mw'], 99)
    imp95 = (c95 - m95) / c95 * 100
    imp99 = (c99 - m99) / c99 * 100
    print(f"  PY{year} {c95:>10.2f} {m95:>10.2f} {imp95:>+8.1f} {c99:>10.2f} {m99:>10.2f} {imp99:>+8.1f}")

# ================================================================
# PER-MONTH TAIL RISK
# ================================================================
print(f"\n{'=' * 70}")
print("PER-MONTH TAIL RISK (P95)")
print(f"{'=' * 70}")

print(f"\n  {'fx':<4} {'Mon':<4} {'Curr P95':>10} {'MW P95':>10} {'Imp%':>8} {'Curr P99':>10} {'MW P99':>10} {'Imp%':>8}")
for fx in range(12):
    fdf = t[t['fx'] == fx]
    if len(fdf) == 0:
        continue
    c95 = np.percentile(fdf['err_current'], 95)
    m95 = np.percentile(fdf['err_mw'], 95)
    c99 = np.percentile(fdf['err_current'], 99)
    m99 = np.percentile(fdf['err_mw'], 99)
    imp95 = (c95 - m95) / c95 * 100
    imp99 = (c99 - m99) / c99 * 100
    print(f"  f{fx:<3} {CAL_MONTHS[fx]:<4} {c95:>10.2f} {m95:>10.2f} {imp95:>+8.1f} {c99:>10.2f} {m99:>10.2f} {imp99:>+8.1f}")

# ================================================================
# SIGNED ERROR DISTRIBUTION
# ================================================================
print(f"\n{'=' * 70}")
print("SIGNED ERROR (actual - pred): bias direction")
print(f"{'=' * 70}")

se_curr = t['actual'] - t['current']
se_mw = t['actual'] - t['mw']

print(f"\n  {'':>20} {'Current':>10} {'MW':>10}")
print(f"  {'Mean':>20} {se_curr.mean():>10.2f} {se_mw.mean():>10.2f}")
print(f"  {'Median':>20} {se_curr.median():>10.2f} {se_mw.median():>10.2f}")
print(f"  {'Std':>20} {se_curr.std():>10.2f} {se_mw.std():>10.2f}")
print(f"  {'Skew':>20} {se_curr.skew():>10.3f} {se_mw.skew():>10.3f}")

# What fraction over/under predicts?
print(f"\n  Over-predicts (pred > actual): Current={( se_curr < 0).mean()*100:.1f}%  MW={(se_mw < 0).mean()*100:.1f}%")
print(f"  Under-predicts (pred < actual): Current={(se_curr > 0).mean()*100:.1f}%  MW={(se_mw > 0).mean()*100:.1f}%")

print(f"\nFinal memory: {mem_mb():.0f} MB")

import ray
ray.shutdown()
