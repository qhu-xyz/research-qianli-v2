"""
V2 Alternative Approaches: Move beyond ratio decomposition entirely.

New ideas:
1. Linear regression: fx_mcp = alpha_fx + beta_fx * annual_mcp (per month)
   - Doesn't assume ratio = 1/12, captures systematic bias via intercept
2. Previous year's monthly MCP as direct predictor (ignore annual)
3. Blend: weighted combination of annual/12 and prev-year monthly MCP
4. Regression with prev-year monthly MCP as additional feature
5. Market-wide seasonal factor (single 12-vector applied to all paths)

All approaches operate at PATH level (sink - source already computed).
Backtest on cleared paths, then verify on trades.
"""
import os
import gc
import resource
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, Ridge

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
# PHASE 1: Load path-level data (annual + monthly MCPs per path)
# ================================================================
print("PHASE 1: Loading path-level data...")

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
print("\nBuilding path-level records...")
year_paths = {}  # year -> DataFrame with columns: src, snk, annual_mcp, f0..f11

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
        rec['monthly_sum'] = sum(fx_vals)
        records.append(rec)

    year_paths[year] = pd.DataFrame(records)
    print(f"  PY{year}: {len(records)} paths")
    del cleared
    gc.collect()

# ================================================================
# PHASE 2: Test alternative approaches
# ================================================================
print(f"\n{'=' * 70}")
print("PHASE 2: Alternative approaches")
print(f"{'=' * 70}")

all_results = []

for test_year in range(2020, 2026):
    # Training data: all years before test_year
    train_years = [y for y in YEARS if y < test_year]

    df_test = year_paths[test_year].set_index(['src', 'snk'])
    prev_year = test_year - 1
    df_prev = year_paths[prev_year].set_index(['src', 'snk']) if prev_year in year_paths else None

    # Build training set for regression
    train_rows = []
    for ty in train_years:
        df_ty = year_paths[ty]
        for _, r in df_ty.iterrows():
            for fx in range(12):
                train_rows.append({
                    'annual_mcp': r['annual_mcp'],
                    'fx': fx,
                    'monthly_mcp': r[f'f{fx}'],
                })
    train_df = pd.DataFrame(train_rows)

    # === Approach 1: Per-month linear regression ===
    # fx_mcp = alpha_fx + beta_fx * annual_mcp
    reg_models = {}
    for fx in range(12):
        mask = train_df['fx'] == fx
        X = train_df.loc[mask, ['annual_mcp']].values
        y = train_df.loc[mask, 'monthly_mcp'].values
        model = Ridge(alpha=1.0)
        model.fit(X, y)
        reg_models[fx] = model

    # === Approach 2: Per-month regression with prev-year monthly as feature ===
    # fx_mcp = alpha + beta1 * annual + beta2 * prev_year_fx
    reg2_models = {}
    if prev_year in year_paths:
        train_rows2 = []
        for ty in train_years:
            ty_prev = ty - 1
            if ty_prev not in year_paths:
                continue
            df_ty = year_paths[ty].set_index(['src', 'snk'])
            df_ty_prev = year_paths[ty_prev].set_index(['src', 'snk'])
            common = set(df_ty.index) & set(df_ty_prev.index)
            for path in common:
                r = df_ty.loc[path]
                r_prev = df_ty_prev.loc[path]
                for fx in range(12):
                    train_rows2.append({
                        'annual_mcp': r['annual_mcp'],
                        'prev_fx': r_prev[f'f{fx}'],
                        'prev_annual': r_prev['annual_mcp'],
                        'fx': fx,
                        'monthly_mcp': r[f'f{fx}'],
                    })
        if train_rows2:
            train_df2 = pd.DataFrame(train_rows2)
            for fx in range(12):
                mask = train_df2['fx'] == fx
                X = train_df2.loc[mask, ['annual_mcp', 'prev_fx', 'prev_annual']].values
                y = train_df2.loc[mask, 'monthly_mcp'].values
                model = Ridge(alpha=1.0)
                model.fit(X, y)
                reg2_models[fx] = model

    # === Approach 5: Market-wide seasonal factor ===
    # Compute average ratio across all training paths
    mw_factors = {}
    for fx in range(12):
        mask = (train_df['fx'] == fx) & (train_df['annual_mcp'].abs() > 10)
        ratios = train_df.loc[mask, 'monthly_mcp'] / train_df.loc[mask, 'annual_mcp']
        # Clip extreme ratios before averaging
        ratios_clipped = ratios.clip(-0.5, 0.5)
        mw_factors[fx] = ratios_clipped.median()

    # === Generate predictions for test year ===
    common_with_prev = set(df_test.index) & set(df_prev.index) if df_prev is not None else set()

    for path in df_test.index:
        row = df_test.loc[path]
        ann = row['annual_mcp']
        if abs(ann) < 10:
            continue

        has_prev = path in common_with_prev

        for fx in range(12):
            actual = row[f'f{fx}']
            naive = ann / 12.0

            # Approach 1: regression (annual only)
            pred_reg = float(reg_models[fx].predict([[ann]])[0])

            # Approach 2: regression with prev-year features
            pred_reg2 = np.nan
            if has_prev and fx in reg2_models:
                prev_row = df_prev.loc[path]
                pred_reg2 = float(reg2_models[fx].predict(
                    [[ann, prev_row[f'f{fx}'], prev_row['annual_mcp']]]
                )[0])

            # Approach 3: previous year's monthly MCP directly
            pred_prev = np.nan
            if has_prev:
                pred_prev = df_prev.loc[path][f'f{fx}']

            # Approach 4: blend naive + prev_year
            pred_blend_prev = np.nan
            if has_prev:
                pred_blend_prev = 0.5 * naive + 0.5 * pred_prev

            # Approach 5: market-wide factor
            pred_mw = ann * mw_factors[fx]

            # Approach 6: blend naive + regression
            pred_blend_reg = 0.5 * naive + 0.5 * pred_reg

            all_results.append({
                'year': test_year, 'fx': fx, 'actual': actual,
                'annual_mcp': ann, 'has_prev': has_prev,
                'pred_naive': naive,
                'pred_reg': pred_reg,
                'pred_reg2': pred_reg2,
                'pred_prev': pred_prev,
                'pred_blend_prev': pred_blend_prev,
                'pred_mw': pred_mw,
                'pred_blend_reg': pred_blend_reg,
            })

    print(f"  PY{test_year}: done, {len([r for r in all_results if r['year'] == test_year])} samples")

results = pd.DataFrame(all_results)
print(f"\nTotal: {len(results)} samples")

# ================================================================
# EVALUATION
# ================================================================
print(f"\n{'=' * 70}")
print("EVALUATION")
print(f"{'=' * 70}")

def eval_methods(df, label=""):
    if label:
        print(f"\n  {label} ({len(df)} samples)")
    methods = [
        ('Naive /12', 'pred_naive'),
        ('Regression (ann)', 'pred_reg'),
        ('Blend 50 N+50 reg', 'pred_blend_reg'),
        ('Market-wide factor', 'pred_mw'),
    ]
    # Only eval prev-year methods on paths with prev data
    has_prev = df[df['has_prev']]
    methods_prev = [
        ('Prev year monthly', 'pred_prev'),
        ('Blend 50 N+50 prev', 'pred_blend_prev'),
        ('Reg (ann+prev)', 'pred_reg2'),
    ]

    for name, col in methods:
        err = np.abs(df['actual'] - df[col])
        err_naive = np.abs(df['actual'] - df['pred_naive'])
        mae = err.mean()
        win = (err < err_naive).mean() * 100
        print(f"    {name:22s}  MAE={mae:8.2f}  WinVsNaive={win:5.1f}%")

    if len(has_prev) > 0:
        print(f"    --- (paths with prev year data: {len(has_prev)}) ---")
        # Re-eval naive on same subset for fair comparison
        err_naive_sub = np.abs(has_prev['actual'] - has_prev['pred_naive'])
        mae_naive_sub = err_naive_sub.mean()
        print(f"    {'Naive /12 (subset)':22s}  MAE={mae_naive_sub:8.2f}")
        for name, col in methods_prev:
            sub = has_prev.dropna(subset=[col])
            if len(sub) == 0:
                continue
            err = np.abs(sub['actual'] - sub[col])
            err_naive = np.abs(sub['actual'] - sub['pred_naive'])
            mae = err.mean()
            win = (err < err_naive).mean() * 100
            print(f"    {name:22s}  MAE={mae:8.2f}  WinVsNaive={win:5.1f}%")

eval_methods(results, "Overall")

for year in sorted(results['year'].unique()):
    eval_methods(results[results['year'] == year], f"PY{year}")

# Per month comparison
print(f"\n  --- Per month ---")
print(f"  {'fx':<4} {'Mon':<4} {'Naive':>8} {'Reg':>8} {'MW':>8} {'Prev':>8} {'B_prev':>8} {'Reg2':>8}")
for fx in range(12):
    fdf = results[results['fx'] == fx]
    m_naive = np.abs(fdf['actual'] - fdf['pred_naive']).mean()
    m_reg = np.abs(fdf['actual'] - fdf['pred_reg']).mean()
    m_mw = np.abs(fdf['actual'] - fdf['pred_mw']).mean()

    has_prev = fdf[fdf['has_prev']].dropna(subset=['pred_prev'])
    m_prev = np.abs(has_prev['actual'] - has_prev['pred_prev']).mean() if len(has_prev) > 0 else np.nan
    m_bprev = np.abs(has_prev['actual'] - has_prev['pred_blend_prev']).mean() if len(has_prev) > 0 else np.nan
    m_reg2 = np.abs(has_prev['actual'] - has_prev['pred_reg2']).mean() if len(has_prev) > 0 else np.nan

    print(f"  f{fx:<3} {CAL_MONTHS[fx]:<4} {m_naive:>8.2f} {m_reg:>8.2f} {m_mw:>8.2f} "
          f"{m_prev:>8.2f} {m_bprev:>8.2f} {m_reg2:>8.2f}")

# ================================================================
# Regression coefficients analysis
# ================================================================
print(f"\n{'=' * 70}")
print("REGRESSION COEFFICIENTS (trained on 2017-2024)")
print(f"{'=' * 70}")

# Retrain on all data for final analysis
all_train = []
for y in range(2017, 2025):
    df_y = year_paths[y]
    for _, r in df_y.iterrows():
        for fx in range(12):
            all_train.append({
                'annual_mcp': r['annual_mcp'], 'fx': fx, 'monthly_mcp': r[f'f{fx}'],
            })
all_train_df = pd.DataFrame(all_train)

print(f"\n  {'fx':<4} {'Mon':<4} {'alpha':>10} {'beta':>10} {'R2':>8} {'note':>20}")
for fx in range(12):
    mask = all_train_df['fx'] == fx
    X = all_train_df.loc[mask, ['annual_mcp']].values
    y = all_train_df.loc[mask, 'monthly_mcp'].values
    model = Ridge(alpha=1.0)
    model.fit(X, y)
    r2 = model.score(X, y)
    alpha = model.intercept_
    beta = model.coef_[0]
    note = ""
    if abs(beta - 1/12) < 0.01:
        note = "~= naive 1/12"
    elif beta > 1/12:
        note = f"slope > 1/12 by {(beta - 1/12)*12:.1%}"
    else:
        note = f"slope < 1/12 by {(1/12 - beta)*12:.1%}"
    print(f"  f{fx:<3} {CAL_MONTHS[fx]:<4} {alpha:>10.2f} {beta:>10.6f} {r2:>8.4f} {note:>20}")

# ================================================================
# PHASE 3: Trades verification (best methods only)
# ================================================================
print(f"\n{'=' * 70}")
print("PHASE 3: Trades verification")
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
del trades, june_trades
gc.collect()

# Build prediction lookup from results DataFrame
# Convert path-level results to a lookup
pred_lookup = {}
for _, r in results.iterrows():
    # Need to figure out src/snk from the results... let me rebuild
    pass

# Rebuild predictions for trades directly
matched_trades = []
for test_year in range(2020, 2026):
    train_years = [y for y in YEARS if y < test_year]
    prev_year = test_year - 1

    # Retrain regression for this test year
    train_rows = []
    for ty in train_years:
        df_ty = year_paths[ty]
        for _, r in df_ty.iterrows():
            for fx in range(12):
                train_rows.append({
                    'annual_mcp': r['annual_mcp'],
                    'fx': fx,
                    'monthly_mcp': r[f'f{fx}'],
                })
    train_df = pd.DataFrame(train_rows)

    reg_models = {}
    for fx in range(12):
        mask = train_df['fx'] == fx
        X = train_df.loc[mask, ['annual_mcp']].values
        y_vals = train_df.loc[mask, 'monthly_mcp'].values
        model = Ridge(alpha=1.0)
        model.fit(X, y_vals)
        reg_models[fx] = model

    # Regression with prev-year
    reg2_models = {}
    train_rows2 = []
    for ty in train_years:
        ty_prev = ty - 1
        if ty_prev not in year_paths:
            continue
        df_ty = year_paths[ty].set_index(['src', 'snk'])
        df_ty_prev = year_paths[ty_prev].set_index(['src', 'snk'])
        common = set(df_ty.index) & set(df_ty_prev.index)
        for path in common:
            r = df_ty.loc[path]
            r_prev = df_ty_prev.loc[path]
            for fx in range(12):
                train_rows2.append({
                    'annual_mcp': r['annual_mcp'],
                    'prev_fx': r_prev[f'f{fx}'],
                    'prev_annual': r_prev['annual_mcp'],
                    'fx': fx,
                    'monthly_mcp': r[f'f{fx}'],
                })
    if train_rows2:
        train_df2 = pd.DataFrame(train_rows2)
        for fx in range(12):
            mask = train_df2['fx'] == fx
            X = train_df2.loc[mask, ['annual_mcp', 'prev_fx', 'prev_annual']].values
            y_vals = train_df2.loc[mask, 'monthly_mcp'].values
            model = Ridge(alpha=1.0)
            model.fit(X, y_vals)
            reg2_models[fx] = model

    # Market-wide factors
    mw_factors = {}
    for fx in range(12):
        mask = (train_df['fx'] == fx) & (train_df['annual_mcp'].abs() > 10)
        ratios = train_df.loc[mask, 'monthly_mcp'] / train_df.loc[mask, 'annual_mcp']
        mw_factors[fx] = ratios.clip(-0.5, 0.5).median()

    # Previous year path data
    df_prev = year_paths[prev_year].set_index(['src', 'snk']) if prev_year in year_paths else None

    ann = annual_nodes[test_year]
    yr_trades = mapped[mapped['year'] == test_year]

    for _, trow in yr_trades.iterrows():
        src, snk = trow['src_name'], trow['snk_name']
        fx = trow['fx']

        if src not in ann.index or snk not in ann.index:
            continue
        ann_path = float(ann[snk] - ann[src])
        if abs(ann_path) < 1e-6:
            continue

        naive = ann_path / 12.0
        pred_reg = float(reg_models[fx].predict([[ann_path]])[0])
        pred_mw = ann_path * mw_factors[fx]
        pred_blend_reg = 0.5 * naive + 0.5 * pred_reg

        rec = {
            'year': test_year, 'fx': fx, 'path': trow['path'],
            'actual_mcp': trow['mcp_mean'],
            'current_mtm1st': trow['mtm_1st_mean'],
            'pred_naive': naive,
            'pred_reg': pred_reg,
            'pred_mw': pred_mw,
            'pred_blend_reg': pred_blend_reg,
        }

        # Previous year methods
        if df_prev is not None and (src, snk) in df_prev.index:
            prev_row = df_prev.loc[(src, snk)]
            rec['pred_prev'] = prev_row[f'f{fx}']
            rec['pred_blend_prev'] = 0.5 * naive + 0.5 * prev_row[f'f{fx}']
            if fx in reg2_models:
                rec['pred_reg2'] = float(reg2_models[fx].predict(
                    [[ann_path, prev_row[f'f{fx}'], prev_row['annual_mcp']]]
                )[0])

        matched_trades.append(rec)

t = pd.DataFrame(matched_trades)
if len(t) > 0:
    t = t.drop_duplicates(subset=['path', 'year', 'fx'])
    print(f"\n  Matched trades (dedup): {len(t)}, unique paths: {t['path'].nunique()}")

    print(f"\n  --- Overall ---")
    for label, col in [('Current mtm_1st', 'current_mtm1st'),
                         ('Our Naive /12', 'pred_naive'),
                         ('Regression (ann)', 'pred_reg'),
                         ('Blend 50 N+50 reg', 'pred_blend_reg'),
                         ('Market-wide factor', 'pred_mw')]:
        err = np.abs(t['actual_mcp'] - t[col])
        err_curr = np.abs(t['actual_mcp'] - t['current_mtm1st'])
        mae = err.mean()
        win = (err < err_curr).mean() * 100
        print(f"    {label:22s}  MAE={mae:8.2f}  WinVsCurrent={win:5.1f}%")

    # Methods requiring prev year
    has_prev = t.dropna(subset=['pred_prev'])
    if len(has_prev) > 0:
        print(f"\n    --- Paths with prev year ({len(has_prev)} trades) ---")
        err_curr_sub = np.abs(has_prev['actual_mcp'] - has_prev['current_mtm1st'])
        print(f"    {'Current (subset)':22s}  MAE={err_curr_sub.mean():8.2f}")
        for label, col in [('Prev year monthly', 'pred_prev'),
                             ('Blend 50 N+50 prev', 'pred_blend_prev'),
                             ('Reg (ann+prev)', 'pred_reg2')]:
            sub = has_prev.dropna(subset=[col])
            err = np.abs(sub['actual_mcp'] - sub[col])
            err_curr = np.abs(sub['actual_mcp'] - sub['current_mtm1st'])
            mae = err.mean()
            win = (err < err_curr).mean() * 100
            print(f"    {label:22s}  MAE={mae:8.2f}  WinVsCurrent={win:5.1f}%")

    # Per year
    for year in sorted(t['year'].unique()):
        yr = t[t['year'] == year]
        print(f"\n    PY{year} ({len(yr)} trades):")
        for label, col in [('Current mtm_1st', 'current_mtm1st'),
                             ('Regression (ann)', 'pred_reg'),
                             ('Blend 50 N+50 reg', 'pred_blend_reg')]:
            err = np.abs(yr['actual_mcp'] - yr[col])
            err_curr = np.abs(yr['actual_mcp'] - yr['current_mtm1st'])
            mae = err.mean()
            win = (err < err_curr).mean() * 100
            print(f"      {label:22s}  MAE={mae:8.2f}  WinVsCurrent={win:5.1f}%")
        # Prev year methods
        hp = yr.dropna(subset=['pred_reg2'])
        if len(hp) > 0:
            for label, col in [('Reg (ann+prev)', 'pred_reg2'),
                                 ('Blend 50 N+50 prev', 'pred_blend_prev')]:
                sub = hp.dropna(subset=[col])
                if len(sub) > 0:
                    err = np.abs(sub['actual_mcp'] - sub[col])
                    err_curr = np.abs(sub['actual_mcp'] - sub['current_mtm1st'])
                    mae = err.mean()
                    win = (err < err_curr).mean() * 100
                    print(f"      {label:22s}  MAE={mae:8.2f}  WinVsCurrent={win:5.1f}% ({len(sub)} trades)")

    # Per month
    print(f"\n  --- Per month on trades ---")
    print(f"  {'fx':<4} {'Mon':<4} {'Current':>10} {'Naive':>10} {'Reg':>10} {'B_reg':>10} {'MW':>10}")
    for fx in range(12):
        fdf = t[t['fx'] == fx]
        if len(fdf) == 0:
            continue
        m_curr = np.abs(fdf['actual_mcp'] - fdf['current_mtm1st']).mean()
        m_naive = np.abs(fdf['actual_mcp'] - fdf['pred_naive']).mean()
        m_reg = np.abs(fdf['actual_mcp'] - fdf['pred_reg']).mean()
        m_breg = np.abs(fdf['actual_mcp'] - fdf['pred_blend_reg']).mean()
        m_mw = np.abs(fdf['actual_mcp'] - fdf['pred_mw']).mean()
        print(f"  f{fx:<3} {CAL_MONTHS[fx]:<4} {m_curr:>10.2f} {m_naive:>10.2f} {m_reg:>10.2f} {m_breg:>10.2f} {m_mw:>10.2f}")

print(f"\nFinal memory: {mem_mb():.0f} MB")

import ray
ray.shutdown()
