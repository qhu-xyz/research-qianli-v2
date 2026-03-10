"""Generate detailed per-quarter baseline comparison tables."""
import numpy as np
import polars as pl

QUARTERS = ["aq1", "aq2", "aq3", "aq4"]
DATA_DIR = "/opt/temp/qianli/annual_research/crossproduct_work"


def stats(mcp_arr, base_arr, label):
    mask = ~np.isnan(base_arr) & ~np.isnan(mcp_arr)
    n = int(mask.sum())
    if n == 0:
        return None
    m, b = mcp_arr[mask], base_arr[mask]
    res = m - b
    ae = np.abs(res)
    dm = (m != 0) & (b != 0)
    dir_pct = float(np.mean(np.sign(m[dm]) == np.sign(b[dm]))) * 100 if dm.sum() > 0 else 0
    big = np.abs(m) > 100
    dbig = big & dm
    dir100 = float(np.mean(np.sign(m[dbig]) == np.sign(b[dbig]))) * 100 if dbig.sum() > 0 else 0
    return {
        "label": label, "n": n, "bias": float(res.mean()),
        "mae": float(ae.mean()), "med_ae": float(np.median(ae)),
        "p95": float(np.percentile(ae, 95)), "p99": float(np.percentile(ae, 99)),
        "dir": dir_pct, "dir100": dir100,
    }


def print_table(rows, n_total, title):
    print(f"\n  {title}")
    hdr = (
        f"  {'Baseline':<20} | {'n':>8} | {'Cov%':>5} | {'Bias':>7} "
        f"| {'MAE':>6} | {'MedAE':>6} | {'p95':>7} | {'p99':>7} | {'Dir%':>5} | {'D>100':>5}"
    )
    sep = (
        f"  {'-'*20}-+-{'-'*8}-+-{'-'*5}-+-{'-'*7}"
        f"-+-{'-'*6}-+-{'-'*6}-+-{'-'*7}-+-{'-'*7}-+-{'-'*5}-+-{'-'*5}"
    )
    print(hdr)
    print(sep)
    for s in rows:
        if s is None:
            continue
        cov = s["n"] / n_total * 100
        print(
            f"  {s['label']:<20} | {s['n']:>8,} | {cov:>5.1f} | {s['bias']:>+7.0f} "
            f"| {s['mae']:>6.0f} | {s['med_ae']:>6.0f} | {s['p95']:>7.0f} | {s['p99']:>7.0f} "
            f"| {s['dir']:>5.1f} | {s['dir100']:>5.1f}"
        )


for q in QUARTERS:
    df = pl.read_parquet(f"{DATA_DIR}/{q}_all_baselines.parquet")
    df = df.filter(pl.col("planning_year") >= 2020)
    n_total = len(df)
    cols = df.columns
    mcp = df["mcp_mean"].to_numpy()

    baselines = [
        ("nodal_f0", "Nodal f0 stitch"),
        ("mtm_1st_mean", "H (DA congestion)"),
        ("f0_path_corr", "f0 path-level"),
        ("f1_path", "f1 path-level"),
        ("prior_r1_path", "Prior R1"),
        ("prior_r2_path", "Prior R2"),
        ("prior_r3_path", "Prior R3"),
    ]
    for qc in ["q2_path", "q3_path", "q4_path"]:
        if qc in cols:
            baselines.append((qc, qc.replace("_path", "").upper() + " forward"))

    rows = []
    for col, label in baselines:
        if col not in cols:
            continue
        arr = df[col].to_numpy().astype(float)
        rows.append(stats(mcp, arr, label))

    print(f"\n{'='*100}")
    print(f"  {q.upper()} — {n_total:,} rows, PY 2020-2025")
    print(f"{'='*100}")
    print_table(rows, n_total, "A. All rows (each baseline uses its own coverage)")

    # Head-to-head: nodal_f0 & f0_path_corr & H all present
    h2h = df.filter(
        pl.col("nodal_f0").is_not_null()
        & pl.col("f0_path_corr").is_not_null()
        & pl.col("mtm_1st_mean").is_not_null()
    )
    n_h2h = len(h2h)
    mcp_h = h2h["mcp_mean"].to_numpy()
    h2h_rows = []
    for col, label in [("nodal_f0", "Nodal f0"), ("f0_path_corr", "f0 path"), ("mtm_1st_mean", "H")]:
        h2h_rows.append(stats(mcp_h, h2h[col].to_numpy().astype(float), label))
    print_table(h2h_rows, n_h2h, f"B. Head-to-head ({n_h2h:,} rows where Nodal+f0path+H all present)")

    # Fully matched C4: all baselines present
    c4_cols = ["nodal_f0", "f0_path_corr", "f1_path", "prior_r3_path", "mtm_1st_mean"]
    c4 = df.filter(pl.all_horizontal([pl.col(c).is_not_null() for c in c4_cols]))
    n_c4 = len(c4)
    mcp_c4 = c4["mcp_mean"].to_numpy()
    c4_rows = []
    for col, label in [
        ("nodal_f0", "Nodal f0"), ("f0_path_corr", "f0 path"),
        ("f1_path", "f1 path"), ("prior_r3_path", "Prior R3"),
        ("prior_r2_path", "Prior R2"), ("prior_r1_path", "Prior R1"),
        ("mtm_1st_mean", "H"),
    ]:
        if col in c4.columns and c4[col].null_count() < n_c4:
            c4_rows.append(stats(mcp_c4, c4[col].to_numpy().astype(float), label))
    print_table(c4_rows, n_c4, f"C. Fully matched ({n_c4:,} rows where Nodal+f0+f1+R3+H all present)")

    # Phase 2 improvement: alpha scaling on all rows with nodal_f0
    valid = df.filter(pl.col("nodal_f0").is_not_null())
    n_v = len(valid)
    mcp_v = valid["mcp_mean"].to_numpy()
    f0_v = valid["nodal_f0"].to_numpy().astype(float)
    p2_rows = [stats(mcp_v, f0_v, "Nodal f0 (raw)")]
    for alpha in [1.45, 1.55, 1.60]:
        p2_rows.append(stats(mcp_v, f0_v * alpha, f"alpha={alpha:.2f}"))
    print_table(p2_rows, n_v, f"D. Phase 2: Alpha scaling ({n_v:,} rows with nodal_f0)")

    del df
