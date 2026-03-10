"""Read v0-reeval and v0007-reeval metrics.json, produce comparison table."""
import json
from pathlib import Path

v0 = json.loads(Path("registry/v0-reeval/metrics.json").read_text())
v7 = json.loads(Path("registry/v0007-reeval/metrics.json").read_text())

v0_pm = v0["per_month"]
v7_pm = v7["per_month"]
v0_agg = v0["aggregate"]["mean"]
v7_agg = v7["aggregate"]["mean"]

months = sorted(set(v0_pm.keys()) & set(v7_pm.keys()))
metrics = ["EV-VC@100", "EV-VC@500", "EV-NDCG", "Spearman", "C-RMSE", "C-MAE"]
higher_better = {"EV-VC@100": True, "EV-VC@500": True, "EV-NDCG": True,
                 "Spearman": True, "C-RMSE": False, "C-MAE": False}

print(f"\n{'='*100}")
print(f"  v0 vs v0007 — Full 2020-2022 Comparison ({len(months)} months)")
print(f"{'='*100}")

# Per-month table (key metrics)
key_metrics = ["EV-VC@100", "EV-VC@500", "Spearman", "C-RMSE"]
hdr = f"{'Month':>8}"
for m in key_metrics:
    hdr += f"  {'v0':>8} {'v7':>8} {'delta':>8}"
print(f"\n{hdr}")
print("-" * 100)

wins = {m: 0 for m in metrics}
for month in months:
    row = f"{month:>8}"
    for m in key_metrics:
        val0 = v0_pm[month].get(m, float("nan"))
        val7 = v7_pm[month].get(m, float("nan"))
        if m in ("C-RMSE", "C-MAE"):
            d = val0 - val7  # lower better, positive = v7 wins
            if d > 0:
                wins[m] += 1
            row += f"  {val0:>8.0f} {val7:>8.0f} {d:>+8.0f}"
        else:
            d = val7 - val0
            if d > 0:
                wins[m] += 1
            row += f"  {val0:>8.4f} {val7:>8.4f} {d:>+8.4f}"
    print(row)

n = len(months)
print("-" * 100)
row = f"{'v7 wins':>8}"
for m in key_metrics:
    row += f"  {'':>8} {'':>8} {f'{wins[m]}/{n}':>8}"
print(row)

# Aggregate
print(f"\n{'='*80}")
print(f"  AGGREGATE (mean across {n} months)")
print(f"{'='*80}")
print(f"  {'Metric':<15} {'v0':>12} {'v0007':>12} {'Delta':>10} {'%':>8} {'Winner':>8}")
print(f"  {'-'*70}")
for m in metrics:
    val0 = v0_agg.get(m, float("nan"))
    val7 = v7_agg.get(m, float("nan"))
    hb = higher_better.get(m, True)
    d = (val7 - val0) if hb else (val0 - val7)
    pct = d / abs(val0) * 100 if val0 != 0 else 0
    w = "v0007" if d > 0 else "v0"
    if m in ("C-RMSE", "C-MAE"):
        print(f"  {m:<15} {val0:>12.1f} {val7:>12.1f} {d:>+10.1f} {pct:>+7.1f}% {w:>8}")
    else:
        print(f"  {m:<15} {val0:>12.4f} {val7:>12.4f} {d:>+10.4f} {pct:>+7.1f}% {w:>8}")
print(f"{'='*80}")

# Yearly breakdown
for year in [2020, 2021, 2022]:
    ym = [m for m in months if m.startswith(str(year))]
    if not ym:
        continue
    print(f"\n  --- {year} ({len(ym)} months) ---")
    for m in ["EV-VC@100", "EV-VC@500", "Spearman", "C-RMSE"]:
        vals0 = [v0_pm[mo].get(m, 0) for mo in ym]
        vals7 = [v7_pm[mo].get(m, 0) for mo in ym]
        mean0 = sum(vals0) / len(vals0)
        mean7 = sum(vals7) / len(vals7)
        hb = higher_better.get(m, True)
        d = (mean7 - mean0) if hb else (mean0 - mean7)
        pct = d / abs(mean0) * 100 if mean0 != 0 else 0
        w = "v7" if d > 0 else "v0"
        if m in ("C-RMSE", "C-MAE"):
            print(f"  {m:<15} v0={mean0:>8.0f}  v7={mean7:>8.0f}  delta={d:>+8.0f} ({pct:>+.1f}%) {w}")
        else:
            print(f"  {m:<15} v0={mean0:>8.4f}  v7={mean7:>8.4f}  delta={d:>+8.4f} ({pct:>+.1f}%) {w}")

print(f"\n  Months evaluated: {n}")
print(f"  v0 skipped: {v0.get('skipped_months', [])}")
print(f"  v7 skipped: {v7.get('skipped_months', [])}")

# Save JSON
out = {
    "months": months, "n": n,
    "v0_agg": v0_agg, "v7_agg": v7_agg,
    "v0_per_month": v0_pm, "v7_per_month": v7_pm,
    "wins_v7": wins,
}
p = Path("registry/comparisons/v0_vs_v0007_full_2020_2022.json")
p.parent.mkdir(parents=True, exist_ok=True)
p.write_text(json.dumps(out, indent=2))
print(f"\n  Saved: {p}")
