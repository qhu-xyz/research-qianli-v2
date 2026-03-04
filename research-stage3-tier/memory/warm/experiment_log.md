# Experiment Log

## v0 — Baseline (2026-03-04)
- **Config**: n_estimators=400, max_depth=5, lr=0.05, subsample=0.8, colsample=0.8, reg_alpha=1.0, reg_lambda=1.0, min_child_weight=25
- **Class weights**: {0:10, 1:5, 2:2, 3:1, 4:0.5}
- **Features**: 34 (11 flow prob + 7 distribution shape + 1 overload + 5 historical + 10 engineered)
- **Results** (12-month mean):
  - Tier-VC@100=0.075, Tier-VC@500=0.217, Tier-NDCG=0.767
  - QWK=0.359, Macro-F1=0.369
  - Tier-Recall@0=0.374, Tier-Recall@1=0.098
- **Key finding**: Tier 4 has 0 samples in all months. Tier-Recall@1 catastrophically low.
