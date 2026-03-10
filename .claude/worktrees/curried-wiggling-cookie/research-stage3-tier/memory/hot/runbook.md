# Runbook

## Environment
- Project: `/home/xyz/workspace/research-qianli-v2/research-stage3-tier`
- Venv: `/home/xyz/workspace/pmodel/.venv/bin/activate`
- Ray: `ray://10.8.0.36:10001`
- PYTHONPATH must include project root for `ml` imports

## Running Benchmark
```bash
cd /home/xyz/workspace/research-qianli-v2/research-stage3-tier
source /home/xyz/workspace/pmodel/.venv/bin/activate
PYTHONPATH=. python ml/benchmark.py --version-id <vid> 2>&1 | tee registry/<vid>/benchmark.log
```

## Running Tests
```bash
cd /home/xyz/workspace/research-qianli-v2/research-stage3-tier
source /home/xyz/workspace/pmodel/.venv/bin/activate
PYTHONPATH=. python -m pytest ml/tests/ -v
```

## Key Files
- `ml/config.py` — TierConfig (hyperparams, features, class weights, bins)
- `ml/train.py` — XGBClassifier training + prediction
- `ml/features.py` — compute_tier_labels, prepare_features, compute_sample_weights
- `ml/evaluate.py` — evaluate_tier_pipeline, aggregate_months
- `ml/pipeline.py` — 6-phase pipeline (load → features → train → predict → evaluate → save)
- `ml/benchmark.py` — 12-month rolling benchmark
- `registry/gates.json` — gate definitions with floors
- `ml/compare.py` — version comparison + promotion logic

## Known Issues
- Tier 4 has 0 samples in all real months (no negative shadow prices)
- predict_tier_probabilities pads missing classes — already fixed
- XGBoost drops unseen classes from predict_proba output
