#!/usr/bin/env bash
set -euo pipefail

# Compare v0 vs v0007 across all 36 months of 2020-2022
# Runs both configs as independent background processes for parallelism

cd /home/xyz/workspace/research-qianli-v2/research-stage2-shadow
source /home/xyz/workspace/pmodel/.venv/bin/activate
export PYTHONPATH=.

ALL_MONTHS="2020-01 2020-02 2020-03 2020-04 2020-05 2020-06 2020-07 2020-08 2020-09 2020-10 2020-11 2020-12 2021-01 2021-02 2021-03 2021-04 2021-05 2021-06 2021-07 2021-08 2021-09 2021-10 2021-11 2021-12 2022-01 2022-02 2022-03 2022-04 2022-05 2022-06 2022-07 2022-08 2022-09 2022-10 2022-11 2022-12"

echo "=== Starting parallel comparison: v0 vs v0007 ==="
echo "    36 months (2020-01 through 2022-12)"
echo "    Time: $(date)"
echo ""

mkdir -p .logs

# v0 config: mcw=10, reg_alpha=0.1 (v0007 defaults are reg_lambda=1.0, mcw=25, reg_alpha=1.0)
echo "[v0]    Launching benchmark (mcw=10, reg_alpha=0.1)..."
python ml/benchmark.py \
    --version-id v0-reeval \
    --ptype f0 --class-type onpeak \
    --eval-months $ALL_MONTHS \
    --overrides '{"regressor": {"min_child_weight": 10, "reg_alpha": 0.1}}' \
    > .logs/v0_reeval.log 2>&1 &
V0_PID=$!
echo "[v0]    PID: $V0_PID"

# v0007 config: current defaults (mcw=25, reg_alpha=1.0, reg_lambda=1.0)
echo "[v0007] Launching benchmark (mcw=25, reg_alpha=1.0)..."
python ml/benchmark.py \
    --version-id v0007-reeval \
    --ptype f0 --class-type onpeak \
    --eval-months $ALL_MONTHS \
    > .logs/v7_reeval.log 2>&1 &
V7_PID=$!
echo "[v0007] PID: $V7_PID"

echo ""
echo "Waiting for both to complete..."
echo "  v0 log:    .logs/v0_reeval.log"
echo "  v0007 log: .logs/v7_reeval.log"

# Wait for both
V0_RC=0
V7_RC=0
wait $V0_PID || V0_RC=$?
echo "[v0]    Done (exit code: $V0_RC) at $(date)"
wait $V7_PID || V7_RC=$?
echo "[v0007] Done (exit code: $V7_RC) at $(date)"

if (( V0_RC != 0 )); then
    echo "ERROR: v0 benchmark failed (exit code $V0_RC)"
    tail -20 .logs/v0_reeval.log
fi
if (( V7_RC != 0 )); then
    echo "ERROR: v0007 benchmark failed (exit code $V7_RC)"
    tail -20 .logs/v7_reeval.log
fi

# Both done — produce comparison
if (( V0_RC == 0 && V7_RC == 0 )); then
    echo ""
    echo "=== Both benchmarks complete. Running comparison ==="
    python scripts/produce_comparison.py
else
    echo "ERROR: One or both benchmarks failed. Check logs."
    exit 1
fi
