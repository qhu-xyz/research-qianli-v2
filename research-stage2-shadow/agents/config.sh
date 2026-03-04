#!/usr/bin/env bash
# All environment-specific settings. Source this at top of every script.
PROJECT_DIR="/home/xyz/workspace/research-qianli-v2/research-stage2-shadow"
RAY_ADDRESS="ray://10.8.0.36:10001"
DATA_ROOT="/opt/temp/tmp/pw_data/spice6"
VENV_ACTIVATE="/home/xyz/workspace/pmodel/.venv/bin/activate"
SMOKE_TEST="${SMOKE_TEST:-false}"
REGISTRY_DISK_LIMIT_MB=10240
CODEX_MODEL="gpt-5.3-codex"
STATE_FILE="${PROJECT_DIR}/state.json"

# Hard timeout limits (seconds) — OS-level kill, defense against agent infinite loops.
# These are the absolute max wall-clock time an agent process can run.
# poll_for_handoff timeouts are shorter; these are the last-resort kill switch.
# Smoke timings: orch=2.5m, worker=6.5m, reviewers=3m, synth=6m
# Real-data timings: worker benchmark ~35-50m (12 months × 400 trees via Ray)
TIMEOUT_ORCHESTRATOR=600     # 10 min (poll: 7 min)
TIMEOUT_WORKER=5400          # 90 min — 2-hypothesis screening (~12 min) + full benchmark (~35 min) + code/tests (~10 min)
TIMEOUT_REVIEWER_CLAUDE=600  # 10 min (poll: 7 min)
TIMEOUT_REVIEWER_CODEX=600   # 10 min (poll: 7 min)
TIMEOUT_SYNTHESIZER=1500     # 25 min (poll: 20 min) — synthesis observed at ~14 min; reads ~15 files + writes 8 files
