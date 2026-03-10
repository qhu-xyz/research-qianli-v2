# Runbook — Safety Rules for Workers (Stage 2: Regression Pipeline)

## Sandbox Constraints
- Only modify files under `ml/`, `registry/${VERSION_ID}/`, and `handoff/`
- NEVER touch `registry/v0/` (baseline is immutable)
- NEVER modify `registry/gates.json` or `ml/evaluate.py` (HUMAN-WRITE-ONLY)
- NEVER modify other `registry/v*/` directories
- NEVER run `rm -rf` or delete registry directories
- ALWAYS commit changes before writing handoff JSON
- On 3x test failure: write failed handoff with error summary, do NOT commit
- Read VERSION_ID from state.json: `VERSION_ID=$(jq -r '.version_id' "${PROJECT_DIR}/state.json")`

## Stage 2 Constraint (CRITICAL)
- **NEVER modify ClassifierConfig in `ml/config.py`** — the classifier is frozen from stage 1
- Only modify `RegressorConfig` parameters (features, hyperparams, mode flags)
- Do NOT change classifier features, classifier hyperparams, or threshold_beta
- The classifier is a black box to stage 2 — treat it as immutable infrastructure

## Test Protocol
1. Run `python -m pytest ml/tests/ -v` after making changes
2. If tests fail, fix and retry (up to 3 attempts)
3. On 3rd failure: stop, write failed handoff with error details

## Commit Protocol
1. Stage only files you modified: `git add ml/ registry/${VERSION_ID}/`
2. Commit with message: `iter{N}: {brief description of changes}`
3. THEN write handoff JSON (never before commit)
