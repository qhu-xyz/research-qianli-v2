#!/usr/bin/env bash
# Audit script for pipeline iterations.
#
# Usage:
#   bash agents/audit_iter.sh --batch-id <batch> --iter <N>          # detailed single-iteration audit
#   bash agents/audit_iter.sh --batch-id <batch> --iter <N> --brief  # compact one-liner per section
#   bash agents/audit_iter.sh --batch-id <batch> --all               # full batch audit (all iterations)
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
source "${SCRIPT_DIR}/config.sh"

BATCH_ID=""
ITER=""
ALL=false
BRIEF=false
PIPELINE_TIME=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --batch-id)       BATCH_ID="$2"; shift 2 ;;
    --batch-id=*)     BATCH_ID="${1#--batch-id=}"; shift ;;
    --iter)           ITER="$2"; shift 2 ;;
    --iter=*)         ITER="${1#--iter=}"; shift ;;
    --all)            ALL=true; shift ;;
    --brief)          BRIEF=true; shift ;;
    --pipeline-time)  PIPELINE_TIME="$2"; shift 2 ;;
    --pipeline-time=*) PIPELINE_TIME="${1#--pipeline-time=}"; shift ;;
    *) echo "Unknown arg: $1"; exit 1 ;;
  esac
done

[[ -n "$BATCH_ID" ]] || { echo "ERROR: --batch-id required"; exit 1; }

# -------------------------------------------------
# Single-iteration audit
# -------------------------------------------------
audit_iter() {
  local N="$1"
  local HANDOFF_DIR="${PROJECT_DIR}/handoff/${BATCH_ID}/iter${N}"
  local REPORT_DIR="${PROJECT_DIR}/reports/${BATCH_ID}/iter${N}"
  local COMP_JSON="${PROJECT_DIR}/registry/comparisons/${BATCH_ID}_iter${N}.json"

  if [[ "$BRIEF" == "true" ]]; then
    echo ""
    echo "--- Iteration ${N} audit ---"
  else
    echo ""
    echo "========================================"
    echo "  AUDIT: ${BATCH_ID} iteration ${N}"
    echo "========================================"
  fi

  # 1. Version created
  local VERSION_ID=""
  if [[ -f "${HANDOFF_DIR}/worker_done.json" ]]; then
    local WORKER_STATUS
    WORKER_STATUS=$(jq -r '.status // "unknown"' "${HANDOFF_DIR}/worker_done.json" 2>/dev/null)
    # Find version from state or handoff
    VERSION_ID=$(jq -r '.version_id // empty' "$STATE_FILE" 2>/dev/null)
    # Try to find from registry comparisons
    if [[ -z "$VERSION_ID" && -f "$COMP_JSON" ]]; then
      VERSION_ID=$(jq -r '.versions | keys | map(select(. != "v0")) | last // empty' "$COMP_JSON" 2>/dev/null)
    fi
    if [[ "$BRIEF" == "true" ]]; then
      echo "  Version: ${VERSION_ID:-?} | Worker: ${WORKER_STATUS}"
    else
      echo ""
      echo "1. VERSION: ${VERSION_ID:-unknown}"
      echo "   Worker status: ${WORKER_STATUS}"
      if [[ "$WORKER_STATUS" == "failed" ]]; then
        local ERR
        ERR=$(jq -r '.error // "no error message"' "${HANDOFF_DIR}/worker_done.json" 2>/dev/null)
        echo "   Error: ${ERR}"
      fi
    fi
  else
    if [[ "$BRIEF" == "true" ]]; then
      echo "  Worker: NO HANDOFF FOUND"
    else
      echo ""
      echo "1. VERSION: no worker handoff found"
    fi
  fi

  # 2. Gate results
  if [[ -f "$COMP_JSON" && -n "$VERSION_ID" ]]; then
    local GA GB
    GA=$(jq -r ".pass_summary[\"${VERSION_ID}\"].group_a_passed // \"?\"" "$COMP_JSON" 2>/dev/null)
    GB=$(jq -r ".pass_summary[\"${VERSION_ID}\"].group_b_passed // \"?\"" "$COMP_JSON" 2>/dev/null)

    if [[ "$BRIEF" == "true" ]]; then
      # One-line gate summary: pull key metrics
      local EVVC100 EVVC500 EVNDCG SPEARMAN
      EVVC100=$(jq -r ".versions[\"${VERSION_ID}\"][\"EV-VC@100\"].mean_value // \"?\"" "$COMP_JSON" 2>/dev/null)
      EVVC500=$(jq -r ".versions[\"${VERSION_ID}\"][\"EV-VC@500\"].mean_value // \"?\"" "$COMP_JSON" 2>/dev/null)
      EVNDCG=$(jq -r ".versions[\"${VERSION_ID}\"][\"EV-NDCG\"].mean_value // \"?\"" "$COMP_JSON" 2>/dev/null)
      SPEARMAN=$(jq -r ".versions[\"${VERSION_ID}\"][\"Spearman\"].mean_value // \"?\"" "$COMP_JSON" 2>/dev/null)
      echo "  Gates: A=${GA} B=${GB} | EV-VC@100=${EVVC100} EV-VC@500=${EVVC500} EV-NDCG=${EVNDCG} Spearman=${SPEARMAN}"
    else
      echo ""
      echo "2. GATES: Group A=${GA}  Group B=${GB}"
      echo ""
      # Detailed per-gate table
      echo "   | Gate | Mean | Floor | L1 | Tail F | L2 | Bot2 | L3 | Overall |"
      echo "   |------|------|-------|----|--------|----|------|----|---------|"
      for GATE in EV-VC@100 EV-VC@500 EV-NDCG Spearman C-RMSE C-MAE EV-VC@1000 R-REC@500; do
        local MV MP TF TP B2 TR OP
        MV=$(jq -r ".versions[\"${VERSION_ID}\"][\"${GATE}\"].mean_value // \"--\"" "$COMP_JSON" 2>/dev/null)
        MP=$(jq -r ".versions[\"${VERSION_ID}\"][\"${GATE}\"].mean_passed // \"?\"" "$COMP_JSON" 2>/dev/null)
        TF=$(jq -r ".versions[\"${VERSION_ID}\"][\"${GATE}\"].tail_failures // \"?\"" "$COMP_JSON" 2>/dev/null)
        TP=$(jq -r ".versions[\"${VERSION_ID}\"][\"${GATE}\"].tail_passed // \"?\"" "$COMP_JSON" 2>/dev/null)
        B2=$(jq -r ".versions[\"${VERSION_ID}\"][\"${GATE}\"].bottom_2_mean // \"--\"" "$COMP_JSON" 2>/dev/null)
        TR=$(jq -r ".versions[\"${VERSION_ID}\"][\"${GATE}\"].tail_regression_passed // \"?\"" "$COMP_JSON" 2>/dev/null)
        OP=$(jq -r ".versions[\"${VERSION_ID}\"][\"${GATE}\"].overall_passed // \"?\"" "$COMP_JSON" 2>/dev/null)
        # Convert true/false to P/F
        MP=$([[ "$MP" == "true" ]] && echo "P" || ([[ "$MP" == "false" ]] && echo "F" || echo "?"))
        TP=$([[ "$TP" == "true" ]] && echo "P" || ([[ "$TP" == "false" ]] && echo "F" || echo "?"))
        TR=$([[ "$TR" == "true" ]] && echo "P" || ([[ "$TR" == "false" ]] && echo "F" || echo "?"))
        OP=$([[ "$OP" == "true" ]] && echo "P" || ([[ "$OP" == "false" ]] && echo "F" || echo "?"))
        printf "   | %-12s | %6s | %6s | %s | %6s | %s | %6s | %s | %7s |\n" \
          "$GATE" "$MV" "" "$MP" "$TF" "$TP" "$B2" "$TR" "$OP"
      done
    fi
  elif [[ "$BRIEF" != "true" ]]; then
    echo ""
    echo "2. GATES: no comparison JSON found"
  fi

  # 3. Promotion decision
  local SYNTH_HANDOFF="${HANDOFF_DIR}/orchestrator_synth_done.json"
  if [[ -f "$SYNTH_HANDOFF" ]]; then
    local PROMOTE
    PROMOTE=$(jq -r '.decisions.promote_version // "null"' "$SYNTH_HANDOFF" 2>/dev/null)
    if [[ "$BRIEF" == "true" ]]; then
      echo "  Promoted: ${PROMOTE}"
    else
      echo ""
      echo "3. PROMOTION: ${PROMOTE}"
      local NEXT_HYP
      NEXT_HYP=$(jq -r '.decisions.next_hypothesis // "none"' "$SYNTH_HANDOFF" 2>/dev/null)
      echo "   Next hypothesis: ${NEXT_HYP}"
    fi
  fi

  # 4. Reviews
  local CLAUDE_REV="${PROJECT_DIR}/reviews/${BATCH_ID}_iter${N}_claude.md"
  local CODEX_REV="${PROJECT_DIR}/reviews/${BATCH_ID}_iter${N}_codex.md"
  if [[ "$BRIEF" == "true" ]]; then
    local C_SZ="missing" X_SZ="missing"
    [[ -f "$CLAUDE_REV" ]] && C_SZ="$(wc -c < "$CLAUDE_REV")b"
    [[ -f "$CODEX_REV" ]] && X_SZ="$(wc -c < "$CODEX_REV")b"
    echo "  Reviews: claude=${C_SZ} codex=${X_SZ}"
  else
    echo ""
    echo "4. REVIEWS:"
    if [[ -f "$CLAUDE_REV" ]]; then
      echo "   Claude: $(wc -l < "$CLAUDE_REV") lines, $(wc -c < "$CLAUDE_REV") bytes"
      echo "   First line: $(head -1 "$CLAUDE_REV")"
    else
      echo "   Claude: MISSING"
    fi
    if [[ -f "$CODEX_REV" ]]; then
      echo "   Codex: $(wc -l < "$CODEX_REV") lines, $(wc -c < "$CODEX_REV") bytes"
      echo "   First line: $(head -1 "$CODEX_REV")"
    else
      echo "   Codex: MISSING (may have timed out -- non-fatal)"
    fi
  fi

  # 5. Agent exit codes (from log files)
  if [[ "$BRIEF" != "true" ]]; then
    echo ""
    echo "5. AGENT LOGS:"
    for AGENT in "orch-${BATCH_ID}-iter${N}" "worker-${BATCH_ID}-iter${N}" "rev-claude-${BATCH_ID}-iter${N}" "rev-codex-${BATCH_ID}-iter${N}" "synth-${BATCH_ID}-iter${N}"; do
      local LOG="${PROJECT_DIR}/.logs/sessions/${AGENT}.log"
      if [[ -f "$LOG" ]]; then
        local SZ EXIT_LINE
        SZ=$(wc -c < "$LOG")
        EXIT_LINE=$(grep -o 'EXIT_CODE=[0-9]*' "$LOG" 2>/dev/null | tail -1 || echo "EXIT_CODE=?")
        echo "   ${AGENT}: ${SZ} bytes, ${EXIT_LINE}"
      else
        echo "   ${AGENT}: no log"
      fi
    done
  fi

  # 6. Errors in handoffs
  if [[ "$BRIEF" != "true" ]]; then
    echo ""
    echo "6. HANDOFF STATUS:"
    for HF in "${HANDOFF_DIR}"/*.json; do
      [[ -f "$HF" ]] || continue
      local FNAME STATUS
      FNAME=$(basename "$HF")
      STATUS=$(jq -r '.status // "unknown"' "$HF" 2>/dev/null)
      if [[ "$STATUS" == "failed" ]]; then
        local ERR
        ERR=$(jq -r '.error // "no detail"' "$HF" 2>/dev/null)
        echo "   ${FNAME}: FAILED -- ${ERR}"
      else
        echo "   ${FNAME}: ${STATUS}"
      fi
    done
  fi

  # 7. Changes summary
  if [[ "$BRIEF" != "true" && -n "$VERSION_ID" ]]; then
    local CHANGES="${PROJECT_DIR}/registry/${VERSION_ID}/changes_summary.md"
    if [[ -f "$CHANGES" ]]; then
      echo ""
      echo "7. CHANGES (first 10 lines):"
      head -10 "$CHANGES" | sed 's/^/   /'
    fi
  fi
}

# -------------------------------------------------
# Full batch audit (runs after all iterations)
# -------------------------------------------------
audit_batch() {
  echo ""
  echo "========================================================"
  echo "         BATCH AUDIT: ${BATCH_ID}"
  echo "========================================================"

  # Pipeline timing
  if [[ -n "$PIPELINE_TIME" ]]; then
    local MINS=$(( PIPELINE_TIME / 60 ))
    local SECS=$(( PIPELINE_TIME % 60 ))
    echo ""
    echo "Pipeline wall time: ${MINS}m${SECS}s"
  fi

  # State machine
  echo ""
  echo "State: $(jq -r '.state' "$STATE_FILE")"

  # Champion status
  local CHAMP
  CHAMP=$(jq -r '.version // "null"' "${PROJECT_DIR}/registry/champion.json" 2>/dev/null)
  echo "Champion: ${CHAMP}"

  # Per-iteration summary
  echo ""
  echo "=== Per-Iteration Summary ==="
  local MAX_N
  MAX_N=$(ls -d "${PROJECT_DIR}/handoff/${BATCH_ID}"/iter* 2>/dev/null | wc -l)
  for I in $(seq 1 "$MAX_N"); do
    BRIEF=true audit_iter "$I"
  done

  # Metric progression across iterations
  echo ""
  echo "=== Metric Progression ==="
  echo ""
  printf "  %-8s | %-8s | %-10s | %-10s | %-10s | %-10s\n" "Iter" "Version" "EV-VC@100" "EV-VC@500" "EV-NDCG" "Spearman"
  printf "  %-8s-+-%-8s-+-%-10s-+-%-10s-+-%-10s-+-%-10s\n" "--------" "--------" "----------" "----------" "----------" "----------"

  # v0 baseline row
  local V0_JSON="${PROJECT_DIR}/registry/comparisons/${BATCH_ID}_iter1.json"
  if [[ -f "$V0_JSON" ]]; then
    local V0_EVVC100 V0_EVVC500 V0_EVNDCG V0_SPEARMAN
    V0_EVVC100=$(jq -r '.versions["v0"]["EV-VC@100"].mean_value // "?"' "$V0_JSON" 2>/dev/null)
    V0_EVVC500=$(jq -r '.versions["v0"]["EV-VC@500"].mean_value // "?"' "$V0_JSON" 2>/dev/null)
    V0_EVNDCG=$(jq -r '.versions["v0"]["EV-NDCG"].mean_value // "?"' "$V0_JSON" 2>/dev/null)
    V0_SPEARMAN=$(jq -r '.versions["v0"]["Spearman"].mean_value // "?"' "$V0_JSON" 2>/dev/null)
    printf "  %-8s | %-8s | %-10s | %-10s | %-10s | %-10s\n" "base" "v0" "$V0_EVVC100" "$V0_EVVC500" "$V0_EVNDCG" "$V0_SPEARMAN"
  fi

  for I in $(seq 1 "$MAX_N"); do
    local CJ="${PROJECT_DIR}/registry/comparisons/${BATCH_ID}_iter${I}.json"
    if [[ -f "$CJ" ]]; then
      # Find the non-v0 version for this iteration
      local VID
      VID=$(jq -r ".versions | keys | map(select(startswith(\"v0\") | not)) | last // \"?\"" "$CJ" 2>/dev/null)
      if [[ "$VID" != "?" && "$VID" != "null" ]]; then
        local I_EVVC100 I_EVVC500 I_EVNDCG I_SPEARMAN
        I_EVVC100=$(jq -r ".versions[\"${VID}\"][\"EV-VC@100\"].mean_value // \"?\"" "$CJ" 2>/dev/null)
        I_EVVC500=$(jq -r ".versions[\"${VID}\"][\"EV-VC@500\"].mean_value // \"?\"" "$CJ" 2>/dev/null)
        I_EVNDCG=$(jq -r ".versions[\"${VID}\"][\"EV-NDCG\"].mean_value // \"?\"" "$CJ" 2>/dev/null)
        I_SPEARMAN=$(jq -r ".versions[\"${VID}\"][\"Spearman\"].mean_value // \"?\"" "$CJ" 2>/dev/null)
        printf "  %-8s | %-8s | %-10s | %-10s | %-10s | %-10s\n" "iter${I}" "$VID" "$I_EVVC100" "$I_EVVC500" "$I_EVNDCG" "$I_SPEARMAN"
      fi
    fi
  done

  # Memory health
  echo ""
  echo "=== Memory Health ==="
  for F in memory/hot/progress.md memory/hot/champion.md memory/hot/learning.md memory/hot/critique_summary.md memory/warm/experiment_log.md memory/warm/decision_log.md memory/warm/hypothesis_log.md; do
    if [[ -f "${PROJECT_DIR}/$F" ]]; then
      local LINES BYTES
      LINES=$(wc -l < "${PROJECT_DIR}/$F")
      BYTES=$(wc -c < "${PROJECT_DIR}/$F")
      printf "  %-45s %4d lines  %6d bytes\n" "$F" "$LINES" "$BYTES"
    else
      printf "  %-45s MISSING\n" "$F"
    fi
  done

  # Registry versions created
  echo ""
  echo "=== Registry ==="
  for VDIR in "${PROJECT_DIR}"/registry/v*/; do
    [[ -d "$VDIR" ]] || continue
    local VN
    VN=$(basename "$VDIR")
    local HAS_METRICS="no" HAS_CONFIG="no" HAS_MODEL="no" HAS_CHANGES="no"
    [[ -f "${VDIR}/metrics.json" ]] && HAS_METRICS="yes"
    [[ -f "${VDIR}/config.json" ]] && HAS_CONFIG="yes"
    [[ -d "${VDIR}/model" ]] && HAS_MODEL="yes"
    [[ -f "${VDIR}/changes_summary.md" ]] && HAS_CHANGES="yes"
    echo "  ${VN}: metrics=${HAS_METRICS} config=${HAS_CONFIG} model=${HAS_MODEL} changes=${HAS_CHANGES}"
  done

  # Git commits during batch
  echo ""
  echo "=== Git Commits (this batch) ==="
  git -C "$PROJECT_DIR" log --oneline --since="2 hours ago" | head -10

  echo ""
  echo "========================================================"
  echo "  Files to inspect:"
  echo "  reports/${BATCH_ID}/iter*/comparison.md"
  echo "  reviews/${BATCH_ID}_iter*_claude.md"
  echo "  reviews/${BATCH_ID}_iter*_codex.md"
  echo "  memory/hot/critique_summary.md"
  echo "  memory/warm/experiment_log.md"
  echo "========================================================"
}

# -------------------------------------------------
# Main dispatch
# -------------------------------------------------
if [[ "$ALL" == "true" ]]; then
  audit_batch
elif [[ -n "$ITER" ]]; then
  audit_iter "$ITER"
else
  echo "ERROR: specify --iter <N> or --all"
  exit 1
fi
