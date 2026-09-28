#!/usr/bin/env bash
# Successive fair-turf bakeoff: wait for Julia → Jev → Laya → Flybrain.
# All unattended. Updates ~/julia-bakeoff/fair-STATUS.json after each arm.
set -euo pipefail

ROOT="${HOME}/julia-bakeoff"
REPO="${ROOT}/decision-systems-bakeoff"
DATA="${ROOT}/fair-data"
OUT="${ROOT}/fair-runs"
STATUS="${ROOT}/fair-STATUS.json"
MASTER_LOG="${ROOT}/fair-successive.log"
JULIA_PID_FILE="${ROOT}/fair-chain.pid"
JULIA_VENV="${ROOT}/venv"
LAYA_PY="${HOME}/laya/venv/bin/python"
GLOVE_DIR="${ROOT}/vectors"
GLOVE="${GLOVE_DIR}/glove.6B.100d.txt"
VENICE_PASS="${HOME}/.ssh/venice-jev-ab.pass"

exec >>"$MASTER_LOG" 2>&1
echo "==== successive queue start $(date -u +%Y-%m-%dT%H:%M:%SZ) ===="

mkdir -p "$OUT" "$GLOVE_DIR"

wait_julia() {
  echo "waiting for Julia fair chain…"
  # Prefer pid file; also accept STATUS phase=done
  if [[ -f "$JULIA_PID_FILE" ]]; then
    local pid
    pid="$(cat "$JULIA_PID_FILE")"
    while kill -0 "$pid" 2>/dev/null; do
      sleep 30
    done
  fi
  # Poll STATUS until phase done or summaries exist for all julia tasks
  local i=0
  while true; do
    if [[ -f "$STATUS" ]] && python3 -c "
import json
st=json.load(open('$STATUS'))
print(st.get('phase',''))
" 2>/dev/null | grep -qx done; then
      echo "Julia STATUS phase=done"
      break
    fi
    # If process gone and we have typed summary, treat as done
    if [[ ! -f "$JULIA_PID_FILE" ]] || ! kill -0 "$(cat "$JULIA_PID_FILE" 2>/dev/null)" 2>/dev/null; then
      if [[ -f "$OUT/julia_typed_decisions_summary.json" ]] || [[ -f "$OUT/julia_massive_scenario_en_summary.json" ]]; then
        echo "Julia process gone; summaries present — continuing"
        break
      fi
    fi
    i=$((i+1))
    if (( i > 720 )); then
      echo "TIMEOUT waiting for Julia (>6h)" >&2
      exit 1
    fi
    sleep 30
  done
}

merge_status() {
  local arm="$1"
  local phase="$2"
  local current="${3:-}"
  python3 - <<PY
import json, glob
from datetime import datetime, timezone
from pathlib import Path
out = Path("$OUT")
status_path = Path("$STATUS")
completed = []
for p in sorted(out.glob("*_summary.json")):
    try:
        completed.append(json.loads(p.read_text()))
    except Exception:
        pass
payload = {
  "run_id": "fair-successive",
  "suite": "fair",
  "fair_bakeoff_id": "julia-fair-turf-2026-09",
  "arm": "$arm",
  "phase": "$phase",
  "current_task": "$current" or None,
  "tasks_planned_all_arms": ["julia","jev","laya","flybrain"],
  "completed": completed,
  "updated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
  "observable": {
    "status_json": str(status_path),
    "out_dir": str(out),
    "how_to_check": "ask Otto for a status check",
  },
}
status_path.write_text(json.dumps(payload, indent=2) + "\n")
print(f"STATUS arm=$arm phase=$phase completed={len(completed)}")
PY
}

run_julia_venv_chain() {
  local arm="$1"
  # shellcheck disable=SC1091
  source "${JULIA_VENV}/bin/activate"
  export PYTHONPATH="${REPO}/src:${PYTHONPATH:-}"
  merge_status "$arm" running starting
  python "${REPO}/scripts/run_fair_chain.py" \
    --data "$DATA" \
    --out "$OUT" \
    --status "$STATUS" \
    --arm "$arm" \
    --tasks agnews,emotion,massive_scenario_en,typed_decisions \
    --usd-stop 2.0
  merge_status "$arm" arm_done ""
}

run_laya_chain() {
  merge_status laya running starting
  export PYTHONPATH="${REPO}/src:${PYTHONPATH:-}"
  # Laya lives in its own venv; bakeoff code via PYTHONPATH
  "$LAYA_PY" "${REPO}/scripts/run_fair_chain.py" \
    --data "$DATA" \
    --out "$OUT" \
    --status "$STATUS" \
    --arm laya \
    --tasks agnews,emotion,massive_scenario_en,typed_decisions
  merge_status laya arm_done ""
}

ensure_glove() {
  if [[ -f "$GLOVE" ]] && [[ $(stat -c%s "$GLOVE") -gt 100000000 ]]; then
    echo "GloVe ready: $GLOVE"
    return 0
  fi
  echo "fetching GloVe…"
  # shellcheck disable=SC1091
  source "${JULIA_VENV}/bin/activate"
  python "${REPO}/scripts/fetch_glove.py" --dir "$GLOVE_DIR"
}

run_flybrain() {
  merge_status flybrain running starting
  # shellcheck disable=SC1091
  source "${JULIA_VENV}/bin/activate"
  export PYTHONPATH="${REPO}/src:${PYTHONPATH:-}"
  ensure_glove
  for task in agnews emotion massive_scenario_en; do
    merge_status flybrain running "$task"
    python "${REPO}/scripts/run_flybrain.py" \
      --data "$DATA" \
      --glove "$GLOVE" \
      --out "$OUT" \
      --task "$task"
  done
  merge_status flybrain done ""
}

# --- main ---
wait_julia
merge_status julia done ""

# Jev
if [[ ! -f "$VENICE_PASS" ]]; then
  echo "MISSING $VENICE_PASS — cannot run Jev" >&2
  merge_status jev error "missing venice pass"
  exit 2
fi
set -a
# shellcheck disable=SC1090
source "$VENICE_PASS"
set +a
export VENICE_API_KEY="${VENICE_JEV_AB_API_KEY:-${VENICE_API_KEY:-}}"
echo "starting Jev fair chain…"
run_julia_venv_chain jev

# Laya
if [[ ! -x "$LAYA_PY" ]]; then
  echo "MISSING Laya python $LAYA_PY" >&2
  merge_status laya error "missing laya venv"
  exit 2
fi
echo "starting Laya fair chain…"
run_laya_chain

# Flybrain
echo "starting Flybrain fair tasks…"
run_flybrain

merge_status all done ""
echo "==== successive queue COMPLETE $(date -u +%Y-%m-%dT%H:%M:%SZ) ===="
