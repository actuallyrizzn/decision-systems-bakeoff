#!/usr/bin/env bash
# Full-fidelity fair-turf redo: archive thin-telemetry runs, re-run Julia→Jev→Laya→Flybrain.
# Does NOT use set -e: arms continue after a failure (same as resume_fix).
set -uo pipefail

ROOT="${HOME}/julia-bakeoff"
REPO="${ROOT}/decision-systems-bakeoff"
DATA="${ROOT}/fair-data"
OUT="${ROOT}/fair-runs"
STATUS="${ROOT}/fair-STATUS.json"
LOG="${ROOT}/fair-full-fidelity-redo.log"
JULIA_VENV="${ROOT}/venv"
LAYA_PY="${HOME}/laya/venv/bin/python"
GLOVE_DIR="${ROOT}/vectors"
GLOVE="${GLOVE_DIR}/glove.6B.100d.txt"
VENICE_PASS="${HOME}/.ssh/venice-jev-ab.pass"
MODEL="${ROOT}/Julia-1"
PY_JULIA="${JULIA_VENV}/bin/python"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"

exec >>"$LOG" 2>&1
echo "==== fair full-fidelity redo $(date -u +%Y-%m-%dT%H:%M:%SZ) ===="

# Kill any in-flight thin-fidelity chain / resume
pkill -f 'ngram_fair_resume_fix.sh' 2>/dev/null || true
pkill -f 'ngram_fair_successive_queue.sh' 2>/dev/null || true
pkill -f 'run_fair_chain.py' 2>/dev/null || true
pkill -f 'run_typed_decisions.py' 2>/dev/null || true
sleep 2

# Archive prior fair-runs (keep evidence; do not reuse thin meters)
if [[ -d "$OUT" ]] && [[ -n "$(ls -A "$OUT" 2>/dev/null || true)" ]]; then
  ARCHIVE="${ROOT}/fair-runs-thin-${STAMP}"
  echo "archiving $OUT → $ARCHIVE"
  mv "$OUT" "$ARCHIVE"
fi
mkdir -p "$OUT"

merge_status() {
  local arm="$1" phase="$2" current="${3:-}"
  "$PY_JULIA" - <<PY
import json
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
  "run_id": "fair-full-fidelity-${STAMP}",
  "suite": "fair",
  "fair_bakeoff_id": "julia-fair-turf-2026-09",
  "telemetry": "full",
  "arm": "$arm",
  "phase": "$phase",
  "current_task": "$current" or None,
  "tasks_planned_all_arms": ["julia", "jev", "laya", "flybrain"],
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

ensure_glove() {
  if [[ -f "$GLOVE" ]] && [[ $(stat -c%s "$GLOVE") -gt 100000000 ]]; then
    echo "GloVe ready"
    return 0
  fi
  mkdir -p "$GLOVE_DIR"
  if [[ -f "$GLOVE_DIR/glove.6B.zip" ]]; then
    echo "extracting glove.6B.100d.txt from zip…"
    "$PY_JULIA" - <<'PY'
import zipfile
from pathlib import Path
z = Path.home() / "julia-bakeoff/vectors/glove.6B.zip"
out = Path.home() / "julia-bakeoff/vectors/glove.6B.100d.txt"
with zipfile.ZipFile(z) as zf:
    with zf.open("glove.6B.100d.txt") as src, out.open("wb") as dst:
        while True:
            chunk = src.read(1 << 20)
            if not chunk:
                break
            dst.write(chunk)
print("extracted", out, out.stat().st_size)
PY
  else
    "$PY_JULIA" "${REPO}/scripts/fetch_glove.py" --dir "$GLOVE_DIR"
  fi
}

export PYTHONPATH="${REPO}/src:${PYTHONPATH:-}"

# --- 1) Julia full (classification + typed) ---
merge_status julia running starting
"$PY_JULIA" "${REPO}/scripts/run_fair_chain.py" \
  --data "$DATA" --out "$OUT" --status "$STATUS" \
  --arm julia --model-dir "$MODEL" \
  --tasks agnews,emotion,massive_scenario_en,typed_decisions \
  --continue-on-error
merge_status julia arm_done ""

# --- 2) Jev ---
if [[ ! -f "$VENICE_PASS" ]]; then
  echo "MISSING venice pass" >&2
  merge_status jev error missing_pass
else
  set -a
  # shellcheck disable=SC1090
  source "$VENICE_PASS"
  set +a
  export VENICE_API_KEY="${VENICE_JEV_AB_API_KEY:-${VENICE_API_KEY:-}}"
  merge_status jev running starting
  "$PY_JULIA" "${REPO}/scripts/run_fair_chain.py" \
    --data "$DATA" --out "$OUT" --status "$STATUS" --arm jev \
    --tasks agnews,emotion,massive_scenario_en,typed_decisions \
    --usd-stop 2.0 --continue-on-error
  merge_status jev arm_done ""
fi

# --- 3) Laya ---
merge_status laya running starting
"$LAYA_PY" "${REPO}/scripts/run_fair_chain.py" \
  --data "$DATA" --out "$OUT" --status "$STATUS" --arm laya \
  --tasks agnews,emotion,massive_scenario_en,typed_decisions --continue-on-error
merge_status laya arm_done ""

# --- 4) Flybrain ---
merge_status flybrain running starting
ensure_glove
for task in agnews emotion massive_scenario_en; do
  merge_status flybrain running "$task"
  "$PY_JULIA" "${REPO}/scripts/run_flybrain.py" \
    --data "$DATA" --glove "$GLOVE" --out "$OUT" --task "$task"
done
merge_status flybrain running typed_decisions
"$PY_JULIA" "${REPO}/scripts/run_typed_decisions.py" \
  --data "$DATA" --out "$OUT" --status "$STATUS" --arm flybrain --glove "$GLOVE"
merge_status all done ""
echo "==== fair full-fidelity redo COMPLETE $(date -u +%Y-%m-%dT%H:%M:%SZ) ===="
