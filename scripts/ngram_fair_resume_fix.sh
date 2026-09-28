#!/usr/bin/env bash
# Resume fair bakeoff after typed/SSL breakage.
# Order: fix-verify typed smoke → julia typed → jev resume class+typed → laya full → flybrain
set -uo pipefail
# intentionally NOT set -e: arms continue even if one fails

ROOT="${HOME}/julia-bakeoff"
REPO="${ROOT}/decision-systems-bakeoff"
DATA="${ROOT}/fair-data"
OUT="${ROOT}/fair-runs"
STATUS="${ROOT}/fair-STATUS.json"
LOG="${ROOT}/fair-resume.log"
JULIA_VENV="${ROOT}/venv"
LAYA_PY="${HOME}/laya/venv/bin/python"
GLOVE_DIR="${ROOT}/vectors"
GLOVE="${GLOVE_DIR}/glove.6B.100d.txt"
VENICE_PASS="${HOME}/.ssh/venice-jev-ab.pass"
MODEL="${ROOT}/Julia-1"
PY_JULIA="${JULIA_VENV}/bin/python"

exec >>"$LOG" 2>&1
echo "==== fair resume $(date -u +%Y-%m-%dT%H:%M:%SZ) ===="

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
  "run_id": "fair-resume-fix",
  "suite": "fair",
  "fair_bakeoff_id": "julia-fair-turf-2026-09",
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

# --- 0) smoke typed (1 case) on Julia ---
merge_status fix running typed_smoke
"$PY_JULIA" "${REPO}/scripts/run_typed_decisions.py" \
  --data "$DATA" --out "$OUT/smoke" --arm julia --model-dir "$MODEL" \
  --limit-cases 1 --status "$STATUS"
SMOKE_RC=$?
if [[ $SMOKE_RC -ne 0 ]]; then
  echo "TYPED SMOKE FAILED rc=$SMOKE_RC" >&2
  merge_status fix error typed_smoke
  exit 2
fi
echo "typed smoke OK"
rm -rf "$OUT/smoke"

# --- 1) Julia typed full ---
merge_status julia running typed_decisions
"$PY_JULIA" "${REPO}/scripts/run_typed_decisions.py" \
  --data "$DATA" --out "$OUT" --arm julia --model-dir "$MODEL" --status "$STATUS"
merge_status julia arm_done ""

# --- 2) Jev: resume incomplete classification + typed ---
if [[ ! -f "$VENICE_PASS" ]]; then
  echo "MISSING venice pass" >&2
  merge_status jev error missing_pass
else
  set -a
  # shellcheck disable=SC1090
  source "$VENICE_PASS"
  set +a
  export VENICE_API_KEY="${VENICE_JEV_AB_API_KEY:-}"
  # resume starts from prior n_scored
  for task_start in "agnews:94" "emotion:315" "massive_scenario_en:2466"; do
    task="${task_start%%:*}"
    start="${task_start##*:}"
    # if already complete, skip
    if "$PY_JULIA" - "$OUT" "$task" <<'PY'
import json, sys
from pathlib import Path
out, task = Path(sys.argv[1]), sys.argv[2]
p = out / f"jev_{task}_summary.json"
if not p.exists():
    raise SystemExit(1)
s = json.loads(p.read_text())
ok = s.get("n_scored") == s.get("n_planned") and not s.get("stopped")
raise SystemExit(0 if ok else 1)
PY
    then
      echo "jev $task already complete — skip"
      continue
    fi
    merge_status jev running "$task"
    "$PY_JULIA" "${REPO}/scripts/run_fair_chain.py" \
      --data "$DATA" --out "$OUT" --status "$STATUS" --arm jev \
      --tasks "$task" --start-at "$start" --skip-typed --continue-on-error --usd-stop 2.0
  done
  merge_status jev running typed_decisions
  "$PY_JULIA" "${REPO}/scripts/run_typed_decisions.py" \
    --data "$DATA" --out "$OUT" --arm jev --status "$STATUS" --usd-stop 2.0
  merge_status jev arm_done ""
fi

# --- 3) Laya full ---
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
merge_status all done ""
echo "==== fair resume COMPLETE $(date -u +%Y-%m-%dT%H:%M:%SZ) ===="
