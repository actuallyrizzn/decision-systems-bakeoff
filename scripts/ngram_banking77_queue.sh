#!/usr/bin/env bash
# Banking77 locked addition — Flybrain ridge, then Laya, then Jev.
# Julia skipped: 77 options > native max 20 (same rule as clinc150).
set -euo pipefail

ROOT="${HOME}/julia-bakeoff"
REPO="${ROOT}/decision-systems-bakeoff"
DATA="${ROOT}/fair-data"
OUT="${ROOT}/banking77-runs"
STATUS="${ROOT}/banking77-STATUS.json"
MASTER_LOG="${ROOT}/banking77-queue.log"
LAYA_PY="${HOME}/laya/venv/bin/python"
GLOVE="${HOME}/moejev/vectors/glove.6B.100d.txt"
VENICE_PASS="${HOME}/.ssh/venice-jev-ab.pass"

mkdir -p "$OUT"
exec >>"$MASTER_LOG" 2>&1
echo "==== banking77 queue start $(date -u +%Y-%m-%dT%H:%M:%SZ) ===="

write_status() {
  local phase="$1"
  local current="${2:-}"
  python3 - <<PY
import json, glob
from datetime import datetime, timezone
from pathlib import Path
out = Path("$OUT")
completed = []
for p in sorted(out.glob("*_summary.json")):
    try:
        completed.append(json.loads(p.read_text()))
    except Exception:
        pass
payload = {
  "run_id": "banking77-locked-additions-2026-10",
  "suite": "locked_additions",
  "task": "banking77",
  "phase": "$phase",
  "current_task": "$current" or None,
  "updated_at": datetime.now(timezone.utc).strftime("%Y-%m-%dT%H:%M:%SZ"),
  "completed": completed,
  "julia": "skipped_native_max_20",
  "lock_doc": "https://tasks.decisionsciencecorp.com/admin/doc.php?id=1466",
}
Path("$STATUS").write_text(json.dumps(payload, indent=2) + "\n")
print("status", payload["phase"], "summaries", len(completed))
PY
}

cd "$REPO"

"$LAYA_PY" scripts/verify_data.py --data "$DATA" --task banking77

write_status running flybrain
echo "==== flybrain banking77 $(date -u +%Y-%m-%dT%H:%M:%SZ) ===="
"$LAYA_PY" scripts/run_flybrain.py --data "$DATA" --glove "$GLOVE" --out "$OUT" --task banking77
write_status running laya

echo "==== laya banking77 $(date -u +%Y-%m-%dT%H:%M:%SZ) ===="
USE_TF=0 "$LAYA_PY" scripts/run_arm.py --data "$DATA" --out "$OUT" --task banking77 --arm laya
write_status running jev

echo "==== jev banking77 $(date -u +%Y-%m-%dT%H:%M:%SZ) ===="
set -a
# shellcheck disable=SC1090
. "$VENICE_PASS"
set +a
"$LAYA_PY" scripts/run_arm.py --data "$DATA" --out "$OUT" --task banking77 --arm jev --usd-stop 5

write_status done
echo "==== banking77 queue done $(date -u +%Y-%m-%dT%H:%M:%SZ) ===="
