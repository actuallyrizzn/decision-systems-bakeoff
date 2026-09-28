#!/usr/bin/env bash
# Setup + launch Julia-1 bakeoff chain on ngram (unattended).
set -euo pipefail
ROOT="${HOME}/julia-bakeoff"
REPO_SRC="${ROOT}/decision-systems-bakeoff"
MODEL="${ROOT}/Julia-1"
VENV="${ROOT}/venv"
DATA="${HOME}/jev-ab/data"
OUT="${ROOT}/runs"
STATUS="${ROOT}/STATUS.json"
LOG="${ROOT}/chain.log"

mkdir -p "$ROOT" "$OUT"
cd "$ROOT"

if [[ ! -f "$REPO_SRC/scripts/run_julia_chain.py" ]]; then
  echo "REPO missing at $REPO_SRC — sync from Otto first" >&2
  exit 2
fi

python3 -m venv "$VENV"
# shellcheck disable=SC1091
source "$VENV/bin/activate"
pip install -q -U pip wheel
pip install -q torch --index-url https://download.pytorch.org/whl/cpu
pip install -q huggingface_hub

if [[ ! -f "$MODEL/julia_config.json" && ! -f "$MODEL/config.json" ]]; then
  python -c "from huggingface_hub import snapshot_download; snapshot_download('SupersonicLabs/Julia-1', local_dir=r'$MODEL')"
fi
pip install -q -e "$MODEL"
pip install -q -e "$REPO_SRC"

python - <<PY
import hashlib, json
from pathlib import Path
repo = Path("$REPO_SRC")
data = Path("$DATA")
lock = json.loads((repo / "lockfile.json").read_text())
for task in ("sst2", "clinc10", "bugsev"):
    p = data / task / "test.tsv"
    h = hashlib.sha256(p.read_bytes()).hexdigest()
    want = lock["tasks"][task]["test_sha256"]
    assert h == want, (task, h, want)
    print("OK", task, h[:12])
PY

cat > "$STATUS" <<EOF
{
  "phase": "launching",
  "updated_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "tasks_planned": ["sst2", "clinc10", "bugsev"],
  "tasks_excluded": ["clinc150"],
  "note": "chain starting"
}
EOF

nohup python "$REPO_SRC/scripts/run_julia_chain.py" \
  --data "$DATA" \
  --out "$OUT" \
  --model-dir "$MODEL" \
  --status "$STATUS" \
  --tasks sst2,clinc10,bugsev \
  >>"$LOG" 2>&1 &
echo $! > "$ROOT/chain.pid"
echo "launched pid=$(cat "$ROOT/chain.pid") log=$LOG status=$STATUS"
