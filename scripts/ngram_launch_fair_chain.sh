#!/usr/bin/env bash
# Sync fair data + bakeoff code to ngram, then launch Julia fair chain unattended.
set -euo pipefail
ROOT="${HOME}/julia-bakeoff"
REPO="${ROOT}/decision-systems-bakeoff"
DATA="${ROOT}/fair-data"
MODEL="${ROOT}/Julia-1"
OUT="${ROOT}/fair-runs"
STATUS="${ROOT}/fair-STATUS.json"
LOG="${ROOT}/fair-chain.log"
VENV="${ROOT}/venv"

mkdir -p "$ROOT" "$OUT" "$DATA"
cd "$ROOT"

if [[ ! -f "$REPO/scripts/run_fair_chain.py" ]]; then
  echo "REPO missing — sync bakeoff first" >&2
  exit 2
fi
if [[ ! -f "$DATA/agnews/test.tsv" ]]; then
  echo "FAIR DATA missing at $DATA" >&2
  exit 2
fi

# shellcheck disable=SC1091
source "$VENV/bin/activate"
pip install -q -e "$REPO" pyarrow 2>/dev/null || pip install -q pyarrow

python "$REPO/scripts/verify_data.py" --data "$DATA" --suite fair

cat > "$STATUS" <<EOF
{
  "phase": "launching",
  "suite": "fair",
  "arm": "julia",
  "updated_at": "$(date -u +%Y-%m-%dT%H:%M:%SZ)",
  "tasks_planned": ["agnews", "emotion", "massive_scenario_en", "typed_decisions"],
  "note": "fair chain starting"
}
EOF

nohup python "$REPO/scripts/run_fair_chain.py" \
  --data "$DATA" \
  --out "$OUT" \
  --status "$STATUS" \
  --arm julia \
  --model-dir "$MODEL" \
  --tasks agnews,emotion,massive_scenario_en,typed_decisions \
  >>"$LOG" 2>&1 &
echo $! > "$ROOT/fair-chain.pid"
echo "launched fair pid=$(cat "$ROOT/fair-chain.pid") log=$LOG status=$STATUS"
