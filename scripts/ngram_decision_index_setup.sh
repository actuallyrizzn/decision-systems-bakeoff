#!/usr/bin/env bash
# Clone the official Decision Index kit onto ngram. Do not train Flybrain on it.
set -euo pipefail

ROOT="${HOME}/julia-bakeoff/decision-index"
KIT="${ROOT}/kit"
LOG="${HOME}/julia-bakeoff/decision-index-setup.log"
mkdir -p "$ROOT"
exec >>"$LOG" 2>&1
echo "==== decision-index setup $(date -u +%Y-%m-%dT%H:%M:%SZ) ===="

if [[ ! -d "$KIT/.git" ]]; then
  git clone --depth 1 https://github.com/apolinario/decision-index.git "$KIT"
else
  git -C "$KIT" pull --ff-only || true
fi

echo "tree:"
ls "$KIT" | head -40
if [[ -f "$KIT/README.md" ]]; then
  echo "----- README head -----"
  head -n 80 "$KIT/README.md"
fi
echo "==== setup listed $(date -u +%Y-%m-%dT%H:%M:%SZ) ===="
