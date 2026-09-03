#!/usr/bin/env bash
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT="$ROOT/workdirs/socialnav_sub/pilot_v1/jdcloud_pilot"
mkdir -p "$OUT"

export GEMINI_API_KEY="${GEMINI_API_KEY:-${PROXY_OPENAI_API_KEY:-}}"
export GEMINI_API_BASE="${GEMINI_API_BASE:-https://modelservice.jdcloud.com/v1/responses}"
export GEMINI_MODEL="${GEMINI_MODEL:-Gemini 3-Pro-Preview}"

if [[ -z "$GEMINI_API_KEY" ]]; then
  printf 'Missing GEMINI_API_KEY or PROXY_OPENAI_API_KEY\n' > "$OUT/background.status"
  exit 2
fi

cd "$ROOT" || exit 1
if PYTHONPATH="$ROOT/spatial_social_code:$ROOT/MA-EgoQA" \
  python -u experiments/socialnav_sub/run_pilot_jdcloud.py \
  > "$OUT/background.log" 2>&1; then
  python experiments/socialnav_sub/export_cases.py
  printf 'complete\n' > "$OUT/background.status"
else
  status=$?
  printf 'failed exit=%s\n' "$status" > "$OUT/background.status"
  exit "$status"
fi
