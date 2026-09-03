#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT="$ROOT/workdirs/socialnav_sub/pilot_v2/jdcloud_grounded_v5"
STATUS="$OUT/pilot.status"
LOG="$OUT/pilot.log"
mkdir -p "$OUT"

exec > >(tee -a "$LOG") 2>&1
trap 'code=$?; printf "exit_code=%s\n" "$code" >> "$STATUS"' EXIT

export GEMINI_API_KEY="${GEMINI_API_KEY:-${PROXY_OPENAI_API_KEY:-}}"
export GEMINI_API_BASE="${GEMINI_API_BASE:-https://modelservice.jdcloud.com/v1/responses}"
export GEMINI_MODEL="${GEMINI_MODEL:-Gemini 3-Pro-Preview}"

printf 'state=running\nstarted=%s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" > "$STATUS"
while true; do
  if PYTHONPATH="$ROOT:$ROOT/spatial_social_code:$ROOT/MA-EgoQA" \
    python -u "$ROOT/experiments/socialnav_sub/run_grounded_contactsheet_v5.py" \
    --pilot-root "$ROOT/workdirs/socialnav_sub/pilot_v2" \
    --max-scenes 3 \
    --request-delay-seconds 70; then
    break
  fi
  printf 'state=rate_limited_retrying\n' > "$STATUS"
  sleep 90
  printf 'state=running\n' > "$STATUS"
done
printf 'state=complete\ncompleted=%s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" > "$STATUS"
