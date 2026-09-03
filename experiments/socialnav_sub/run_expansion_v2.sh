#!/usr/bin/env bash
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT="$ROOT/workdirs/socialnav_sub/pilot_v2"
mkdir -p "$OUT/jdcloud_pilot"

python "$ROOT/experiments/socialnav_sub/prepare_pilot.py" \
  --output-root "$OUT" \
  --n-scenes 20 \
  > "$OUT/prepare.log" 2>&1
prepare_status=$?
if [[ $prepare_status -ne 0 ]]; then
  printf 'failed stage=prepare exit=%s\n' "$prepare_status" > "$OUT/jdcloud_pilot/background.status"
  exit "$prepare_status"
fi

export GEMINI_API_BASE="${GEMINI_API_BASE:-https://modelservice.jdcloud.com/v1/responses}"
export GEMINI_MODEL="${GEMINI_MODEL:-Gemini 3-Pro-Preview}"
if [[ -z "${GEMINI_API_KEY:-}" ]]; then
  printf 'failed stage=config missing=GEMINI_API_KEY\n' > "$OUT/jdcloud_pilot/background.status"
  exit 2
fi

printf 'running\n' > "$OUT/jdcloud_pilot/background.status"
if PYTHONPATH="$ROOT/spatial_social_code:$ROOT/MA-EgoQA" \
  python -u "$ROOT/experiments/socialnav_sub/run_pilot_jdcloud.py" --pilot-root "$OUT" \
  > "$OUT/jdcloud_pilot/background.log" 2>&1; then
  printf 'complete\n' > "$OUT/jdcloud_pilot/background.status"
else
  status=$?
  printf 'failed stage=inference exit=%s\n' "$status" > "$OUT/jdcloud_pilot/background.status"
  exit "$status"
fi
