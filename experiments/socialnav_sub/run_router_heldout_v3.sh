#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT="$ROOT/workdirs/socialnav_sub/router_v3_heldout"
mkdir -p "$OUT/jdcloud_pilot"
printf 'preparing\n' > "$OUT/jdcloud_pilot/background.status"

python "$ROOT/experiments/socialnav_sub/prepare_pilot.py" \
  --output-root "$OUT" --scene-start 20 --n-scenes 40 --oracle-optional \
  > "$OUT/prepare.log" 2>&1

export GEMINI_API_BASE="${GEMINI_API_BASE:-https://modelservice.jdcloud.com/v1/responses}"
export GEMINI_MODEL="${GEMINI_MODEL:-Gemini 3-Pro-Preview}"
if [[ -z "${GEMINI_API_KEY:-}" ]]; then
  printf 'failed stage=config missing=GEMINI_API_KEY\n' > "$OUT/jdcloud_pilot/background.status"
  exit 2
fi

printf 'running\n' > "$OUT/jdcloud_pilot/background.status"
PYTHONPATH="$ROOT/spatial_social_code:$ROOT/MA-EgoQA" python -u \
  "$ROOT/experiments/socialnav_sub/run_pilot_jdcloud.py" \
  --pilot-root "$OUT" --conditions direct visual_predicted_ps \
  > "$OUT/jdcloud_pilot/background.log" 2>&1
python "$ROOT/experiments/socialnav_sub/summarize_routed.py" \
  --predictions "$OUT/jdcloud_pilot/predictions.jsonl" \
  --output "$OUT/jdcloud_pilot/routed_summary.json" \
  >> "$OUT/jdcloud_pilot/background.log" 2>&1
printf 'complete\n' > "$OUT/jdcloud_pilot/background.status"
