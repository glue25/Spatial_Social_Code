#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
OUT="$ROOT/workdirs/socialnav_sub/pilot_v2"
STATUS="$OUT/jdcloud_pilot/plus_frames_v4.status"
LOG="$OUT/jdcloud_pilot/plus_frames_v4.log"

exec > >(tee -a "$LOG") 2>&1
trap 'code=$?; printf "exit_code=%s\n" "$code" >> "$STATUS"' EXIT

export GEMINI_API_KEY="${GEMINI_API_KEY:-${PROXY_OPENAI_API_KEY:-}}"
export GEMINI_API_BASE="${GEMINI_API_BASE:-https://modelservice.jdcloud.com/v1/responses}"
export GEMINI_MODEL="${GEMINI_MODEL:-Gemini 3-Pro-Preview}"
if [[ -z "$GEMINI_API_KEY" ]]; then
  printf 'state=blocked_missing_credentials\n' > "$STATUS"
  exit 2
fi

printf 'state=running\nstarted=%s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" > "$STATUS"
complete() {
  python -c 'import json, pathlib, sys; p=pathlib.Path(sys.argv[1]); rows=[json.loads(x) for x in p.read_text().splitlines() if x] if p.is_file() else []; keys={(r.get("scene_id"), r.get("question_key")) for r in rows if r.get("condition")=="visual_predicted_ps_plus_frames" and r.get("answer") is not None}; raise SystemExit(0 if len(keys)==60 else 1)' \
    "$OUT/jdcloud_pilot/predictions.jsonl"
}
while ! complete; do
  if ! PYTHONPATH="$ROOT/spatial_social_code:$ROOT/MA-EgoQA" \
    python -u "$ROOT/experiments/socialnav_sub/run_pilot_jdcloud.py" \
    --pilot-root "$OUT" \
    --conditions visual_predicted_ps_plus_frames \
    --request-delay-seconds 65; then
    printf 'state=rate_limited_retrying\n' > "$STATUS"
    sleep 60
    printf 'state=running\n' > "$STATUS"
  fi
done
printf 'state=complete\ncompleted=%s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" > "$STATUS"
