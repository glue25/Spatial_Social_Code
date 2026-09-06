#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PRIORITY_PYTHON:-$ROOT/cap-x/.venv-libero/bin/python}"
CREDENTIALS="${GEMINI_CREDENTIALS_FILE:-/pfs/yfsong/code/JoyVideo/AVGen-Bench/eval.sh}"
OUT="$ROOT/workdirs/priority_ssc_v1"
STATUS="$OUT/cross_model.status"
LOG="$OUT/cross_model.log"
mkdir -p "$OUT"
exec >>"$LOG" 2>&1
trap 'code=$?; printf "exit_code=%s\nstopped=%s\n" "$code" "$(date -u "+%Y-%m-%dT%H:%M:%SZ")" >>"$STATUS"' EXIT
cd "$ROOT"

reasoning_complete() {
  "$PYTHON" -c 'import json,pathlib,sys; p=pathlib.Path(sys.argv[1]); d=json.loads(p.read_text()) if p.is_file() else {}; raise SystemExit(0 if d.get("fully_complete")==d.get("selected")==100 else 1)' "$1"
}

for dataset in mmtom maegoqa; do
  printf 'state=running_%s_gemini\nupdated=%s\n' "$dataset" "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" >"$STATUS"
  summary="$OUT/${dataset}_jdcloud_reasoning_summary.json"
  until reasoning_complete "$summary"; do
    "$PYTHON" -u experiments/priority_ssc_v1/run_reasoning_baselines.py \
      --dataset "$dataset" --backend jdcloud --total 100 \
      --credentials-file "$CREDENTIALS" --min-request-interval 65
    reasoning_complete "$summary" || sleep 60
  done
done

printf 'state=running_attribution_gemini\nupdated=%s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" >"$STATUS"
until "$PYTHON" -c 'import json,pathlib; p=pathlib.Path("workdirs/priority_ssc_v1/attribution_jdcloud_summary.json"); d=json.loads(p.read_text()) if p.is_file() else {}; raise SystemExit(0 if d.get("complete_pairs")==d.get("selected_pairs")==100 else 1)'; do
  "$PYTHON" -u experiments/priority_ssc_v1/run_attribution_intervention.py \
    --backend jdcloud --total 100 --credentials-file "$CREDENTIALS" --min-request-interval 65
  sleep 60
done

bash experiments/priority_ssc_v1/publish_results.sh
printf 'state=complete\ncompleted=%s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" >"$STATUS"
