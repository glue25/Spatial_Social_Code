#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PRIORITY_PYTHON:-$ROOT/cap-x/.venv-libero/bin/python}"
OUT="$ROOT/workdirs/priority_ssc_v1"
STATUS="$OUT/queue.status"
LOG="$OUT/queue.log"
mkdir -p "$OUT"
exec >>"$LOG" 2>&1

set_status() {
  printf 'state=%s\nupdated=%s\n' "$1" "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" >"$STATUS"
}
trap 'code=$?; printf "exit_code=%s\nstopped=%s\n" "$code" "$(date -u "+%Y-%m-%dT%H:%M:%SZ")" >>"$STATUS"' EXIT
cd "$ROOT"

set_status running_mmtom_reasoning_baselines
"$PYTHON" -u experiments/priority_ssc_v1/run_reasoning_baselines.py \
  --dataset mmtom --backend local-qwen32b --total 100

set_status running_maegoqa_reasoning_baselines
"$PYTHON" -u experiments/priority_ssc_v1/run_reasoning_baselines.py \
  --dataset maegoqa --backend local-qwen32b --total 100

set_status running_agent_attribution_intervention
"$PYTHON" -u experiments/priority_ssc_v1/run_attribution_intervention.py \
  --backend local-qwen32b --total 100

if [[ ! -f "$OUT/human_audit/manifest.json" ]]; then
  "$PYTHON" experiments/priority_ssc_v1/prepare_human_audit.py
fi
while ! "$PYTHON" experiments/priority_ssc_v1/aggregate_human_audit.py; do
  set_status waiting_for_two_reviewers_and_adjudication
  sleep 60
done

set_status running_mmtom_four_cell
"$PYTHON" -u experiments/priority_ssc_v1/run_four_cell.py \
  --dataset mmtom --backend local-qwen32b

set_status running_maegoqa_four_cell
"$PYTHON" -u experiments/priority_ssc_v1/run_four_cell.py \
  --dataset maegoqa --backend local-qwen32b

set_status publishing_to_github
bash experiments/priority_ssc_v1/publish_results.sh
set_status complete
