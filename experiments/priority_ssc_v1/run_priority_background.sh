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

reasoning_complete() {
  "$PYTHON" -c 'import json,pathlib,sys; p=pathlib.Path(sys.argv[1]); d=json.loads(p.read_text()) if p.is_file() else {}; raise SystemExit(0 if d.get("fully_complete")==d.get("selected")==100 else 1)' "$1"
}

run_reasoning_until_complete() {
  local dataset="$1"
  local summary="$OUT/${dataset}_local-qwen32b_reasoning_summary.json"
  until reasoning_complete "$summary"; do
    "$PYTHON" -u experiments/priority_ssc_v1/run_reasoning_baselines.py \
      --dataset "$dataset" --backend local-qwen32b --total 100
    reasoning_complete "$summary" || sleep 60
  done
}

set_status running_mmtom_reasoning_baselines
run_reasoning_until_complete mmtom

set_status running_maegoqa_reasoning_baselines
run_reasoning_until_complete maegoqa

set_status running_agent_attribution_intervention
until "$PYTHON" -c 'import json,pathlib; p=pathlib.Path("workdirs/priority_ssc_v1/attribution_local-qwen32b_summary.json"); d=json.loads(p.read_text()) if p.is_file() else {}; raise SystemExit(0 if d.get("complete_pairs")==d.get("selected_pairs")==100 else 1)'; do
  "$PYTHON" -u experiments/priority_ssc_v1/run_attribution_intervention.py \
    --backend local-qwen32b --total 100
  sleep 60
done

if [[ ! -f "$OUT/human_audit/manifest.json" ]]; then
  "$PYTHON" experiments/priority_ssc_v1/prepare_human_audit.py
fi
while ! "$PYTHON" experiments/priority_ssc_v1/aggregate_human_audit.py; do
  set_status waiting_for_two_reviewers_and_adjudication
  sleep 60
done

set_status running_mmtom_four_cell
until "$PYTHON" -c 'import json,pathlib; p=pathlib.Path("workdirs/priority_ssc_v1/mmtom_local-qwen32b_four_cell_summary.json"); d=json.loads(p.read_text()) if p.is_file() else {}; raise SystemExit(0 if d.get("fully_complete")==100 else 1)'; do
  "$PYTHON" -u experiments/priority_ssc_v1/run_four_cell.py \
    --dataset mmtom --backend local-qwen32b
  sleep 60
done

set_status running_maegoqa_four_cell
until "$PYTHON" -c 'import json,pathlib; p=pathlib.Path("workdirs/priority_ssc_v1/maegoqa_local-qwen32b_four_cell_summary.json"); d=json.loads(p.read_text()) if p.is_file() else {}; raise SystemExit(0 if d.get("fully_complete")==100 else 1)'; do
  "$PYTHON" -u experiments/priority_ssc_v1/run_four_cell.py \
    --dataset maegoqa --backend local-qwen32b
  sleep 60
done

set_status publishing_to_github
bash experiments/priority_ssc_v1/publish_results.sh
set_status complete
