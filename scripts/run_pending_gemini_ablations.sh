#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$ROOT/workdirs/three_environment_ablations"
LOG="$OUT/gemini_queue.log"
STATUS="$OUT/gemini_queue.status"
MA_PYTHON="${MA_ABLATION_PYTHON:-$ROOT/MA-EgoQA/.venv/bin/python}"
CREDENTIALS_FILE="${GEMINI_CREDENTIALS_FILE:-$ROOT/credentials.env}"

mkdir -p "$OUT"
exec > >(tee -a "$LOG") 2>&1
trap 'code=$?; date -u "+stopped=%Y-%m-%dT%H:%M:%SZ" >> "$STATUS"; printf "exit_code=%s\n" "$code" >> "$STATUS"' EXIT
date -u '+started=%Y-%m-%dT%H:%M:%SZ' > "$STATUS"

credentials_ready() {
  CREDENTIALS_FILE="$CREDENTIALS_FILE" "$MA_PYTHON" - <<'PY' >/dev/null 2>&1
import os
from pathlib import Path
from egomas.src.jdcloud_gemini import load_service_config

load_service_config(Path(os.environ["CREDENTIALS_FILE"]))
PY
}

wait_for_credentials() {
  cd "$ROOT/MA-EgoQA"
  while ! credentials_ready; do
    printf 'state=waiting_for_gemini_credentials\ncredentials_file=%s\n' "$CREDENTIALS_FILE" > "$STATUS"
    echo "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] waiting for frozen Gemini endpoint credentials at $CREDENTIALS_FILE"
    sleep 60
  done
  cd "$ROOT"
}

ma_complete() {
  "$MA_PYTHON" -c 'import json, pathlib, sys; p=pathlib.Path(sys.argv[1]); d=json.loads(p.read_text()) if p.is_file() else {}; raise SystemExit(0 if d.get("successful") == 100 and d.get("errors") == 0 else 1)' \
    "$ROOT/MA-EgoQA/data/baselines/video_component_ablation_balanced100_summary.json"
}

equal_call_complete() {
  "$MA_PYTHON" -c 'import json, pathlib, sys; p=pathlib.Path(sys.argv[1]); d=json.loads(p.read_text()) if p.is_file() else {}; raise SystemExit(0 if d.get("total") == 600 and d.get("valid", 0) + d.get("invalid", 0) == 600 else 1)' \
    "$ROOT/workdirs/iclr_ablations_v1/equal_call_direct_score.json"
}

cd "$ROOT"

# MA-EgoQA is the requested priority job.  Keep retrying only missing/error
# rows so the queue survives unavailable or temporarily failing endpoints.
until ma_complete; do
  wait_for_credentials
  printf 'state=running_ma_component_ablation\ncredentials_file=%s\n' "$CREDENTIALS_FILE" > "$STATUS"
  if ! "$MA_PYTHON" MA-EgoQA/egomas/src/evaluate_video_component_ablation_jdcloud.py \
    --credentials-file "$CREDENTIALS_FILE" --workers 4; then
    echo "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] MA runner failed; retrying in 60 seconds"
  fi
  ma_complete || sleep 60
done

# Then finish the registered call-count control.  The Python runner also
# retries saved error rows while preserving successful checkpoints.
until equal_call_complete; do
  wait_for_credentials
  printf 'state=running_mmtom_equal_call\ncredentials_file=%s\n' "$CREDENTIALS_FILE" > "$STATUS"
  if ! "$MA_PYTHON" experiments/iclr_ablations_v1/run_equal_call_direct.py \
    --credentials-file "$CREDENTIALS_FILE" --workers 4; then
    echo "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] equal-call runner failed; retrying in 60 seconds"
  fi
  if ! "$MA_PYTHON" experiments/iclr_ablations_v1/run_equal_call_direct.py --score; then
    echo "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] equal-call output incomplete; retrying in 60 seconds"
    sleep 60
  fi
done

"$MA_PYTHON" experiments/iclr_ablations_v1/render_mmtom_ablation_report.py

printf 'state=complete\ncompleted=%s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" > "$STATUS"
touch "$OUT/GEMINI_ABLATIONS_COMPLETE"
