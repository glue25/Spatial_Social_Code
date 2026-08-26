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

cd "$ROOT/MA-EgoQA"
while ! credentials_ready; do
  printf 'state=waiting_for_gemini_credentials\ncredentials_file=%s\n' "$CREDENTIALS_FILE" > "$STATUS"
  echo "[$(date -u '+%Y-%m-%dT%H:%M:%SZ')] waiting for frozen Gemini endpoint credentials at $CREDENTIALS_FILE"
  sleep 60
done

cd "$ROOT"
printf 'state=running_mmtom_equal_call\ncredentials_file=%s\n' "$CREDENTIALS_FILE" > "$STATUS"
"$MA_PYTHON" experiments/iclr_ablations_v1/run_equal_call_direct.py \
  --credentials-file "$CREDENTIALS_FILE" --workers 4
"$MA_PYTHON" experiments/iclr_ablations_v1/run_equal_call_direct.py --score

printf 'state=running_ma_component_ablation\ncredentials_file=%s\n' "$CREDENTIALS_FILE" > "$STATUS"
"$MA_PYTHON" MA-EgoQA/egomas/src/evaluate_video_component_ablation_jdcloud.py \
  --credentials-file "$CREDENTIALS_FILE" --workers 4

printf 'state=complete\ncompleted=%s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" > "$STATUS"
touch "$OUT/GEMINI_ABLATIONS_COMPLETE"
