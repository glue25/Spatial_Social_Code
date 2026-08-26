#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
IDS_FILE="${ROOT_DIR}/experiments/configs/comind_downloaded_50_ids.txt"
RECORDING_ROOT="${ROOT_DIR}/workdirs/comind_full_recording"
OUTPUT_ROOT="${ROOT_DIR}/workdirs/comind_downloaded_50"
PYTHON_BIN="${ROOT_DIR}/cap-x/.venv-libero/bin/python"

mkdir -p "${OUTPUT_ROOT}"
mapfile -t IDS < <(awk 'NF && $1 !~ /^#/ {print $1}' "${IDS_FILE}")

while true; do
  missing=0
  for recording_id in "${IDS[@]}"; do
    if [[ ! -s "${RECORDING_ROOT}/${recording_id}/comparison.json" ]]; then
      missing=$((missing + 1))
    fi
  done
  printf '[%s] evaluated=%d/50\n' \
    "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$((50 - missing))" \
    >"${OUTPUT_ROOT}/status.log"
  if [[ "${missing}" -eq 0 ]]; then
    break
  fi
  sleep 300
done

temporary="${OUTPUT_ROOT}/predictions.jsonl.tmp"
: >"${temporary}"
for recording_id in "${IDS[@]}"; do
  awk 'NF {print}' "${RECORDING_ROOT}/${recording_id}/predictions.jsonl" \
    >>"${temporary}"
done
mv "${temporary}" "${OUTPUT_ROOT}/predictions.jsonl"

cd "${ROOT_DIR}"
"${PYTHON_BIN}" -m spatial_social_code.comind \
  "${OUTPUT_ROOT}/predictions.jsonl" --paired \
  --output "${OUTPUT_ROOT}/comparison_paired.json"
"${PYTHON_BIN}" -m spatial_social_code.comind \
  "${OUTPUT_ROOT}/predictions.jsonl" \
  --output "${OUTPUT_ROOT}/comparison_all_successful.json"
touch "${OUTPUT_ROOT}/REPORT_COMPLETE"
