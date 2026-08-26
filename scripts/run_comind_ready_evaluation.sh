#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${ROOT_DIR}/external_datasets/comind"
TRAIN_IDS="${DATA_ROOT}/split/train.txt"
RECORDING_WORK_ROOT="${ROOT_DIR}/workdirs/comind_full_recording"
WORK_ROOT="${ROOT_DIR}/workdirs/comind_ready_evaluation"
PYTHON_BIN="${ROOT_DIR}/cap-x/.venv-libero/bin/python"

mkdir -p "${WORK_ROOT}"
STATUS_LOG="${WORK_ROOT}/status.log"
mapfile -t RECORDING_IDS < <(awk 'NF {print $1}' "${TRAIN_IDS}")

log() {
  printf '[%s] %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$*" | tee -a "${STATUS_LOG}"
}

update_partial_report() {
  local temporary="${WORK_ROOT}/predictions.jsonl.tmp"
  : >"${temporary}"
  local recording_id predictions comparison
  for recording_id in "${RECORDING_IDS[@]}"; do
    predictions="${RECORDING_WORK_ROOT}/${recording_id}/predictions.jsonl"
    comparison="${RECORDING_WORK_ROOT}/${recording_id}/comparison.json"
    if [[ -s "${predictions}" && -s "${comparison}" ]]; then
      awk 'NF {print}' "${predictions}" >>"${temporary}"
    fi
  done
  if [[ -s "${temporary}" ]]; then
    mv "${temporary}" "${WORK_ROOT}/predictions.jsonl"
    cd "${ROOT_DIR}"
    "${PYTHON_BIN}" -m spatial_social_code.comind \
      "${WORK_ROOT}/predictions.jsonl" --paired \
      --output "${WORK_ROOT}/comparison_paired_partial.json"
  else
    rm -f "${temporary}"
  fi
}

log "Ready-data evaluation watcher started"
while true; do
  incomplete=0
  made_progress=0
  for recording_id in "${RECORDING_IDS[@]}"; do
    comparison="${RECORDING_WORK_ROOT}/${recording_id}/comparison.json"
    if [[ -s "${comparison}" ]]; then
      continue
    fi
    incomplete=$((incomplete + 1))
    video_root="${DATA_ROOT}/recordings/${recording_id}/mp4s"
    if [[ ! -s "${video_root}/helper_trimmed_sync.mp4" || \
          ! -s "${video_root}/leader_trimmed_sync.mp4" ]]; then
      continue
    fi
    log "evaluating ready recording ${recording_id}"
    until bash "${ROOT_DIR}/scripts/run_comind_full_recording.sh" "${recording_id}" \
      >>"${WORK_ROOT}/evaluation.log" 2>&1; do
      log "evaluation ${recording_id} failed; retrying in 60s"
      sleep 60
    done
    log "completed recording ${recording_id}"
    update_partial_report
    made_progress=1
  done
  if [[ "${incomplete}" -eq 0 ]]; then
    touch "${WORK_ROOT}/EVALUATION_COMPLETE"
    log "All 55 train recordings evaluated"
    exit 0
  fi
  if [[ "${made_progress}" -eq 0 ]]; then
    log "No new complete videos; waiting 60s (${incomplete} recordings pending)"
    sleep 60
  fi
done
