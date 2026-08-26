#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
DATA_ROOT="${ROOT_DIR}/external_datasets/comind"
TRAIN_IDS="${DATA_ROOT}/split/train.txt"
RECORDING_WORK_ROOT="${ROOT_DIR}/workdirs/comind_full_recording"
SUMMARY_ROOT="${ROOT_DIR}/workdirs/comind_full_train"
PYTHON_BIN="${ROOT_DIR}/cap-x/.venv-libero/bin/python"
GPU_POLL_SECONDS="${COMIND_GPU_POLL_SECONDS:-300}"
DOWNLOAD_WORKERS="${COMIND_DOWNLOAD_WORKERS:-4}"

mkdir -p "${SUMMARY_ROOT}"
STATUS_LOG="${SUMMARY_ROOT}/status.log"

log() {
  printf '[%s] %s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$*" | tee -a "${STATUS_LOG}"
}

mapfile -t RECORDING_IDS < <(awk 'NF {print $1}' "${TRAIN_IDS}")
if [[ "${#RECORDING_IDS[@]}" -ne 55 ]]; then
  log "ERROR: expected 55 train recordings, found ${#RECORDING_IDS[@]}"
  exit 2
fi
if ! [[ "${DOWNLOAD_WORKERS}" =~ ^[1-8]$ ]]; then
  log "ERROR: COMIND_DOWNLOAD_WORKERS must be an integer from 1 to 8"
  exit 2
fi

DOWNLOAD_LOG_ROOT="${SUMMARY_ROOT}/downloads"
mkdir -p "${DOWNLOAD_LOG_ROOT}"

download_worker() {
  local worker_id="$1"
  local index recording_id
  for ((index = worker_id; index < ${#RECORDING_IDS[@]}; index += DOWNLOAD_WORKERS)); do
    recording_id="${RECORDING_IDS[index]}"
    log "download $((index + 1))/55 ${recording_id}: worker $((worker_id + 1)) starting"
    until bash "${ROOT_DIR}/scripts/download_comind_pilot_ego.sh" \
      "${DATA_ROOT}" "${recording_id}" both \
      >>"${DOWNLOAD_LOG_ROOT}/${recording_id}.log" 2>&1; do
      log "download ${recording_id}: transient failure; retrying in 60s"
      sleep 60
    done
    touch "${DOWNLOAD_LOG_ROOT}/${recording_id}.complete"
    log "download $((index + 1))/55 ${recording_id}: complete"
  done
}

log "Stage 1/4: downloading and SHA-256 verifying train MP4s with ${DOWNLOAD_WORKERS} workers"
download_pids=()
for ((worker_id = 0; worker_id < DOWNLOAD_WORKERS; worker_id++)); do
  download_worker "${worker_id}" &
  download_pids+=("$!")
done
download_failed=0
for pid in "${download_pids[@]}"; do
  if ! wait "${pid}"; then
    download_failed=1
  fi
done
if [[ "${download_failed}" -ne 0 ]]; then
  log "ERROR: one or more download workers failed"
  exit 4
fi
touch "${SUMMARY_ROOT}/DOWNLOAD_COMPLETE"
log "All 55 train recordings downloaded and verified"

log "Stage 2/4: waiting for a usable NVIDIA GPU"
while ! nvidia-smi -L >>"${SUMMARY_ROOT}/gpu_probe.log" 2>&1; do
  log "GPU unavailable; retrying in ${GPU_POLL_SECONDS}s"
  sleep "${GPU_POLL_SECONDS}"
done
log "GPU available"

log "Stage 3/4: running paired Direct and Spatial-Social evaluation"
for index in "${!RECORDING_IDS[@]}"; do
  recording_id="${RECORDING_IDS[index]}"
  comparison="${RECORDING_WORK_ROOT}/${recording_id}/comparison.json"
  if [[ -s "${comparison}" ]]; then
    log "evaluate $((index + 1))/55 ${recording_id}: already complete"
    continue
  fi
  log "evaluate $((index + 1))/55 ${recording_id}: starting"
  bash "${ROOT_DIR}/scripts/run_comind_full_recording.sh" "${recording_id}" \
    >>"${SUMMARY_ROOT}/evaluation.log" 2>&1
  log "evaluate $((index + 1))/55 ${recording_id}: complete"
done

log "Stage 4/4: merging predictions and scoring"
MERGED_TMP="${SUMMARY_ROOT}/predictions.jsonl.tmp"
: >"${MERGED_TMP}"
for recording_id in "${RECORDING_IDS[@]}"; do
  predictions="${RECORDING_WORK_ROOT}/${recording_id}/predictions.jsonl"
  if [[ ! -s "${predictions}" ]]; then
    log "ERROR: missing predictions for ${recording_id}"
    exit 3
  fi
  awk 'NF {print}' "${predictions}" >>"${MERGED_TMP}"
done
mv "${MERGED_TMP}" "${SUMMARY_ROOT}/predictions.jsonl"

cd "${ROOT_DIR}"
"${PYTHON_BIN}" -m spatial_social_code.comind \
  "${SUMMARY_ROOT}/predictions.jsonl" \
  --output "${SUMMARY_ROOT}/comparison_all_successful.json"
"${PYTHON_BIN}" -m spatial_social_code.comind \
  "${SUMMARY_ROOT}/predictions.jsonl" \
  --paired \
  --output "${SUMMARY_ROOT}/comparison_paired.json"
touch "${SUMMARY_ROOT}/PIPELINE_COMPLETE"
log "Full train pipeline complete: ${SUMMARY_ROOT}/comparison_paired.json"
