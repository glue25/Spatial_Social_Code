#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RECORDING_ID="${1:-222861ac-b5d7-4f92-b682-d6c639ede7eb}"
DATA_ROOT="${ROOT_DIR}/external_datasets/comind"
WORK_ROOT="${ROOT_DIR}/workdirs/comind_full_recording/${RECORDING_ID}"
PYTHON_BIN="${ROOT_DIR}/cap-x/.venv-libero/bin/python"
MODEL_PATH="/pfs/xiabin/models/Qwen3-VL-8B-Instruct"
LOCK_ROOT="${ROOT_DIR}/workdirs/comind_locks"

mkdir -p "${WORK_ROOT}" "${LOCK_ROOT}"

# Prevent duplicate evaluation when the ready-data worker and the final full
# pipeline meet at the same recording.
exec 8>"${LOCK_ROOT}/${RECORDING_ID}.lock"
flock 8
if [[ -s "${WORK_ROOT}/comparison.json" ]]; then
  printf 'CoMind recording already complete: %s\n' "${WORK_ROOT}/comparison.json"
  exit 0
fi

# Re-check both immutable source videos before extracting evaluation frames.
bash "${ROOT_DIR}/scripts/download_comind_pilot_ego.sh" \
  "${DATA_ROOT}" "${RECORDING_ID}" both \
  >"${WORK_ROOT}/verify_videos.log" 2>&1

cd "${ROOT_DIR}"
"${PYTHON_BIN}" -m spatial_social_code.prepare_comind_pilot \
  --recording-id "${RECORDING_ID}" \
  --output-root "${WORK_ROOT}/frames" \
  --limit-per-task 0 \
  --image-size 1408 \
  >"${WORK_ROOT}/prepare.log" 2>&1

# One local Qwen process at a time. The GPU can temporarily disappear from this
# pod, so wait instead of turning an otherwise valid recording into error rows.
exec 9>"${LOCK_ROOT}/local_qwen_gpu.lock"
flock 9
while ! nvidia-smi -L >"${WORK_ROOT}/gpu_wait.log" 2>&1; do
  sleep 60
done

SPATIAL_SOCIAL_MODEL_PATH="${MODEL_PATH}" \
SPATIAL_SOCIAL_LOCAL_FILES_ONLY=1 \
"${PYTHON_BIN}" -m spatial_social_code.run_comind_codex_pilot \
  --backend local-qwen \
  --local-model-path "${MODEL_PATH}" \
  --manifest "${WORK_ROOT}/frames/manifest.jsonl" \
  --output "${WORK_ROOT}/predictions.jsonl" \
  >"${WORK_ROOT}/runner.log" 2>&1

"${PYTHON_BIN}" -m spatial_social_code.comind \
  "${WORK_ROOT}/predictions.jsonl" \
  --output "${WORK_ROOT}/comparison.json"

printf 'CoMind full-recording evaluation complete: %s\n' \
  "${WORK_ROOT}/comparison.json"
