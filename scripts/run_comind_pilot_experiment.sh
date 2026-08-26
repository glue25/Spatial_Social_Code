#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
RECORDING_ID="${1:-222861ac-b5d7-4f92-b682-d6c639ede7eb}"
LIMIT_PER_TASK="${2:-3}"
DATA_ROOT="${ROOT_DIR}/external_datasets/comind"
WORK_ROOT="${ROOT_DIR}/workdirs/comind_pilot/scored_${LIMIT_PER_TASK}pt"
PYTHON_BIN="${ROOT_DIR}/cap-x/.venv-libero/bin/python"

mkdir -p "${WORK_ROOT}"

bash "${ROOT_DIR}/scripts/download_comind_pilot_ego.sh" \
  "${DATA_ROOT}" "${RECORDING_ID}" helper \
  >"${WORK_ROOT}/download_helper.log" 2>&1 &
HELPER_PID=$!
bash "${ROOT_DIR}/scripts/download_comind_pilot_ego.sh" \
  "${DATA_ROOT}" "${RECORDING_ID}" leader \
  >"${WORK_ROOT}/download_leader.log" 2>&1 &
LEADER_PID=$!

wait "${HELPER_PID}"
wait "${LEADER_PID}"

cd "${ROOT_DIR}"
"${PYTHON_BIN}" -m spatial_social_code.prepare_comind_pilot \
  --recording-id "${RECORDING_ID}" \
  --output-root "${WORK_ROOT}/frames" \
  --limit-per-task "${LIMIT_PER_TASK}" \
  --image-size 1408 \
  >"${WORK_ROOT}/prepare.log" 2>&1

"${PYTHON_BIN}" -m spatial_social_code.run_comind_codex_pilot \
  --backend local-qwen \
  --local-model-path /pfs/xiabin/models/Qwen3-VL-8B-Instruct \
  --manifest "${WORK_ROOT}/frames/manifest.jsonl" \
  --output "${WORK_ROOT}/codex_predictions.jsonl" \
  >"${WORK_ROOT}/runner.log" 2>&1

"${PYTHON_BIN}" -m spatial_social_code.comind \
  "${WORK_ROOT}/codex_predictions.jsonl" \
  --output "${WORK_ROOT}/comparison.json"

printf 'CoMind pilot experiment complete: %s\n' "${WORK_ROOT}/comparison.json"
