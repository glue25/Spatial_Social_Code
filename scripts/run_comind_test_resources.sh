#!/usr/bin/env bash
set -uo pipefail

# Download only the public CoMind test resources consumed by our inference
# pipeline: leader/helper synchronized MP4 views and transcripts. The broader
# official "mp4" part also contains GoPro, merged and gaze-rendered videos that
# this pipeline does not read, so those are intentionally excluded.

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
downloader="${project_root}/external/CoMind/comind_download.py"
ego_downloader="${project_root}/scripts/download_comind_pilot_ego.sh"
target="${project_root}/external_datasets/comind"
test_ids="${target}/split/test.txt"
retry_delay_seconds=300
download_workers="${COMIND_TEST_DOWNLOAD_WORKERS:-4}"

timestamp() {
  date -u '+%Y-%m-%dT%H:%M:%SZ'
}

run_until_success() {
  local label="$1"
  shift
  local attempt=1
  while true; do
    echo "[$(timestamp)] START ${label} (attempt ${attempt})"
    if "$@"; then
      echo "[$(timestamp)] DONE ${label}"
      return 0
    fi
    echo "[$(timestamp)] RETRY ${label} in ${retry_delay_seconds}s"
    attempt=$((attempt + 1))
    sleep "${retry_delay_seconds}"
  done
}

if [[ ! -s "${test_ids}" ]]; then
  echo "Missing CoMind test ID list: ${test_ids}" >&2
  exit 2
fi

if ! [[ "${download_workers}" =~ ^[1-8]$ ]]; then
  echo "COMIND_TEST_DOWNLOAD_WORKERS must be an integer from 1 to 8" >&2
  exit 2
fi

mapfile -t recording_ids < <(awk 'NF {print $1}' "${test_ids}")
if [[ "${#recording_ids[@]}" -ne 25 ]]; then
  echo "Expected 25 CoMind test recordings, found ${#recording_ids[@]}" >&2
  exit 2
fi

download_recording() {
  local recording_id="$1"
  local one_id_file="${target}/split/.current-test-${recording_id}.txt"
  printf '%s\n' "${recording_id}" > "${one_id_file}"
  run_until_success \
    "test ego videos ${recording_id}" \
    bash "${ego_downloader}" "${target}" "${recording_id}" both
  run_until_success \
    "test transcripts ${recording_id}" \
    env PYTHONUNBUFFERED=1 python "${downloader}" "${one_id_file}" \
      --target "${target}" \
      --parts transcripts \
      --no-meshes \
      --no-annotations \
      --yes \
      --retries 8 \
      --timeout 120
}

download_worker() {
  local worker_id="$1"
  local index
  for ((index = worker_id; index < ${#recording_ids[@]}; index += download_workers)); do
    download_recording "${recording_ids[index]}"
  done
}

pids=()
for ((worker_id = 0; worker_id < download_workers; worker_id++)); do
  download_worker "${worker_id}" &
  pids+=("$!")
done
for pid in "${pids[@]}"; do
  wait "${pid}"
done

touch "${target}/TEST_EGO_MP4_TRANSCRIPTS_COMPLETE"
echo "[$(timestamp)] COMPLETE all CoMind test ego MP4s and transcripts are verified"
