#!/usr/bin/env bash
set -uo pipefail

# Resume-aware CoMind v1 full-resource downloader.
# Downloading one recording at a time makes transient manifest failures
# retryable without re-verifying every already-completed recording.

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
downloader="${project_root}/external/CoMind/comind_download.py"
target="${project_root}/external_datasets/comind"
retry_delay_seconds=300

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

mkdir -p "${target}"

# This also writes the authoritative built-in train/test ID lists under
# <target>/split/. Common assets are tiny and are fetched before recordings.
run_until_success \
  "common assets (meshes, annotations)" \
  env PYTHONUNBUFFERED=1 python "${downloader}" \
    --target "${target}" \
    --parts meshes,annotations \
    --yes \
    --retries 8 \
    --timeout 120

for split in test train; do
  ids_file="${target}/split/${split}.txt"
  while IFS= read -r recording_id; do
    [[ -z "${recording_id}" ]] && continue
    one_id_file="${target}/split/.current-${recording_id}.txt"
    printf '%s\n' "${recording_id}" > "${one_id_file}"
    run_until_success \
      "${split} recording ${recording_id}" \
      env PYTHONUNBUFFERED=1 python "${downloader}" "${one_id_file}" \
        --target "${target}" \
        --parts all \
        --no-meshes \
        --no-annotations \
        --yes \
        --retries 8 \
        --timeout 120
  done < "${ids_file}"
done

touch "${target}/ALL_RESOURCES_COMPLETE"
echo "[$(timestamp)] COMPLETE all CoMind resources are downloaded and verified"
