#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
TARGET_DIR="${1:-${ROOT_DIR}/external_datasets/comind}"
RECORDING_ID="${2:-f0b49334-565a-4d30-a010-15d92e511c9a}"
ASSET_SELECTION="${3:-both}"
VIDEO_DIR="${TARGET_DIR}/recordings/${RECORDING_ID}/mp4s"
BASE_URL="https://comind.ethz.ch/dataset/${RECORDING_ID}/mp4s"
MANIFEST_PATH="${TARGET_DIR}/recordings/${RECORDING_ID}/healthcheck.json"
LOCK_DIR="${TARGET_DIR}/.download_locks"

mkdir -p "${VIDEO_DIR}" "${LOCK_DIR}"
exec 7>"${LOCK_DIR}/${RECORDING_ID}.lock"
flock 7

if [[ ! -f "${MANIFEST_PATH}" ]]; then
  mkdir -p "$(dirname "${MANIFEST_PATH}")"
  curl --fail --location --retry 100 --retry-all-errors --retry-delay 10 \
    --output "${MANIFEST_PATH}" \
    "https://comind.ethz.ch/dataset/${RECORDING_ID}/healthcheck.json"
fi

asset_metadata() {
  local filename="$1"
  python -c '
import json
import sys

manifest_path, asset_path = sys.argv[1:]
manifest = json.load(open(manifest_path, encoding="utf-8"))
for item in manifest["files"]:
    if item["path"] == asset_path:
        print(item["size"], item["hash"])
        break
else:
    raise SystemExit(f"asset not present in manifest: {asset_path}")
' "${MANIFEST_PATH}" "mp4s/${filename}"
}

download_and_verify() {
  local filename="$1"
  local expected_size="$2"
  local expected_sha256="$3"
  local destination="${VIDEO_DIR}/${filename}"
  local partial="${destination}.part"

  if [[ -f "${destination}" ]]; then
    local current_size
    current_size="$(stat -c '%s' "${destination}")"
    if [[ "${current_size}" == "${expected_size}" ]] && \
       printf '%s  %s\n' "${expected_sha256}" "${destination}" | sha256sum --check --status; then
      echo "${filename}: already downloaded and verified"
      return 0
    fi
    echo "${filename}: existing final file failed verification" >&2
    return 1
  fi

  echo "${filename}: downloading with multi-connection resume support"
  aria2c \
    --allow-overwrite=true \
    --auto-file-renaming=false \
    --connect-timeout=30 \
    --continue=true \
    --file-allocation=none \
    --lowest-speed-limit=1K \
    --max-tries=0 \
    --max-connection-per-server=16 \
    --min-split-size=16M \
    --retry-wait=10 \
    --split=16 \
    --summary-interval=30 \
    --timeout=30 \
    --dir="${VIDEO_DIR}" \
    --out="${filename}.part" \
    "${BASE_URL}/${filename}"

  local downloaded_size
  downloaded_size="$(stat -c '%s' "${partial}")"
  if [[ "${downloaded_size}" != "${expected_size}" ]]; then
    echo "${filename}: size mismatch (${downloaded_size} != ${expected_size})" >&2
    return 1
  fi
  printf '%s  %s\n' "${expected_sha256}" "${partial}" | sha256sum --check
  mv "${partial}" "${destination}"
  echo "${filename}: complete"
}

if [[ "${ASSET_SELECTION}" == "both" || "${ASSET_SELECTION}" == "helper" ]]; then
  read -r helper_size helper_sha256 < <(asset_metadata "helper_trimmed_sync.mp4")
  download_and_verify \
    "helper_trimmed_sync.mp4" \
    "${helper_size}" \
    "${helper_sha256}"
fi

if [[ "${ASSET_SELECTION}" == "both" || "${ASSET_SELECTION}" == "leader" ]]; then
  read -r leader_size leader_sha256 < <(asset_metadata "leader_trimmed_sync.mp4")
  download_and_verify \
    "leader_trimmed_sync.mp4" \
    "${leader_size}" \
    "${leader_sha256}"
fi

if [[ "${ASSET_SELECTION}" != "both" && "${ASSET_SELECTION}" != "helper" && "${ASSET_SELECTION}" != "leader" ]]; then
  echo "asset selection must be one of: both, helper, leader" >&2
  exit 2
fi

echo "CoMind pilot ego videos downloaded and verified."
