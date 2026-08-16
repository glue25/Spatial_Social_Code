#!/usr/bin/env bash
set -euo pipefail

PROJECT_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
CODE_ROOT="${PROJECT_ROOT}/external"
DATA_ROOT="${PROJECT_ROOT}/external_datasets"
HF_BASE="${HF_ENDPOINT:-https://hf-mirror.com}"

mkdir -p \
  "${CODE_ROOT}" \
  "${CODE_ROOT}/CoMind" \
  "${DATA_ROOT}/holoassist/annotations" \
  "${DATA_ROOT}/holoassist/splits"

clone_if_missing() {
  local url="$1"
  local destination="$2"
  if [[ -d "${destination}/.git" ]]; then
    echo "READY code: ${destination}"
  elif [[ -e "${destination}" ]]; then
    echo "SKIP non-git path already exists: ${destination}"
  else
    git clone --depth 1 "${url}" "${destination}"
  fi
}

download_if_missing() {
  local url="$1"
  local destination="$2"
  if [[ -s "${destination}" ]]; then
    echo "READY data: ${destination}"
  else
    curl --fail --location --retry 3 --connect-timeout 20 \
      --output "${destination}.partial" "${url}"
    mv "${destination}.partial" "${destination}"
  fi
}

clone_if_missing \
  "https://github.com/Buzz-Beater/EgoTaskQA.git" \
  "${CODE_ROOT}/EgoTaskQA"
clone_if_missing \
  "https://github.com/taeinkwon/PyHoloAssist.git" \
  "${CODE_ROOT}/PyHoloAssist"
clone_if_missing \
  "https://github.com/Ember-HoloAssist/holoassist-release.git" \
  "${CODE_ROOT}/holoassist-release"
clone_if_missing \
  "https://github.com/KangsanKim07/MA-EgoQA.git" \
  "${CODE_ROOT}/MA-EgoQA"
clone_if_missing \
  "https://github.com/UCSB-AI/EnactTom.git" \
  "${CODE_ROOT}/EnactTom"
download_if_missing \
  "https://comind.ethz.ch/scripts/comind_download.py" \
  "${CODE_ROOT}/CoMind/comind_download.py"

download_if_missing \
  "https://hl2data.z5.web.core.windows.net/holoassist-data-release/data-annotation-trainval-v1_1.json" \
  "${DATA_ROOT}/holoassist/annotations/data-annotation-trainval-v1_1.json"
download_if_missing \
  "https://holoassist.github.io/label_files/data-splits-v1_2.zip" \
  "${DATA_ROOT}/holoassist/splits/data-splits-v1_2.zip"

if [[ ! -s "${DATA_ROOT}/holoassist/splits/train-v1_2.txt" ]]; then
  unzip -o \
    "${DATA_ROOT}/holoassist/splits/data-splits-v1_2.zip" \
    -d "${DATA_ROOT}/holoassist/splits"
fi

if [[ -d "${DATA_ROOT}/ma_egoqa/.git" ]]; then
  echo "READY data: ${DATA_ROOT}/ma_egoqa"
else
  git clone --depth 1 \
    "${HF_BASE}/datasets/KangsanKim71/MA-EgoQA" \
    "${DATA_ROOT}/ma_egoqa"
fi

if [[ -d "${DATA_ROOT}/enacttom/.git" ]]; then
  echo "READY data: ${DATA_ROOT}/enacttom"
else
  git clone --depth 1 \
    "${HF_BASE}/datasets/ucsbai/enact_tom_300_public" \
    "${DATA_ROOT}/enacttom"
fi

echo
echo "Automatic preparation complete."
echo "EgoTaskQA data still requires its license form:"
echo "  https://sites.google.com/view/egotaskqa/download"
echo "CoMind transcript pilot command:"
echo "  python3 external/CoMind/comind_download.py experiments/configs/comind_pilot_ids.txt --target external_datasets/comind --no-meshes --parts transcripts,annotations --yes"
echo "Large raw videos and Habitat/HSSD assets are intentionally not downloaded."
echo "Run: python3 scripts/validate_datasets.py"
