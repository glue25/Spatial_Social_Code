#!/usr/bin/env bash
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$ROOT/workdirs/habitat_enacttom/frozen12"
LOG="$OUT/background.log"
STATUS="$OUT/background.status"
PYTHON_BIN="${ENACTTOM_PYTHON:-$ROOT/.venv-habitat/bin/python}"

mkdir -p "$OUT"
exec > >(tee -a "$LOG") 2>&1
trap 'code=$?; date -u "+stopped=%Y-%m-%dT%H:%M:%SZ" >> "$STATUS"; printf "exit_code=%s\n" "$code" >> "$STATUS"' EXIT
date -u '+started=%Y-%m-%dT%H:%M:%SZ' > "$STATUS"

cd "$ROOT"
export HF_ENDPOINT="${HF_ENDPOINT:-https://hf-mirror.com}"
export HF_HUB_DISABLE_XET=1

echo "[Habitat] resume minimal HSSD/OVMM assets for frozen12"
if ! "$PYTHON_BIN" experiments/habitat_enacttom/download_frozen_assets.py; then
  echo "[Habitat] asset download failed"
  exit 1
fi

echo "[Habitat] verify all 12 frozen tasks"
if bash experiments/habitat_enacttom/wait_and_verify_frozen12.sh; then
  touch "$OUT/VERIFY_COMPLETE"
  date -u '+completed=%Y-%m-%dT%H:%M:%SZ' >> "$STATUS"
  exit 0
fi

echo "[Habitat] all tasks were attempted; one or more verification failures remain"
date -u '+verification_completed_with_failures=%Y-%m-%dT%H:%M:%SZ' >> "$STATUS"
exit 2
