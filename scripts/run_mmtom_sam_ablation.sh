#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$ROOT/workdirs/iclr_ablations_v1"
LOG="$OUT/mmtom_sam_ablation.log"
STATUS="$OUT/mmtom_sam_ablation.status"
PYTHON_BIN="${MA_ABLATION_PYTHON:-$ROOT/MA-EgoQA/.venv/bin/python}"
RUNNER="$ROOT/experiments/iclr_ablations_v1/run_mmtom_sam_ablation.py"

mkdir -p "$OUT"
exec > >(tee -a "$LOG") 2>&1
trap 'code=$?; printf "exit_code=%s\n" "$code" >> "$STATUS"' EXIT
printf 'state=starting\nstarted=%s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" > "$STATUS"

export GEMINI_API_KEY="${GEMINI_API_KEY:-${PROXY_OPENAI_API_KEY:-}}"
export GEMINI_API_BASE="${GEMINI_API_BASE:-https://modelservice.jdcloud.com/v1/responses}"
export GEMINI_MODEL="${GEMINI_MODEL:-Gemini 3-Pro-Preview}"
if [[ -z "$GEMINI_API_KEY" ]]; then
  printf 'state=blocked_missing_credentials\n' > "$STATUS"
  exit 2
fi
curl -fsS http://127.0.0.1:8114/health >/dev/null

cd "$ROOT"
run_until_complete() {
  local samples_per_type="$1"
  local expected="$2"
  while true; do
    "$PYTHON_BIN" -u "$RUNNER" --samples-per-type "$samples_per_type" --workers 2
    if "$PYTHON_BIN" - "$expected" <<'PY'
import json
import sys
from pathlib import Path

expected = int(sys.argv[1])
path = Path("workdirs/iclr_ablations_v1/mmtom_sam_ablation_summary.json")
summary = json.loads(path.read_text(encoding="utf-8"))
raise SystemExit(
    0
    if summary.get("successful") == expected and summary.get("errors") == 0
    else 1
)
PY
    then
      break
    fi
    printf 'state=retrying_errors_%s\n' "$expected" > "$STATUS"
  done
}

printf 'state=running_frozen14\n' > "$STATUS"
run_until_complete 2 14
printf 'state=expanding_frozen70\n' > "$STATUS"
run_until_complete 10 70

printf 'state=complete\ncompleted=%s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" > "$STATUS"
