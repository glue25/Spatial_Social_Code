#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$ROOT/workdirs/iclr_ablations_v1"
LOG="$OUT/ma_reasoning_baselines.log"
STATUS="$OUT/ma_reasoning_baselines.status"
PYTHON_BIN="${MA_ABLATION_PYTHON:-$ROOT/MA-EgoQA/.venv/bin/python}"
RUNNER="$ROOT/experiments/iclr_ablations_v1/run_maegoqa_reasoning_baselines.py"

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

cd "$ROOT"
printf 'state=running_frozen100\n' > "$STATUS"
while true; do
  # Repairs request a larger output budget, so keep concurrency at one to stay
  # below the service's tokens-per-minute limit.
  "$PYTHON_BIN" -u "$RUNNER" --workers 1
  if "$PYTHON_BIN" - <<'PY'
import json
from pathlib import Path

path = Path("workdirs/iclr_ablations_v1/ma_reasoning_baselines_summary.json")
summary = json.loads(path.read_text(encoding="utf-8"))
raise SystemExit(
    0 if summary.get("successful") == 100 and summary.get("errors") == 0 else 1
)
PY
  then
    break
  fi
  printf 'state=repairing_invalid_outputs\n' > "$STATUS"
  sleep 30
done

printf 'state=complete\ncompleted=%s\n' "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" > "$STATUS"
