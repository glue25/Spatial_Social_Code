#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON="${PRIORITY_PYTHON:-$ROOT/cap-x/.venv-libero/bin/python}"
CREDENTIALS="${GEMINI_CREDENTIALS_FILE:-/pfs/yfsong/code/JoyVideo/AVGen-Bench/eval.sh}"
OUTPUT="$ROOT/spatial_social_code/mmtom_option_blind_gemini3pro.jsonl"
SUMMARY="$ROOT/spatial_social_code/mmtom_option_blind_gemini3pro_summary.json"
RUN_DIR="$ROOT/workdirs/priority_ssc_v1"
STATUS="$RUN_DIR/option_blind_mmtom.status"
LOG="$RUN_DIR/option_blind_mmtom.log"
mkdir -p "$RUN_DIR"
exec >>"$LOG" 2>&1

finish() {
  code=$?
  printf 'exit_code=%s\nstopped=%s\n' "$code" "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" >>"$STATUS"
}
trap finish EXIT

printf 'state=running\nstarted=%s\noutput=%s\n' \
  "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$OUTPUT" >"$STATUS"

cd "$ROOT"
"$PYTHON" -u spatial_social_code/evaluate_mmtom_jdcloud.py \
  --samples-per-type 100 \
  --workers 1 \
  --min-request-interval 65 \
  --credentials-file "$CREDENTIALS" \
  --output "$OUTPUT"

git --git-dir=.spatial_social_public_git --work-tree=. add -f -- "$OUTPUT" "$SUMMARY"
if ! git --git-dir=.spatial_social_public_git --work-tree=. diff --cached --quiet -- "$OUTPUT" "$SUMMARY"; then
  git --git-dir=.spatial_social_public_git --work-tree=. commit \
    -m "Add option-blind MMToM evaluation results" -- "$OUTPUT" "$SUMMARY"
  git --git-dir=.spatial_social_public_git --work-tree=. push origin main
fi

printf 'state=complete\ncompleted=%s\noutput=%s\n' \
  "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$OUTPUT" >"$STATUS"
