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
complete() {
  "$PYTHON" -c \
    'import json,pathlib,sys; p=pathlib.Path(sys.argv[1]); d=json.loads(p.read_text()) if p.is_file() else {}; raise SystemExit(0 if d.get("successful")==600 and d.get("errors")==0 else 1)' \
    "$SUMMARY"
}

until complete; do
  printf 'state=running_or_retrying\nupdated=%s\noutput=%s\n' \
    "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$OUTPUT" >"$STATUS"
  "$PYTHON" -u spatial_social_code/evaluate_mmtom_jdcloud.py \
    --samples-per-type 100 \
    --workers 1 \
    --min-request-interval 65 \
    --stop-on-error \
    --credentials-file "$CREDENTIALS" \
    --output "$OUTPUT" || true
  complete || {
    printf 'state=waiting_after_api_error\nupdated=%s\nretry_in_seconds=600\noutput=%s\n' \
      "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$OUTPUT" >"$STATUS"
    sleep 600
  }
done

git --git-dir=.spatial_social_public_git --work-tree=. add -f -- "$OUTPUT" "$SUMMARY"
if ! git --git-dir=.spatial_social_public_git --work-tree=. diff --cached --quiet -- "$OUTPUT" "$SUMMARY"; then
  git --git-dir=.spatial_social_public_git --work-tree=. commit \
    -m "Add option-blind MMToM evaluation results" -- "$OUTPUT" "$SUMMARY"
  git --git-dir=.spatial_social_public_git --work-tree=. push origin main
fi

printf 'state=complete\ncompleted=%s\noutput=%s\n' \
  "$(date -u '+%Y-%m-%dT%H:%M:%SZ')" "$OUTPUT" >"$STATUS"
