#!/usr/bin/env bash
set -u

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
STATUS="$ROOT/workdirs/habitat_enacttom/frozen12/asset_download_status.json"
TASKS="$ROOT/workdirs/habitat_enacttom/frozen12/tasks"
OUT="$ROOT/workdirs/habitat_enacttom/frozen12/verify"

until rg -q '"complete": true' "$STATUS" 2>/dev/null; do
  sleep 30
done

mkdir -p "$OUT"
cd "$ROOT/external/EnactTom" || exit 1
export ENACTTOM_PYTHON="$ROOT/.venv-habitat/bin/python"
export ENACTTOM_EPISODES_PATH="$ROOT/workdirs/habitat_enacttom/frozen12/episodes.json.gz"

passed=0
failed=0
attempted=0
results="$OUT/results.jsonl"
: > "$results"
for task in "$TASKS"/*.json; do
  attempted=$((attempted + 1))
  stem="$(basename "$task" .json)"
  if ./enacttom/run.sh verify \
      --task "$task" \
      --output-dir "$OUT/$stem" \
      --report-file "$OUT/$stem.json"; then
    passed=$((passed + 1))
    printf '{"task":"%s","success":true}\n' "$stem" >> "$results"
  else
    failed=$((failed + 1))
    printf '{"task":"%s","success":false}\n' "$stem" >> "$results"
  fi
done

printf '{"complete":true,"expected":12,"attempted":%d,"passed":%d,"failed":%d}\n' \
  "$attempted" "$passed" "$failed" > "$OUT/summary.json"
test "$attempted" -eq 12
test "$failed" -eq 0
