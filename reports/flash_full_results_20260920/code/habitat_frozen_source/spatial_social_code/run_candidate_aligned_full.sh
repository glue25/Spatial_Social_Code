#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="/pfs/zhangxiaoyuan/project/physical_code"
PYTHON="$ROOT/cap-x/.venv-libero/bin/python"
PREFIX="$ROOT/spatial_social_code/mmtom_candidate_aligned_full"
STATUS="${PREFIX}.status"
LOG="${PREFIX}.log"

cd "$ROOT"
exec >> "$LOG" 2>&1
trap 'rc=$?; if [[ "$rc" -eq 0 ]]; then echo COMPLETE > "$STATUS"; else echo "FAILED:$rc" > "$STATUS"; fi' EXIT

echo RUNNING > "$STATUS"
"$PYTHON" spatial_social_code/evaluate_spatial_social.py \
    --samples-per-type 100 \
    --social-source hybrid \
    --conditions spatial_social \
    --output "${PREFIX}.jsonl" \
    --summary "${PREFIX}_summary.json"

"$PYTHON" spatial_social_code/compare_native_autotom.py \
    "${PREFIX}.jsonl" \
    --output "${PREFIX}_vs_autotom.json" \
    --examples-output "${PREFIX}_native_wins.json"
