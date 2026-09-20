#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="/pfs/zhangxiaoyuan/project/physical_code"
AUTOTOM_STATUS="$ROOT/AutoToM/autotom_qwen3vl_required_frames.status"
STATUS="$ROOT/spatial_social_code/mmtom_full600.status"
LOG="$ROOT/spatial_social_code/mmtom_full600.log"

cd "$ROOT"
exec >> "$LOG" 2>&1
trap 'rc=$?; if [[ "$rc" -eq 0 ]]; then echo COMPLETE > "$STATUS"; else echo "FAILED:$rc" > "$STATUS"; fi' EXIT

echo WAITING_FOR_AUTOTOM > "$STATUS"
while true; do
    state=$(sed -n '1p' "$AUTOTOM_STATUS" 2>/dev/null || true)
    if [[ "$state" == "COMPLETE" ]]; then
        break
    fi
    if [[ "$state" == FAILED:* ]]; then
        echo "AutoToM failed before full comparison: $state" >&2
        exit 1
    fi
    sleep 60
done

echo ALIGNING_AUTOTOM > "$STATUS"
bash spatial_social_code/run_autotom_alignment_corrections.sh

echo RUNNING > "$STATUS"
bash spatial_social_code/run_mmtom_full_sharded.sh \
    100 2 mmtom_full600 \
    automated_qwen3vl_mm_required_aligned \
    automated_qwen3vl_mm_required
"$ROOT/cap-x/.venv-libero/bin/python" \
    spatial_social_code/analyze_gated_fusion.py \
    spatial_social_code/mmtom_full600_merged.jsonl \
    --validation-fraction 0.2 \
    --output spatial_social_code/mmtom_full600_gated.json
