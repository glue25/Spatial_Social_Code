#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="/pfs/zhangxiaoyuan/project/physical_code"
METRICS="$ROOT/AutoToM/results/metrics"
STATUS="$ROOT/spatial_social_code/mmtom_native_hybrid_full.status"

cd "$ROOT"
echo WAITING_FOR_47_ALIGNED_AUTOTOM_METRICS > "$STATUS"
while true; do
    count=$(find "$METRICS" -maxdepth 1 -type f \
        -name 'automated_qwen3vl_mm_required_aligned_MMToM-QA_*_back1_reduce1_seed42_metrics.json' \
        | wc -l)
    if [[ "$count" -ge 47 ]]; then
        break
    fi
    sleep 60
done

bash spatial_social_code/run_native_hybrid_full.sh
