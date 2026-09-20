#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="/pfs/zhangxiaoyuan/project/physical_code"
PYTHON="$ROOT/cap-x/.venv-libero/bin/python"
EXP="$ROOT/spatial_social_code"
SAMPLES_PER_TYPE="${1:-100}"
SHARDS="${2:-2}"
SOURCE="${3:-hybrid}"
PREFIX="${4:-mmtom_spatial_social_${SOURCE}}"

cd "$ROOT"
pids=()
outputs=()
for shard in $(seq 0 $((SHARDS - 1))); do
    output="$EXP/${PREFIX}_shard${shard}.jsonl"
    outputs+=("$output")
    "$PYTHON" spatial_social_code/evaluate_spatial_social.py \
        --samples-per-type "$SAMPLES_PER_TYPE" \
        --num-shards "$SHARDS" \
        --shard-id "$shard" \
        --social-source "$SOURCE" \
        --output "$output" \
        --summary "$EXP/${PREFIX}_shard${shard}_summary.json" \
        > "$EXP/${PREFIX}_shard${shard}.log" 2>&1 &
    pids+=("$!")
done

failed=0
for pid in "${pids[@]}"; do
    if ! wait "$pid"; then
        failed=1
    fi
done
if [[ "$failed" -ne 0 ]]; then
    echo "One or more Spatial-Social Code shards failed" >&2
    exit 1
fi

"$PYTHON" spatial_social_code/merge_spatial_social.py \
    "${outputs[@]}" \
    --output "$EXP/${PREFIX}_merged.jsonl" \
    --summary "$EXP/${PREFIX}_merged_summary.json"
