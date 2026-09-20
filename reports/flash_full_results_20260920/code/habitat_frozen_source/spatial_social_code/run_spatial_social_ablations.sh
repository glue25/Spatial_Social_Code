#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="/pfs/zhangxiaoyuan/project/physical_code"
SAMPLES_PER_TYPE="${1:-100}"
SHARDS="${2:-2}"

cd "$ROOT"
for source in neural rules bayesian neural_rules hybrid; do
    bash spatial_social_code/run_spatial_social_full_sharded.sh \
        "$SAMPLES_PER_TYPE" "$SHARDS" "$source" "mmtom_spatial_social_${source}"
    "$ROOT/cap-x/.venv-libero/bin/python" \
        spatial_social_code/analyze_spatial_social.py \
        "spatial_social_code/mmtom_spatial_social_${source}_merged.jsonl" \
        --output "spatial_social_code/mmtom_spatial_social_${source}_analysis.json"
done
