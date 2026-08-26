#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
PY=cap-x/.venv-libero/bin/python
OUT=workdirs/comind_optimization_v2
LOG="$OUT/full_grounding_parallel.log"
exec > "$LOG" 2>&1
trap 'code=$?; date -u "+parallel_stopped=%Y-%m-%dT%H:%M:%SZ" >> "$OUT/full_grounding_status.txt"; echo "parallel_exit_code=$code" >> "$OUT/full_grounding_status.txt"' EXIT

date -u '+parallel_started=%Y-%m-%dT%H:%M:%SZ' >> "$OUT/full_grounding_status.txt"

$PY experiments/comind_optimization_v2/prepare_grounding_shards.py \
  --manifest "$OUT/gpu_grounding_full_manifest.jsonl" \
  --existing "$OUT/two_stage_grounding_full_predictions.jsonl" \
  --output-prefix "$OUT/gpu_grounding_remaining_shard" \
  --shards 2

pids=()
for shard in 0 1; do
  $PY experiments/comind_optimization_v2/run_two_stage_grounding.py \
    --manifest "$OUT/gpu_grounding_remaining_shard_${shard}.jsonl" \
    --output "$OUT/two_stage_grounding_shard_${shard}.jsonl" \
    --raw-root "$OUT/two_stage_shard_${shard}_raw" \
    --limit 0 &
  pids+=("$!")
done
for pid in "${pids[@]}"; do
  wait "$pid"
done

$PY experiments/comind_optimization_v2/merge_grounding_outputs.py \
  --manifest "$OUT/gpu_grounding_full_manifest.jsonl" \
  --inputs "$OUT/two_stage_grounding_full_predictions.jsonl" "$OUT/two_stage_grounding_shard_0.jsonl" "$OUT/two_stage_grounding_shard_1.jsonl" \
  --output "$OUT/two_stage_grounding_full_merged.jsonl"
mv "$OUT/two_stage_grounding_full_merged.jsonl" "$OUT/two_stage_grounding_full_predictions.jsonl"

# A final single-worker pass skips all successes and retries only failed or
# missing keys, making the parallel run fully resumable and complete.
$PY experiments/comind_optimization_v2/run_two_stage_grounding.py \
  --manifest "$OUT/gpu_grounding_full_manifest.jsonl" \
  --output "$OUT/two_stage_grounding_full_predictions.jsonl" \
  --raw-root "$OUT/two_stage_full_retry_raw" \
  --limit 0

$PY experiments/comind_optimization_v2/evaluate_gpu_grounding.py \
  "$OUT/two_stage_grounding_full_predictions.jsonl" \
  --output "$OUT/two_stage_grounding_full_evaluation.json"

$PY experiments/comind_optimization_v2/compare_balanced_pilot.py \
  --manifest "$OUT/gpu_grounding_full_manifest.jsonl" \
  --base workdirs/comind_full_train/predictions.jsonl \
  --two-stage "$OUT/two_stage_grounding_full_predictions.jsonl" \
  --per-task 0 \
  --baseline-label 'local Qwen3-VL-8B full-train baseline (same backbone)' \
  --output "$OUT/two_stage_full_vs_local8b.json" \
  --report "$OUT/two_stage_full_vs_local8b.md"

$PY experiments/comind_optimization_v2/compare_oof_identity_gate.py \
  --manifest "$OUT/gpu_grounding_full_manifest.jsonl" \
  --base workdirs/comind_full_train/predictions.jsonl \
  --two-stage "$OUT/two_stage_grounding_full_predictions.jsonl" \
  --output "$OUT/two_stage_full_oof_gate.json" \
  --report "$OUT/two_stage_full_oof_gate.md"

date -u '+completed=%Y-%m-%dT%H:%M:%SZ' >> "$OUT/full_grounding_status.txt"
touch "$OUT/FULL_GROUNDING_COMPLETE"
