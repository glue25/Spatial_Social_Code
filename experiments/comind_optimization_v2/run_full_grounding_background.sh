#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$ROOT"
PY=cap-x/.venv-libero/bin/python
OUT=workdirs/comind_optimization_v2
exec > "$OUT/full_grounding_background.log" 2>&1

trap 'code=$?; date -u "+stopped=%Y-%m-%dT%H:%M:%SZ" >> "$OUT/full_grounding_status.txt"; echo "exit_code=$code" >> "$OUT/full_grounding_status.txt"' EXIT

date -u '+started=%Y-%m-%dT%H:%M:%SZ' > "$OUT/full_grounding_status.txt"

$PY experiments/comind_optimization_v2/prepare_full_gpu_manifest.py \
  --output "$OUT/gpu_grounding_full_manifest.jsonl"

if [[ ! -f "$OUT/two_stage_grounding_full_predictions.jsonl" ]]; then
  cp "$OUT/two_stage_grounding_predictions.jsonl" "$OUT/two_stage_grounding_full_predictions.jsonl"
fi

$PY experiments/comind_optimization_v2/run_two_stage_grounding.py \
  --manifest "$OUT/gpu_grounding_full_manifest.jsonl" \
  --output "$OUT/two_stage_grounding_full_predictions.jsonl" \
  --raw-root "$OUT/two_stage_full_raw" \
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
