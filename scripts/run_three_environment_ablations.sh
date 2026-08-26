#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
OUT="$ROOT/workdirs/three_environment_ablations"
LOG="$OUT/local_refresh.log"
STATUS="$OUT/local_refresh.status"
PYTHON_BIN="${ABLATION_PYTHON:-$ROOT/cap-x/.venv-libero/bin/python}"
MA_PYTHON="${MA_ABLATION_PYTHON:-$ROOT/MA-EgoQA/.venv/bin/python}"

mkdir -p "$OUT"
exec > >(tee -a "$LOG") 2>&1
trap 'code=$?; date -u "+stopped=%Y-%m-%dT%H:%M:%SZ" >> "$STATUS"; printf "exit_code=%s\n" "$code" >> "$STATUS"' EXIT
date -u '+started=%Y-%m-%dT%H:%M:%SZ' > "$STATUS"

cd "$ROOT"

echo "[MMToM] refresh deterministic component ablations"
"$PYTHON_BIN" experiments/iclr_ablations_v1/run_offline_mmtom_ablations.py
"$PYTHON_BIN" experiments/iclr_ablations_v1/run_mmtom_execution_effect.py --score

echo "[CoMind] refresh recording-heldout grounding and weak-field audits"
"$PYTHON_BIN" experiments/comind_optimization_v2/audit_full_oof_run.py \
  --manifest workdirs/comind_optimization_v2/gpu_grounding_full_manifest.jsonl \
  --grounding workdirs/comind_optimization_v2/two_stage_grounding_full_predictions.jsonl \
  --baseline workdirs/comind_full_train/predictions.jsonl \
  --oof workdirs/comind_optimization_v2/two_stage_full_oof_gate.json \
  --output workdirs/comind_optimization_v2/full_oof_audit.json \
  --report workdirs/comind_optimization_v2/full_oof_audit.md
"$PYTHON_BIN" experiments/comind_optimization_v2/optimize_weak_fields_oof.py \
  --routed workdirs/comind_optimization_v2/full_train_oof_predictions.jsonl \
  --manifest workdirs/comind_optimization_v2/gpu_grounding_full_manifest.jsonl \
  --grounding workdirs/comind_optimization_v2/two_stage_grounding_full_predictions.jsonl \
  --output workdirs/comind_optimization_v2/weak_fields_nested_oof.json \
  --predictions workdirs/comind_optimization_v2/weak_fields_nested_predictions.jsonl \
  --report workdirs/comind_optimization_v2/weak_fields_nested_oof.md

echo "[MA-EgoQA] validate the frozen five-condition protocol and media manifest"
"$MA_PYTHON" MA-EgoQA/egomas/src/evaluate_video_component_ablation_jdcloud.py --dry-run

echo "[tests] validate registered ablations and CoMind/EnactToM logic"
"$PYTHON_BIN" -m pytest -q \
  experiments/iclr_ablations_v1/test_artifacts.py \
  experiments/comind_optimization_v2/test_comind_v2.py \
  experiments/comind_optimization_v2/test_two_stage_grounding.py \
  experiments/comind_optimization_v2/test_evaluate_gpu_grounding.py \
  tests/test_comind_optimization.py \
  tests/test_enacttom.py

date -u '+completed=%Y-%m-%dT%H:%M:%SZ' >> "$STATUS"
touch "$OUT/LOCAL_REFRESH_COMPLETE"
