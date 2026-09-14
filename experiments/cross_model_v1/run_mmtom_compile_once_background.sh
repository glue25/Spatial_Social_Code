#!/usr/bin/env bash
set -euo pipefail
cd "$(dirname "$0")/../.."
mkdir -p workdirs/cross_model_v1
python3 experiments/cross_model_v1/run_mmtom_gpt56_compile_once.py \
  --groups 20 \
  --model gpt-5.6-sol \
  --reasoning-effort medium \
  > workdirs/cross_model_v1/mmtom_gpt56_compile_once20.log 2>&1
