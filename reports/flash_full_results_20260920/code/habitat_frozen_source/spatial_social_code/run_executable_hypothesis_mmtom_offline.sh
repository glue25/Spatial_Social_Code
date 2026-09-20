#!/usr/bin/env bash
set -euo pipefail

cd /pfs/zhangxiaoyuan/project/physical_code

output="spatial_social_code/mmtom_executable_hypothesis_offline_v2.jsonl"
log="spatial_social_code/mmtom_executable_hypothesis_offline_v2.log"
status="spatial_social_code/mmtom_executable_hypothesis_offline_v2.status"

echo RUNNING > "${status}"
while true; do
  if python -u spatial_social_code/evaluate_executable_hypothesis_offline.py \
    --benchmark mmtom \
    --source spatial_social_code/mmtom_dense_authoritative_gemini3pro.jsonl \
    --output "${output}" >> "${log}" 2>&1; then
    echo COMPLETE > "${status}"
    break
  fi
  echo "retrying resumable evaluation" >> "${log}"
  sleep 30
done
