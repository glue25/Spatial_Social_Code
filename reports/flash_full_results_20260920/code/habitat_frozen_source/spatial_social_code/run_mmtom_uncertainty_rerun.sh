#!/usr/bin/env bash
set -euo pipefail

cd /pfs/zhangxiaoyuan/project/physical_code

python_path="/pfs/zhangxiaoyuan/project/physical_code/cap-x/.venv-libero/bin/python"
output_path="spatial_social_code/mmtom_uncertainty_role_rerun_gemini3pro.jsonl"
summary_path="spatial_social_code/mmtom_uncertainty_role_rerun_gemini3pro_summary.json"

while true; do
  "${python_path}" -u spatial_social_code/refine_mmtom_uncertain_jdcloud.py \
    --workers 16 \
    --output "${output_path}"
  if "${python_path}" -c \
    'import json,sys; s=json.load(open(sys.argv[1])); raise SystemExit(0 if s.get("successful")==600 else 1)' \
    "${summary_path}"; then
    break
  fi
  sleep 30
done
