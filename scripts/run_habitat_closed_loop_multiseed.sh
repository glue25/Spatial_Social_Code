#!/usr/bin/env bash
set -euo pipefail

project_root="/pfs/zhangxiaoyuan/project/physical_code"
cd "$project_root"

seeds=(891012 891013)
master_log="$project_root/workdirs/habitat_closed_loop/multiseed_extension_status.log"

mkdir -p "$project_root/workdirs/habitat_closed_loop"

for seed in "${seeds[@]}"; do
  output="$project_root/workdirs/habitat_closed_loop/multiseed_seed${seed}"
  if [[ -f "$output/COMPLETE" ]]; then
    echo "seed=$seed status=already_complete" >> "$master_log"
    continue
  fi

  mkdir -p "$output"
  echo "seed=$seed status=running started_utc=$(date -u +%FT%TZ)" >> "$master_log"
  "$project_root/.venv-habitat/bin/python" \
    "$project_root/experiments/habitat_closed_loop/run_regression.py" \
    --mode dev36 \
    --seed "$seed" \
    --output "$output" \
    > "$output.driver.log" 2>&1

  accepted=0
  while IFS= read -r artifact_dir; do
    "$project_root/.venv-habitat/bin/python" \
      "$project_root/experiments/habitat_closed_loop/validate_run.py" \
      "$artifact_dir" >> "$output.validation.log"
    accepted=$((accepted + 1))
  done < <(find "$output/results" -type d -name closed_loop_artifacts | sort)

  if [[ "$accepted" -ne 36 ]]; then
    echo "seed=$seed status=validation_failed accepted=$accepted" >> "$master_log"
    exit 1
  fi

  "$project_root/.venv-habitat/bin/python" \
    "$project_root/experiments/habitat_closed_loop/analyze_dev36.py" \
    "$output" --output "$output/analysis" >> "$output.analysis.log" 2>&1

  touch "$output/COMPLETE"
  echo "seed=$seed status=complete accepted=36 finished_utc=$(date -u +%FT%TZ)" >> "$master_log"
done

echo "all_seeds_complete finished_utc=$(date -u +%FT%TZ)" >> "$master_log"
