#!/usr/bin/env bash
set -euo pipefail

project_root="/pfs/zhangxiaoyuan/project/physical_code"
runner_name="scripts/run_habitat_closed_loop_multiseed.sh"
status_log="$project_root/workdirs/habitat_closed_loop/multiseed_finalize_status.log"
seed_a="$project_root/workdirs/habitat_closed_loop/multiseed_seed891012"
seed_b="$project_root/workdirs/habitat_closed_loop/multiseed_seed891013"
result_dir="$project_root/reports/habitat_closed_loop_multiseed"

cd "$project_root"
mkdir -p "$result_dir"
echo "status=waiting started_utc=$(date -u +%FT%TZ)" >> "$status_log"

while [[ ! -f "$seed_a/COMPLETE" || ! -f "$seed_b/COMPLETE" ]]; do
  if ! pgrep -f "^bash $runner_name$" >/dev/null; then
    echo "status=blocked reason=runner_stopped_before_complete time=$(date -u +%FT%TZ)" >> "$status_log"
    exit 1
  fi
  sleep 30
done

"$project_root/.venv-habitat/bin/python" \
  "$project_root/experiments/habitat_closed_loop/analyze_multiseed.py" \
  "$project_root/workdirs/habitat_closed_loop/dev36_final_v3" \
  "$seed_a" "$seed_b" --output "$result_dir" \
  > "$result_dir.analysis.log" 2>&1

"$project_root/.venv-habitat/bin/python" \
  "$project_root/experiments/habitat_closed_loop/visualize_episode1166.py" \
  >> "$result_dir.analysis.log" 2>&1
"$project_root/.venv-habitat/bin/python" \
  "$project_root/experiments/habitat_closed_loop/compile_ssc_environment.py" \
  >> "$result_dir.analysis.log" 2>&1

git_args=(--git-dir="$project_root/.spatial_social_public_git" --work-tree="$project_root")
git "${git_args[@]}" add \
  experiments/habitat_closed_loop/analyze_multiseed.py \
  experiments/habitat_closed_loop/analyze_dev36.py \
  experiments/habitat_closed_loop/compile_ssc_environment.py \
  experiments/habitat_closed_loop/visualize_episode1166.py \
  scripts/run_habitat_closed_loop_multiseed.sh \
  scripts/finalize_habitat_closed_loop_multiseed.sh \
  reports/ssc_representation_planning_generation_zh.md \
  reports/cases/habitat_ep1166_dynamic_ssc_visualization.png \
  reports/cases/habitat_ep1166_dynamic_ssc_visualization.json \
  reports/cases/habitat_ep1166_ssc_generated_environment.json \
  reports/habitat_closed_loop_multiseed/habitat_multiseed_raw.csv \
  reports/habitat_closed_loop_multiseed/habitat_multiseed_summary.json \
  reports/habitat_closed_loop_multiseed/habitat_multiseed_results.md

if ! git "${git_args[@]}" diff --cached --quiet; then
  git "${git_args[@]}" commit -m "Add closed-loop SSC multi-seed evidence and environment compiler"
  git "${git_args[@]}" push origin main >> "$status_log" 2>&1
fi

echo "status=complete uploaded=true finished_utc=$(date -u +%FT%TZ)" >> "$status_log"
