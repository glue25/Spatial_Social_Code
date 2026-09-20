#!/usr/bin/env bash
set -euo pipefail

ROOT="/pfs/zhangxiaoyuan/project/physical_code"
PY="$ROOT/.venv-habitat/bin/python"
TASKS="$ROOT/reports/habitat_generated_environment_cases/tasks"
RUN_ROOT="$ROOT/workdirs/habitat_closed_loop/generated_environment_cases"
REPORT="$ROOT/reports/habitat_generated_environment_cases"
LOG="$RUN_ROOT/pipeline.log"
STATUS="$RUN_ROOT/pipeline.status"

cd "$ROOT"
mkdir -p "$RUN_ROOT" "$REPORT"
printf 'RUNNING %s\n' "$(date -u +%FT%TZ)" >"$STATUS"
trap 'rc=$?; if [[ $rc -eq 0 ]]; then printf "COMPLETE %s\n" "$(date -u +%FT%TZ)" >"$STATUS"; else printf "FAILED rc=%s %s\n" "$rc" "$(date -u +%FT%TZ)" >"$STATUS"; fi' EXIT

{
  date -u
  "$PY" experiments/habitat_closed_loop/build_environment_cases.py --output "$TASKS"

  # The first pair is the smoke gate. Resume mode then preserves it while
  # completing the remaining six rollouts.
  "$PY" experiments/habitat_closed_loop/run_environment_cases.py \
    --tasks "$TASKS" --output "$RUN_ROOT" --case fact_owner_agent1 --resume

  while IFS= read -r artifact; do
    "$PY" experiments/habitat_closed_loop/validate_run.py "$artifact"
  done < <(find "$RUN_ROOT/results" -path '*/closed_loop_artifacts' -type d | sort)

  "$PY" experiments/habitat_closed_loop/run_environment_cases.py \
    --tasks "$TASKS" --output "$RUN_ROOT" --resume

  while IFS= read -r artifact; do
    "$PY" experiments/habitat_closed_loop/validate_run.py "$artifact"
  done < <(find "$RUN_ROOT/results" -path '*/closed_loop_artifacts' -type d | sort)

  "$PY" experiments/habitat_closed_loop/analyze_environment_cases.py \
    "$RUN_ROOT" --tasks "$TASKS" --output "$REPORT"
  "$PY" experiments/build_complete_supplement.py

  git --git-dir="$ROOT/.spatial_social_public_git" --work-tree="$ROOT" add -- \
    experiments/habitat_closed_loop/build_environment_cases.py \
    experiments/habitat_closed_loop/run_environment_cases.py \
    experiments/habitat_closed_loop/analyze_environment_cases.py \
    experiments/habitat_closed_loop/run_environment_cases_background.sh \
    experiments/build_complete_supplement.py \
    reports/habitat_generated_environment_cases \
    reports/supplement_complete_zh.md
  git --git-dir="$ROOT/.spatial_social_public_git" --work-tree="$ROOT" add -f -- \
    reports/supplement_complete_manifest.json
  if ! git --git-dir="$ROOT/.spatial_social_public_git" --work-tree="$ROOT" diff --cached --quiet; then
    git --git-dir="$ROOT/.spatial_social_public_git" --work-tree="$ROOT" commit \
      -m "Add executable SSC environment interventions and supplement"
  fi
  git --git-dir="$ROOT/.spatial_social_public_git" --work-tree="$ROOT" push origin main
  date -u
  echo PIPELINE_COMPLETE
} >>"$LOG" 2>&1
