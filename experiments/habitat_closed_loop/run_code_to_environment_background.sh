#!/usr/bin/env bash
set -euo pipefail

ROOT="/pfs/zhangxiaoyuan/project/physical_code"
PY="$ROOT/.venv-habitat/bin/python"
BUNDLE="$ROOT/reports/code_to_environment/habitat_ep1166_generated_v1"
RUNS="$ROOT/workdirs/habitat_closed_loop/code_generated_full"
LOG="$RUNS/pipeline.log"
STATUS="$RUNS/pipeline.status"

cd "$ROOT"
mkdir -p "$RUNS"
printf 'RUNNING %s\n' "$(date -u +%FT%TZ)" >"$STATUS"
trap 'rc=$?; if [[ $rc -eq 0 ]]; then printf "COMPLETE %s\n" "$(date -u +%FT%TZ)" >"$STATUS"; else printf "FAILED rc=%s %s\n" "$rc" "$(date -u +%FT%TZ)" >"$STATUS"; fi' EXIT

{
  date -u
  "$PY" experiments/habitat_closed_loop/compile_ssc_environment.py
  "$PY" experiments/habitat_closed_loop/instantiate_ssc_environment.py
  "$PY" experiments/habitat_closed_loop/run_code_generated_environment.py \
    --bundle "$BUNDLE" --output "$RUNS" --resume

  while IFS= read -r artifact; do
    "$PY" experiments/habitat_closed_loop/validate_run.py "$artifact"
  done < <(find "$RUNS/results" -path '*/closed_loop_artifacts' -type d | sort)

  "$PY" experiments/habitat_closed_loop/analyze_code_generated_environment.py \
    --bundle "$BUNDLE" --runs "$RUNS" --output "$BUNDLE"
  "$PY" -m pytest -q \
    experiments/habitat_closed_loop/test_code_to_environment.py \
    experiments/habitat_closed_loop/test_closed_loop.py

  git --git-dir="$ROOT/.spatial_social_public_git" --work-tree="$ROOT" add -- \
    experiments/habitat_closed_loop/compile_ssc_environment.py \
    experiments/habitat_closed_loop/instantiate_ssc_environment.py \
    experiments/habitat_closed_loop/run_code_generated_environment.py \
    experiments/habitat_closed_loop/analyze_code_generated_environment.py \
    experiments/habitat_closed_loop/test_code_to_environment.py \
    experiments/habitat_closed_loop/run_code_to_environment_background.sh \
    reports/cases/habitat_ep1166_ssc_generated_environment.json \
    reports/code_to_environment \
    reports/habitat_ssc_executable_pipeline_zh.md
  if ! git --git-dir="$ROOT/.spatial_social_public_git" --work-tree="$ROOT" diff --cached --quiet; then
    git --git-dir="$ROOT/.spatial_social_public_git" --work-tree="$ROOT" commit \
      -m "Compile Spatial-Social Code into executable Habitat environments"
  fi
  git --git-dir="$ROOT/.spatial_social_public_git" --work-tree="$ROOT" push origin main
  date -u
  echo PIPELINE_COMPLETE
} >>"$LOG" 2>&1
