#!/usr/bin/env bash
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SOURCE_TASKS="$ROOT/workdirs/habitat_enacttom/frozen12/tasks"
OUT="$ROOT/workdirs/habitat_enacttom/frozen12/benchmark_qwen8b_spatial_guard_v2_seed3"
STATUS="$OUT/status.jsonl"
CONDITIONS=(physical_only physical_social)

mkdir -p "$OUT/tasks" "$OUT/results"
touch "$STATUS"
until curl -fsS http://127.0.0.1:8891/health >/dev/null; do
  sleep 15
done

for condition in "${CONDITIONS[@]}"; do
  "$ROOT/.venv-habitat/bin/python" \
    "$ROOT/experiments/habitat_enacttom/build_task_conditioned_code.py" \
    --input-dir "$SOURCE_TASKS" \
    --output-dir "$OUT/tasks/$condition" \
    --condition "$condition" \
    --spatial-version v2
done

cd "$ROOT/external/EnactTom" || exit 1
export ENACTTOM_PYTHON="$ROOT/.venv-habitat/bin/python"
export ENACTTOM_EPISODES_PATH="$ROOT/workdirs/habitat_enacttom/frozen12/episodes.json.gz"
export ENACTTOM_BENCHMARK_SEED=891003
export OPENAI_API_KEY="local-qwen"
export OPENAI_BASE_URL="http://127.0.0.1:8891/v1"

mapfile -t TASK_NAMES < <(find "$SOURCE_TASKS" -maxdepth 1 -type f -name '*.json' -printf '%f\n' | sort)
for task_name in "${TASK_NAMES[@]}"; do
  stem="${task_name%.json}"
  for condition in "${CONDITIONS[@]}"; do
    task_path="$OUT/tasks/$condition/$task_name"
    output_dir="$OUT/results/$condition/$stem"
    summary="$output_dir/results/benchmark_summary.json"
    if [[ -s "$summary" ]]; then
      printf '{"seed":891003,"condition":"%s","task":"%s","skipped_existing":true}\n' \
        "$condition" "$stem" >> "$STATUS"
      continue
    fi
    mkdir -p "$output_dir"
    started="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
    if ./enacttom/run.sh benchmark \
        --task "$task_path" \
        --output-dir "$output_dir" \
        --model qwen3-vl-8b-instruct \
        --llm openai_chat \
        --num-times 1 \
        --max-llm-calls 20 \
        --benchmark-run-mode standard \
        --observation-mode text \
        --no-calibration; then
      process_success=true
    else
      process_success=false
    fi
    printf '{"seed":891003,"condition":"%s","task":"%s","process_success":%s,"started":"%s","finished":"%s"}\n' \
      "$condition" "$stem" "$process_success" "$started" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$STATUS"
  done
done
