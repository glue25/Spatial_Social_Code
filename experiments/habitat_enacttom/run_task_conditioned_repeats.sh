#!/usr/bin/env bash
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SOURCE_TASKS="$ROOT/workdirs/habitat_enacttom/frozen12/tasks"
OUT="$ROOT/workdirs/habitat_enacttom/frozen12/benchmark_qwen8b_task_conditioned_repeats"
STATUS="$OUT/status.jsonl"
CONDITIONS=(direct physical_only social_only physical_social)
REPEATS=(2 3)
SEEDS=(891002 891003)

mkdir -p "$OUT/tasks" "$OUT/results"
touch "$STATUS"
until curl -fsS http://127.0.0.1:8891/health >/dev/null; do
  sleep 15
done

for condition in physical_only social_only physical_social; do
  "$ROOT/.venv-habitat/bin/python" \
    "$ROOT/experiments/habitat_enacttom/build_task_conditioned_code.py" \
    --input-dir "$SOURCE_TASKS" \
    --output-dir "$OUT/tasks/$condition" \
    --condition "$condition"
done

cd "$ROOT/external/EnactTom" || exit 1
export ENACTTOM_PYTHON="$ROOT/.venv-habitat/bin/python"
export ENACTTOM_EPISODES_PATH="$ROOT/workdirs/habitat_enacttom/frozen12/episodes.json.gz"
export OPENAI_API_KEY="local-qwen"
export OPENAI_BASE_URL="http://127.0.0.1:8891/v1"

mapfile -t TASK_NAMES < <(find "$SOURCE_TASKS" -maxdepth 1 -type f -name '*.json' -printf '%f\n' | sort)
for repeat_offset in "${!REPEATS[@]}"; do
  repeat_index="${REPEATS[$repeat_offset]}"
  export ENACTTOM_BENCHMARK_SEED="${SEEDS[$repeat_offset]}"
  for task_name in "${TASK_NAMES[@]}"; do
    stem="${task_name%.json}"
    for condition in "${CONDITIONS[@]}"; do
      if [[ "$condition" == direct ]]; then
        task_path="$SOURCE_TASKS/$task_name"
      else
        task_path="$OUT/tasks/$condition/$task_name"
      fi
      output_dir="$OUT/results/repeat_$repeat_index/$condition/$stem"
      summary="$output_dir/results/benchmark_summary.json"
      if [[ -s "$summary" ]]; then
        printf '{"repeat":%d,"seed":%d,"condition":"%s","task":"%s","skipped_existing":true}\n' \
          "$repeat_index" "$ENACTTOM_BENCHMARK_SEED" "$condition" "$stem" >> "$STATUS"
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
      printf '{"repeat":%d,"seed":%d,"condition":"%s","task":"%s","process_success":%s,"started":"%s","finished":"%s"}\n' \
        "$repeat_index" "$ENACTTOM_BENCHMARK_SEED" "$condition" "$stem" \
        "$process_success" "$started" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$STATUS"
    done
  done
done
