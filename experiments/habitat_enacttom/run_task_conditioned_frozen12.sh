#!/usr/bin/env bash
set -uo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SOURCE_TASKS="$ROOT/workdirs/habitat_enacttom/frozen12/tasks"
OUT="$ROOT/workdirs/habitat_enacttom/frozen12/benchmark_qwen8b_task_conditioned_v2"
STATUS="$OUT/status.jsonl"
CONDITIONS=(physical_social physical_only social_only)

mkdir -p "$OUT"
touch "$STATUS"
until curl -fsS http://127.0.0.1:8891/health >/dev/null; do
  sleep 15
done

for condition in "${CONDITIONS[@]}"; do
  task_dir="$OUT/tasks/$condition"
  "$ROOT/.venv-habitat/bin/python" \
    "$ROOT/experiments/habitat_enacttom/build_task_conditioned_code.py" \
    --input-dir "$SOURCE_TASKS" \
    --output-dir "$task_dir" \
    --condition "$condition"
done

cd "$ROOT/external/EnactTom" || exit 1
export ENACTTOM_PYTHON="$ROOT/.venv-habitat/bin/python"
export ENACTTOM_EPISODES_PATH="$ROOT/workdirs/habitat_enacttom/frozen12/episodes.json.gz"
export OPENAI_API_KEY="local-qwen"
export OPENAI_BASE_URL="http://127.0.0.1:8891/v1"

for condition in "${CONDITIONS[@]}"; do
  task_dir="$OUT/tasks/$condition"
  while IFS= read -r task_path; do
    task_name="$(basename "$task_path")"
    stem="${task_name%.json}"
    output_dir="$OUT/results/$condition/$stem"
    summary="$output_dir/results/benchmark_summary.json"
    if [[ -s "$summary" ]]; then
      printf '{"condition":"%s","task":"%s","skipped_existing":true}\n' \
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
      success=true
    else
      success=false
    fi
    printf '{"condition":"%s","task":"%s","process_success":%s,"started":"%s","finished":"%s"}\n' \
      "$condition" "$stem" "$success" "$started" "$(date -u +%Y-%m-%dT%H:%M:%SZ)" >> "$STATUS"
  done < <(find "$task_dir" -maxdepth 1 -type f -name '*.json' ! -name manifest.json -print | sort)
done
