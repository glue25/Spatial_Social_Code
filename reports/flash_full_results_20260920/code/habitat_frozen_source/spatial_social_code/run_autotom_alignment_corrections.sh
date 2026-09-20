#!/usr/bin/env bash
set -Eeuo pipefail

ROOT="/pfs/zhangxiaoyuan/project/physical_code"
PYTHON="$ROOT/cap-x/.venv-libero/bin/python"
ALIGNED_CSV="$ROOT/spatial_social_code/mmtom_autotom_aligned.csv"

cd "$ROOT"
mapfile -t correction_indices < <(
    "$PYTHON" spatial_social_code/prepare_aligned_mmtom_autotom.py --print-indices
)
if [[ "${#correction_indices[@]}" -eq 0 ]]; then
    echo "No AutoToM candidate-alignment corrections are required."
    exit 0
fi

echo "Rerunning ${#correction_indices[@]} candidate-mismatched questions with an isolated cache prefix."
bash AutoToM/run_mmtom_autotom_local.sh \
    --mmtom_image_input \
    --mmtom_questions_path "$ROOT/MMToM-QA/Benchmark/questions.json" \
    --mmtom_video_path "$ROOT/MMToM-QA-data/required_frames" \
    --mmtom_num_frames 8 \
    --mmtom_csv_path "$ALIGNED_CSV" \
    --output_model_name "automated_qwen3vl_mm_required_aligned" \
    --indices "${correction_indices[@]}"
