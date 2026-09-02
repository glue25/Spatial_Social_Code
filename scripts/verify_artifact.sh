#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")/.."

python -c "import spatial_social_code; from spatial_social_code.runtime import execute_spatial"
python -m pytest -q tests experiments/habitat_enacttom/test_task_conditioned_code.py
python spatial_social_code/test_spatial_social.py
python scripts/verify_reported_results.py
