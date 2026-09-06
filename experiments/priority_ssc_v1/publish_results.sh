#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
GIT_DIR="$ROOT/.spatial_social_public_git"
OUT="$ROOT/workdirs/priority_ssc_v1"
mkdir -p "$ROOT/reports"

for source in \
  "$OUT/mmtom_local-qwen32b_reasoning_summary.json" \
  "$OUT/maegoqa_local-qwen32b_reasoning_summary.json" \
  "$OUT/mmtom_jdcloud_reasoning_summary.json" \
  "$OUT/maegoqa_jdcloud_reasoning_summary.json" \
  "$OUT/attribution_local-qwen32b_summary.json" \
  "$OUT/attribution_jdcloud_summary.json" \
  "$OUT/human_audit/audit_summary.json" \
  "$OUT/mmtom_local-qwen32b_four_cell_summary.json" \
  "$OUT/maegoqa_local-qwen32b_four_cell_summary.json"; do
  if [[ -f "$source" ]]; then
    cp "$source" "$ROOT/reports/priority_ssc_$(basename "$source")"
  fi
done

git --git-dir="$GIT_DIR" --work-tree="$ROOT" add experiments/priority_ssc_v1
git --git-dir="$GIT_DIR" --work-tree="$ROOT" add -f reports/priority_ssc_*_summary.json 2>/dev/null || true
if ! git --git-dir="$GIT_DIR" --work-tree="$ROOT" diff --cached --quiet; then
  git --git-dir="$GIT_DIR" --work-tree="$ROOT" commit -m "Add priority SSC controls and verified-state experiments"
fi
git --git-dir="$GIT_DIR" --work-tree="$ROOT" push origin main
