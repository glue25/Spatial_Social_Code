#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_PREFIX="$ROOT/.venv-habitat"
DATA_ROOT="$ROOT/external/EnactTom/data"
LOG="$ROOT/workdirs/habitat_enacttom_paper_assets.log"
STATUS="$ROOT/workdirs/habitat_enacttom_paper_assets.status"

cd "$ROOT"
exec >"$LOG" 2>&1
trap 'code=$?; date -u "+stopped=%Y-%m-%dT%H:%M:%SZ" >>"$STATUS"; echo "exit_code=$code" >>"$STATUS"' EXIT
date -u '+started=%Y-%m-%dT%H:%M:%SZ' >"$STATUS"

export HF_ENDPOINT=https://hf-mirror.com
# The mirror's Xet batch endpoint currently returns a stale CDN hostname, while
# ordinary resolve downloads use the reachable official hf.co CDN.
export HF_HUB_DISABLE_XET=1
HF="$ENV_PREFIX/bin/hf"

mkdir -p "$DATA_ROOT/robots/hab_spot_arm"
"$HF" download ai-habitat/hab_spot_arm \
  --repo-type dataset --revision v2.0 \
  --local-dir "$DATA_ROOT/robots/hab_spot_arm"

mkdir -p "$DATA_ROOT/datasets/enacttom_episodes/v0_0"
"$HF" download ai-habitat/partnr_episodes v0_0/train_2k.json.gz \
  --repo-type dataset \
  --local-dir "$DATA_ROOT/datasets/enacttom_episodes"

mkdir -p "$DATA_ROOT/objects_ovmm"
"$HF" download ai-habitat/OVMM_objects \
  --repo-type dataset \
  --local-dir "$DATA_ROOT/objects_ovmm"

mkdir -p "$DATA_ROOT/versioned_data/hssd-hab"
"$HF" download hssd/hssd-hab \
  --repo-type dataset --revision partnr \
  --local-dir "$DATA_ROOT/versioned_data/hssd-hab"
ln -sfn versioned_data/hssd-hab "$DATA_ROOT/hssd-hab"

test -f "$DATA_ROOT/hssd-hab/hssd-hab-partnr.scene_dataset_config.json"
test -f "$DATA_ROOT/datasets/enacttom_episodes/v0_0/train_2k.json.gz"
test -d "$DATA_ROOT/objects_ovmm/train_val/hssd/configs/objects"
test -f "$DATA_ROOT/robots/hab_spot_arm/urdf/hab_spot_arm.urdf"
gzip -t "$DATA_ROOT/datasets/enacttom_episodes/v0_0/train_2k.json.gz"

date -u '+completed=%Y-%m-%dT%H:%M:%SZ' >>"$STATUS"
touch "$ROOT/workdirs/HABITAT_PAPER_ASSETS_COMPLETE"
