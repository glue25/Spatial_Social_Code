#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_PREFIX="$ROOT/.venv-habitat"
DATA_ROOT="$ROOT/external/EnactTom/data"
LOG="$ROOT/workdirs/habitat_enacttom_assets.log"
STATUS="$ROOT/workdirs/habitat_enacttom_assets.status"

cd "$ROOT"
exec >"$LOG" 2>&1
trap 'code=$?; date -u "+stopped=%Y-%m-%dT%H:%M:%SZ" >>"$STATUS"; echo "exit_code=$code" >>"$STATUS"' EXIT
date -u '+started=%Y-%m-%dT%H:%M:%SZ' >"$STATUS"

"$ENV_PREFIX/bin/python" -c \
  'import habitat, habitat_sim, habitat_baselines, torch; print({"habitat_sim": habitat_sim.__version__, "torch": torch.__version__, "cuda": torch.cuda.is_available()})'
touch "$ROOT/workdirs/HABITAT_ENV_COMPLETE"

mkdir -p "$DATA_ROOT"
# Habitat-Sim's downloader shells out to git with huggingface.co URLs.  Keep
# the upstream downloader unchanged while routing those URLs through the
# reachable mirror for this process only.
export GIT_CONFIG_COUNT=1
export GIT_CONFIG_KEY_0=url.https://hf-mirror.com/.insteadOf
export GIT_CONFIG_VALUE_0=https://huggingface.co/
rmdir "$DATA_ROOT/versioned_data/replica_cad_dataset" 2>/dev/null || true
"$ENV_PREFIX/bin/python" -m habitat_sim.utils.datasets_download \
  --uids rearrange_task_assets hab_spot_arm hab3-episodes \
  --data-path "$DATA_ROOT/" \
  --no-replace \
  --no-prune

mkdir -p "$DATA_ROOT/datasets/enacttom_episodes/v0_0"
curl --http1.1 -fL --retry 20 --retry-delay 10 --connect-timeout 30 \
  -o "$DATA_ROOT/datasets/enacttom_episodes/v0_0/train_2k.json.gz" \
  "https://hf-mirror.com/datasets/ai-habitat/partnr_episodes/resolve/main/v0_0/train_2k.json.gz"
gzip -t "$DATA_ROOT/datasets/enacttom_episodes/v0_0/train_2k.json.gz"

date -u '+completed=%Y-%m-%dT%H:%M:%SZ' >>"$STATUS"
touch "$ROOT/workdirs/HABITAT_CORE_ASSETS_COMPLETE"
