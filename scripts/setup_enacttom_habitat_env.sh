#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
ENV_PREFIX="$ROOT/.venv-habitat"
LOG="$ROOT/workdirs/habitat_enacttom_env.log"
STATUS="$ROOT/workdirs/habitat_enacttom_env.status"
cd "$ROOT"
exec > "$LOG" 2>&1
trap 'code=$?; date -u "+stopped=%Y-%m-%dT%H:%M:%SZ" >> "$STATUS"; echo "exit_code=$code" >> "$STATUS"' EXIT
date -u '+started=%Y-%m-%dT%H:%M:%SZ' > "$STATUS"

conda install --solver classic -p "$ENV_PREFIX" \
  pytorch==2.4.1 torchvision==0.19.1 torchaudio==2.4.1 pytorch-cuda=12.4 \
  'mkl<2025' 'intel-openmp<2025' -c pytorch -c nvidia -y

conda install --solver classic -p "$ENV_PREFIX" \
  habitat-sim=0.3.3 withbullet headless -c conda-forge -c aihabitat -y

while [[ ! -f external/habitat-lab/PINNED_COMMIT ]]; do
  sleep 30
done

"$ENV_PREFIX/bin/python" -m pip install -e external/habitat-lab/habitat-lab
"$ENV_PREFIX/bin/python" -m pip install -e external/habitat-lab/habitat-baselines
"$ENV_PREFIX/bin/python" -m pip install \
  pillow==10.4.0 numpy-quaternion==2023.0.4 matplotlib==3.6.3 \
  opencv-python==4.10.0.82 openai==2.24.0 pandas pytest \
  unified-planning==1.3.0 up-fast-downward==0.5.2
"$ENV_PREFIX/bin/python" -m pip install -e external/EnactTom --no-deps

"$ENV_PREFIX/bin/python" -c 'import habitat, habitat_sim, habitat_baselines, torch; print({"habitat": habitat.__file__, "habitat_sim": habitat_sim.__file__, "torch": torch.__version__, "cuda": torch.cuda.is_available()})'
date -u '+completed=%Y-%m-%dT%H:%M:%SZ' >> "$STATUS"
touch workdirs/HABITAT_ENV_COMPLETE
