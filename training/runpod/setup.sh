#!/usr/bin/env bash
# Setup lingkungan training kabin di RunPod (image PyTorch resmi, sudah ada torch + CUDA).
# Pakai: bash training/runpod/setup.sh   (dari mana saja; skrip pindah ke root repo)
set -euo pipefail
cd "$(dirname "$0")/../.."
echo "repo: $(pwd)"

# libGL untuk opencv-python (image PyTorch RunPod belum tentu punya)
if command -v apt-get >/dev/null 2>&1; then
  apt-get update -qq && apt-get install -y -qq libgl1 libglib2.0-0 tmux rsync >/dev/null || true
fi

# venv yang mewarisi torch CUDA dari image (tidak mengunduh ulang torch)
# VENV_DIR: taruh venv di disk lokal container jika /workspace adalah network FS yang lambat untuk
# file kecil (secure cloud: MooseFS); .venv-train lalu menjadi symlink ke sana.
VENV_DIR=${VENV_DIR:-.venv-train}
python3 -m venv --system-site-packages "$VENV_DIR"
if [ "$VENV_DIR" != ".venv-train" ]; then ln -sfn "$VENV_DIR" .venv-train; fi
# shellcheck disable=SC1091
source .venv-train/bin/activate
python -m pip install -q -U pip
python -m pip install -q -r training/requirements-train.txt
# sama dengan laptop: tanpa opencv-python-headless
python -m pip install -q --no-deps albumentations==2.0.8 albucore==0.0.24

python - <<'PY'
import torch, ultralytics, albumentations, cv2, onnxruntime
print("torch", torch.__version__, "cuda", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "-")
print("ultralytics", ultralytics.__version__, "albumentations", albumentations.__version__,
      "opencv", cv2.__version__, "onnxruntime", onnxruntime.__version__)
PY
echo "setup selesai. Lanjut: bash training/runpod/run_train.sh"
