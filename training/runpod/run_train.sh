#!/usr/bin/env bash
# Build dataset publik penuh + training YOLO11n kabin + export ONNX + gate report.
# Jalankan di dalam tmux supaya tidak putus saat SSH terputus:
#   tmux new -s train 'bash training/runpod/run_train.sh'
#
# Variabel opsional (default = resep Tech Spec bag. 5):
#   LIMIT=50        uji cepat (maks gambar per sumber); kosong = build penuh
#   SKIP_BUILD=1    pakai dataset yang sudah dibangun (training/datasets/cabin.yaml)
#   EPOCHS=150 BATCH=32 DEVICE=0 WORKERS=8 NAME=cabin_v1 CACHE=ram
#   WEIGHTS_NAME=cabin_best   (default cabin_smoke jika LIMIT di-set)
# Unduhan bisa dilanjutkan: jalankan ulang skrip; file yang sudah ada tidak diunduh lagi.
set -euo pipefail
cd "$(dirname "$0")/../.."
# shellcheck disable=SC1091
source .venv-train/bin/activate

LIMIT=${LIMIT:-}
EPOCHS=${EPOCHS:-150}
BATCH=${BATCH:-32}
DEVICE=${DEVICE:-0}
WORKERS=${WORKERS:-8}
NAME=${NAME:-cabin_v1}
CACHE=${CACHE:-}
# uji cepat (LIMIT di-set) TIDAK boleh menimpa models/weights/cabin_best.* yang dipakai runtime
if [ -n "$LIMIT" ]; then WEIGHTS_NAME=${WEIGHTS_NAME:-cabin_smoke}; else WEIGHTS_NAME=${WEIGHTS_NAME:-cabin_best}; fi
LOGDIR=training/runs/logs
mkdir -p "$LOGDIR"
STAMP=$(date +%Y%m%d-%H%M%S)

if [ -z "${SKIP_BUILD:-}" ]; then
  python training/scripts/build_cabin_dataset.py ${LIMIT:+--limit "$LIMIT"} 2>&1 | tee "$LOGDIR/build-$STAMP.log"
fi

python training/train_cabin.py --device "$DEVICE" --epochs "$EPOCHS" --batch "$BATCH" \
  --workers "$WORKERS" --name "$NAME" --exist-ok --weights-name "$WEIGHTS_NAME" ${CACHE:+--cache "$CACHE"} \
  2>&1 | tee "$LOGDIR/train-$STAMP.log"

echo
echo "Selesai. Hasil:"
ls -la "models/weights/$WEIGHTS_NAME.pt" "models/weights/$WEIGHTS_NAME.onnx"
echo "Gate report: training/runs/$NAME/gate_report.md"
