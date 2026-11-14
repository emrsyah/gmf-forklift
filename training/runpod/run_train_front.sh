#!/usr/bin/env bash
# Build dataset publik kamera DEPAN + training YOLO11n depan + export ONNX + laporan per kelas.
# Prasyarat: bash training/runpod/setup.sh (venv .venv-train sama dengan kabin).
# Jalankan di tmux:
#   tmux new -s front 'bash training/runpod/run_train_front.sh'
#
# Variabel opsional (default = resep Tech Spec bag. 6):
#   LIMIT=50        uji cepat (maks gambar per sumber); kosong = build penuh (max_images di sources_front.yaml)
#   ONLY="coco_person loco"   hanya sumber tertentu
#   SKIP_BUILD=1    pakai dataset yang sudah ada (training/datasets/front.yaml)
#   EPOCHS=150 BATCH=32 DEVICE=0 WORKERS=8 NAME=front_v1 CACHE=ram IMGSZ=640
#   WEIGHTS_NAME=front_best   (default front_smoke jika LIMIT di-set)
# Apron Dataset (registrasi) & data sintetis: salin dulu ke data/manual/apron, data/manual/synthetic
# (lihat sources_front.yaml -> manual). Jika folder tidak ada, sumber itu dilewati.
set -euo pipefail
cd "$(dirname "$0")/../.."
# shellcheck disable=SC1091
source .venv-train/bin/activate

LIMIT=${LIMIT:-}
ONLY=${ONLY:-}
EPOCHS=${EPOCHS:-150}
BATCH=${BATCH:-32}
DEVICE=${DEVICE:-0}
WORKERS=${WORKERS:-8}
IMGSZ=${IMGSZ:-640}
NAME=${NAME:-front_v1}
CACHE=${CACHE:-}
# uji cepat tidak boleh menimpa models/weights/front_best.*
if [ -n "$LIMIT" ]; then WEIGHTS_NAME=${WEIGHTS_NAME:-front_smoke}; else WEIGHTS_NAME=${WEIGHTS_NAME:-front_best}; fi
LOGDIR=training/runs/logs
mkdir -p "$LOGDIR"
STAMP=$(date +%Y%m%d-%H%M%S)

if [ -z "${SKIP_BUILD:-}" ]; then
  # Build penuh: unduh zip LOCO sekali (769 MB) daripada ~5.000 permintaan HTTP Range (local_zip di sources_front.yaml)
  LOCO_ZIP=data/manual/loco/dataset.zip
  if [ -z "$LIMIT" ] && [ ! -s "$LOCO_ZIP" ]; then
    mkdir -p "$(dirname "$LOCO_ZIP")"
    curl -fL --retry 5 -o "$LOCO_ZIP.part" \
      'https://webdisk.ads.mwn.de/Handlers/AnonymousDownload.ashx?folder=73e976ba&path=LOCO%5Cv1%5Cdataset.zip' \
      && mv "$LOCO_ZIP.part" "$LOCO_ZIP"
  fi
  # shellcheck disable=SC2086
  python training/scripts/build_front_dataset.py ${LIMIT:+--limit "$LIMIT"} ${ONLY:+--only $ONLY} \
    2>&1 | tee "$LOGDIR/build-front-$STAMP.log"
fi

python training/train_front.py --device "$DEVICE" --epochs "$EPOCHS" --batch "$BATCH" --imgsz "$IMGSZ" \
  --workers "$WORKERS" --name "$NAME" --exist-ok --weights-name "$WEIGHTS_NAME" ${CACHE:+--cache "$CACHE"} \
  2>&1 | tee "$LOGDIR/train-front-$STAMP.log"

echo
echo "Selesai. Hasil:"
ls -la "models/weights/$WEIGHTS_NAME.pt" "models/weights/$WEIGHTS_NAME.onnx"
echo "Laporan: training/runs/$NAME/front_report.md"
