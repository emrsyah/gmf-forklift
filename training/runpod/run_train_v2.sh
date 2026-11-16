#!/usr/bin/env bash
# cabin_v2 (2026-11-16): build dataset (sumber v1 + ponsel Open Images + negatif keras COCO), cek bahwa
# split test v1 terbentuk ulang, fine-tune dari cabin_v1, bandingkan v1 vs v2 pada split yang sama,
# lalu tulis tarball hasil + sha256 + penanda DONE. Laptop menarik hasil (scp) dan menghentikan pod.
#
# Prasyarat di pod: repo di /workspace/gmf-project (tanpa data/rekaman), models/weights/cabin_v1_ref.pt
# (= cabin_v1), bash training/runpod/setup.sh sudah jalan.
#
#   tmux new -d -s v2 'bash training/runpod/run_train_v2.sh 2>&1 | tee /workspace/run_v2.log'
#
# Variabel: LIMIT (smoke), EPOCHS=60 PATIENCE=20 BATCH=32 WORKERS=8 NAME=cabin_v2
#           DEADLINE_H=6   pod menghentikan dirinya sendiri setelah sekian jam apa pun yang terjadi
#           GRACE_MIN=45   setelah DONE, tunggu laptop menarik hasil sebelum pod stop sendiri
set -uo pipefail
cd "$(dirname "$0")/../.."
# shellcheck disable=SC1091
source .venv-train/bin/activate

LIMIT=${LIMIT:-}
EPOCHS=${EPOCHS:-60}
PATIENCE=${PATIENCE:-20}
BATCH=${BATCH:-32}
WORKERS=${WORKERS:-8}
NAME=${NAME:-cabin_v2}
DEADLINE_H=${DEADLINE_H:-6}
GRACE_MIN=${GRACE_MIN:-45}
OUT=/workspace/out_$NAME
mkdir -p "$OUT" training/runs/logs

# Sesi SSH/tmux tidak mewarisi env container; ambil ID pod + kredensial pod sendiri (disuntik RunPod,
# bukan kunci pengguna) dari proses 1. SELF_STOP=0 mematikan watchdog dan stop otomatis (uji asap).
if [ -r /proc/1/environ ]; then
  while IFS= read -r -d '' kv; do
    case "$kv" in RUNPOD_POD_ID=*|RUNPOD_API_KEY=*) export "${kv?}";; esac
  done < /proc/1/environ
fi
SELF_STOP=${SELF_STOP:-1}
self_stop() {
  # runpodctl tanpa config menolak kunci pod (Unauthorized); GraphQL podStop menerimanya (diuji 2026-11-16).
  # Stop (bukan terminate): volume /workspace dengan hasil tetap ada.
  if [ "$SELF_STOP" = "1" ] && [ -n "${RUNPOD_POD_ID:-}" ] && [ -n "${RUNPOD_API_KEY:-}" ]; then
    curl -s -H "Content-Type: application/json" -H "Authorization: Bearer $RUNPOD_API_KEY" \
      https://api.runpod.io/graphql \
      -d "{\"query\":\"mutation { podStop(input:{podId:\\\"$RUNPOD_POD_ID\\\"}) { id desiredStatus } }\"}"
  fi
}
# watchdog biaya: stop paksa setelah DEADLINE_H jam
( sleep $((DEADLINE_H * 3600)); echo "[watchdog] deadline ${DEADLINE_H} h -> stop pod"; self_stop ) &

stage() { echo "$1" > "$OUT/STAGE"; echo "== $(date +%H:%M:%S) $1"; }

stage build
python training/scripts/build_cabin_dataset.py ${LIMIT:+--limit "$LIMIT"} --version "$NAME" \
  2>&1 | tee training/runs/logs/build-$NAME.log
cp "data/cabin/$NAME/dataset_info.yaml" "data/cabin/$NAME/splits.csv" "$OUT/" 2>/dev/null

stage compare_v1
python training/scripts/compare_cabin.py --weights cabin_v1=models/weights/cabin_v1_ref.pt \
  --device 0 --out "$OUT/compare_v1_only" 2>&1 | tail -60

stage train
python training/train_cabin.py --model models/weights/cabin_v1_ref.pt --device 0 --epochs "$EPOCHS" \
  --patience "$PATIENCE" --batch "$BATCH" --workers "$WORKERS" --name "$NAME" --exist-ok \
  --weights-name "$NAME" 2>&1 | tee training/runs/logs/train-$NAME.log

stage compare
python training/scripts/compare_cabin.py --weights cabin_v1=models/weights/cabin_v1_ref.pt \
  "$NAME=training/runs/$NAME/weights/best.pt" --device 0 --out "$OUT/compare" 2>&1 | tail -80

stage pack
cp -r "training/runs/$NAME" "$OUT/run" 2>/dev/null
cp "models/weights/$NAME.pt" "models/weights/$NAME.onnx" "$OUT/" 2>/dev/null
cp training/runs/logs/*-"$NAME".log "$OUT/" 2>/dev/null
cp -r data/raw/openimages_phone/attribution.csv data/raw/*/source.yaml "$OUT/" 2>/dev/null
for s in data/raw/*/; do cp "$s/source.yaml" "$OUT/source_$(basename "$s").yaml" 2>/dev/null; done
tar czf "/workspace/results_$NAME.tgz" -C /workspace "out_$NAME"
sha256sum "/workspace/results_$NAME.tgz" | cut -d' ' -f1 > "/workspace/results_$NAME.sha256"
stage DONE
echo "hasil: /workspace/results_$NAME.tgz ($(cat /workspace/results_$NAME.sha256))"

# laptop biasanya sudah menarik dan menghentikan pod; jika tidak, stop sendiri setelah GRACE_MIN
sleep $((GRACE_MIN * 60))
echo "[grace] ${GRACE_MIN} min setelah DONE -> stop pod"
self_stop
