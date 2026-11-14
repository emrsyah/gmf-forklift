# Training YOLO kabin di RunPod (L5)

Membangun dataset kabin dari **dataset publik saja** (tanpa token) lalu melatih YOLO11n sesuai Tech Spec bag. 5.
Hasil: `models/weights/cabin_best.pt`, `models/weights/cabin_best.onnx`, dan `training/runs/cabin_v1/gate_report.md`.

> Gate report dihitung pada **split test data publik**, bukan gerbang L5 (orang di luar tim, kamera kabin).
> Jangan klaim L5 lulus dari angka ini.

## 1. Buat pod

- Template: **RunPod PyTorch** (mis. `runpod/pytorch:2.4.0-py3.11-cuda12.4.1-devel-ubuntu22.04`).
- GPU: 1x RTX 4090 / A5000 / L4 (24 GB cukup untuk YOLO11n, batch 32, imgsz 640).
- Volume disk: **50 GB** di `/workspace` (data mentah + dataset NIR + run ~10 GB).
- Aktifkan **SSH over exposed TCP** dan pasang public key Anda. Catat `IP` dan `PORT` dari tab *Connect*.

Perkiraan waktu: build dataset 45-90 menit (unduhan per file dari HuggingFace + COCO, pose-fill di GPU),
training 150 epoch ~1,5-3 jam di RTX 4090 (berhenti lebih awal jika `patience=30` tercapai).

## 2. Salin repo ke pod (dari laptop, Git Bash)

Jalankan di root repo laptop. `data/`, `.venv/`, dan `training/runs/` tidak ikut.

```bash
IP=1.2.3.4; PORT=12345            # ganti dari tab Connect RunPod
tar czf - --exclude=./.venv --exclude=./data --exclude=./training/runs --exclude=./.git \
    --exclude='*.pyc' --exclude=./models/mediapipe . \
  | ssh -p $PORT root@$IP "mkdir -p /workspace/gmf-project && tar xzf - -C /workspace/gmf-project"
```

(Alternatif jika repo sudah di-push: `git clone <url> /workspace/gmf-project` di pod.)

## 3. Setup + training (di pod)

```bash
ssh -p $PORT root@$IP
cd /workspace/gmf-project
bash training/runpod/setup.sh

# uji cepat dulu (~5 menit): 50 gambar/sumber, 2 epoch
LIMIT=50 EPOCHS=2 NAME=runpod_smoke bash training/runpod/run_train.sh

# training penuh di tmux (aman jika SSH putus; keluar tmux: Ctrl-b lalu d; masuk lagi: tmux attach -t train)
tmux new -s train 'bash training/runpod/run_train.sh; bash'
```

Perintah Ultralytics yang dijalankan `train_cabin.py` (dari `training/configs/train_cabin.yaml`):

```
yolo detect train model=models/weights/yolo11n.pt data=training/datasets/cabin.yaml imgsz=640 epochs=150 patience=30 \
  batch=32 hsv_h=0 hsv_s=0 hsv_v=0.4 degrees=10 translate=0.1 scale=0.4 fliplr=0.5 mosaic=1.0 close_mosaic=15 seed=0
```

Opsi:
- `CACHE=ram` mempercepat epoch jika RAM pod cukup (dataset ~10-15 ribu gambar).
- `SKIP_BUILD=1` melatih ulang tanpa membangun dataset lagi.
- `HF_TOKEN=hf_xxx` (opsional) menaikkan rate limit HuggingFace jika unduhan sering kena HTTP 429. Tidak wajib.
- `ROBOFLOW_API_KEY=...` (opsional) mengaktifkan sumber Roboflow yang sudah diisi di `training/datasets/sources.yaml`.

## 4. Ambil hasil ke laptop (dari laptop, root repo)

```bash
ssh -p $PORT root@$IP "cd /workspace/gmf-project && tar czf - models/weights/cabin_best.pt models/weights/cabin_best.onnx \
    training/runs/cabin_v1 training/runs/logs data/cabin/cabin_v1/dataset_info.yaml data/cabin/cabin_v1/splits.csv" \
  | tar xzf - -C .
cat training/runs/cabin_v1/gate_report.md
```

Runtime laptop sudah menunjuk `models/weights/cabin_best.onnx` (`configs/runtime.yaml` -> `cabin.yolo.weights`).
Setelah itu **hentikan pod** (Stop/Terminate) agar tidak terus ditagih.

## 5. Jika gagal

| Gejala | Penyebab / solusi |
|---|---|
| `HTTP 429` saat download | Rate limit HF anonim. Skrip menunggu `Retry-After` dan mengulang; atau set `HF_TOKEN`. |
| `libGL.so.1` tidak ada | `apt-get install -y libgl1 libglib2.0-0` (setup.sh sudah mencoba). |
| CUDA OOM | `BATCH=16`. |
| DataLoader macet | `WORKERS=4`. |
| Disk penuh | Hapus `data/raw` setelah build (`data/cabin/cabin_v1` yang dipakai training). |
