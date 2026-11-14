# Forklift Safety AI — UC2 GMF (Fase 1, laptop)

Konteks lengkap: `CLAUDE.md`. Status pekerjaan dan hal yang perlu dicek: `STATUS.md`.

## Setup (Windows, Python 3.10 = sama dengan JetPack 6)

```bash
uv venv .venv --python 3.10
uv pip install --python .venv/Scripts/python.exe --index-url https://download.pytorch.org/whl/cpu torch torchvision
uv pip install --python .venv/Scripts/python.exe -r requirements.txt
# model wajah MediaPipe
curl -L -o models/mediapipe/face_landmarker.task https://storage.googleapis.com/mediapipe-models/face_landmarker/face_landmarker/float16/latest/face_landmarker.task
```

## Menjalankan

```bash
.venv/Scripts/python -m src.main --source webcam --fps-only     # L1: kamera + FPS
.venv/Scripts/python -m src.main --source webcam                 # kabin lengkap (q keluar, c kalibrasi ulang)
.venv/Scripts/python -m src.main --source webcam --simulate-nir  # gambar mirip kamera NIR
.venv/Scripts/python -m src.main --source rtsp://IP-HP:8554/live # kamera HP (aplikasi IP camera)
.venv/Scripts/python -m tools.check_pose                         # cek tanda yaw/pitch/roll (30 detik)
.venv/Scripts/python tools/run_front.py --source webcam          # kamera depan: orang + tracking + jarak
```

Saat start, tatap lurus ke depan ~5 detik (kalibrasi). Event JSON, snapshot, dan klip MP4 tersimpan di `events/YYYY-MM-DD/`.
YOLO kabin (D-06 ponsel, D-10 APD) aktif jika `models/weights/cabin_best.onnx` ada (lihat `training/`).

## Rekam skenario dan evaluasi

```bash
.venv/Scripts/python -m tools.recorder --peraga P1            # hotkey [ ] pilih skenario, SPASI tandai
.venv/Scripts/python -m eval.replay_eval --truth eval/scenarios/truth.csv
.venv/Scripts/python -m eval.replay_eval --truth ... --set D05_distraction.pitch_only_extra_hold_s=1.0
```

## Tes

```bash
.venv/Scripts/python -m pytest
```

## Struktur

`configs/` (semua threshold, pemetaan kode, sumber kamera) · `src/capture` · `src/backends` (ultralytics / onnxruntime / tensorrt) ·
`src/cabin` (wajah, sinyal, kalibrasi, YOLO kabin, indikator D-xx) · `src/decision` · `src/io` (beep/GPIO, file/MQTT, klip) ·
`src/front` · `src/ui` · `tools/` · `training/` · `eval/` · `tests/` · `docs/` (spec + `JETSON_MIGRATION.md`, `FRONT_CAMERA.md`).
