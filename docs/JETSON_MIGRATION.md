# Pindah ke Jetson Orin Nano (Fase 2)

Kode dibuat agar pindah ke Jetson cukup dengan **mengganti config dan backend**, lalu **kalibrasi ulang threshold**. Daftar ini mengikuti CLAUDE.md bagian 9.

## 0. Quick-start Linux / Jetson + webcam USB (belum diverifikasi di perangkat)

```bash
# JetPack 6 sudah membawa Python 3.10, CUDA, TensorRT, dan OpenCV dengan GStreamer.
python3 -m venv --system-site-packages .venv && . .venv/bin/activate   # pakai OpenCV/TensorRT sistem
pip install mediapipe ultralytics onnxruntime paho-mqtt pyyaml lap pytest   # opencv-python JANGAN dipasang: pakai OpenCV sistem (GStreamer)
# torch untuk Jetson: wheel dari NVIDIA / image Ultralytics JetPack 6 (bukan dari PyPI)
# MediaPipe di aarch64: BELUM DIVERIFIKASI ada wheel untuk versi ini -> cek `python -c "import mediapipe"` dulu.
python -m src.main --source webcam --no-show                 # webcam USB, /dev/video0, tanpa layar
python -m src.main --source webcam:1                          # index lain
# atau set cabin.source.type: gstreamer + pipeline v4l2src (contoh di bawah)
```

Di Linux beep laptop memakai bel terminal; di Jetson set `alert.type: gpio`.

## 1. Yang diganti lewat config (tanpa ubah kode)

| Hal | Laptop (sekarang) | Jetson | Di mana |
|---|---|---|---|
| Sumber kamera kabin | `cabin.source.type: webcam` | `gstreamer` + `pipeline:` (contoh di bawah) | `configs/runtime.yaml` |
| Backend YOLO kabin | `onnxruntime` + `cabin_best.onnx` | `tensorrt` + `models/engines/cabin_best.engine` | `cabin.yolo.backend/weights` |
| Backend YOLO depan | `ultralytics` + `yolo11n.pt` | `tensorrt` + `front_best.engine` | `front.yolo` |
| Alert | `alert.type: laptop_beep` | `gpio` + `gpio_pin` | `alert` |
| Event | file JSONL | `sink.mqtt.enabled: true`, host/topik Minerva | `sink.mqtt` |
| Simulasi NIR | `simulate_nir: false/true` | **true** (frame NIR mono → 3 channel + CLAHE, sama seperti training) | `cabin.simulate_nir` |
| ROI kursi pengemudi | `null` | isi `[x, y, w, h]` relatif agar penumpang tidak terbaca | `cabin.roi` |
| ID perangkat | `laptop-dev` | `forklift-01` | `device_id`, `event_id_prefix` |

Contoh pipeline GStreamer (cek nama sensor/mode di board):

```yaml
# CSI (NoIR):
pipeline: "nvarguscamerasrc sensor-id=0 ! video/x-raw(memory:NVMM),width=1280,height=720,framerate=30/1 ! nvvidconv ! video/x-raw,format=BGRx ! videoconvert ! video/x-raw,format=BGR ! appsink drop=1 max-buffers=1"
# USB:
pipeline: "v4l2src device=/dev/video0 ! video/x-raw,width=1280,height=720,framerate=30/1 ! videoconvert ! video/x-raw,format=BGR ! appsink drop=1 max-buffers=1"
```

OpenCV di Jetson harus dibuild dengan GStreamer (OpenCV bawaan JetPack sudah).

## 2. Build engine TensorRT (DI Jetson, sekali per versi JetPack/model)

```bash
sudo nvpmodel -q && sudo nvpmodel -m <mode-max> && sudo jetson_clocks   # catat mode daya di logbook
python -m tools.export_onnx models/weights/cabin_best.pt --format engine --half
python -m tools.export_onnx models/weights/front_best.pt --format engine --half
# simpan ke models/engines/, nama berisi versi JetPack + TensorRT
```

Engine **tidak portabel** antar versi JetPack/GPU — jangan salin dari mesin lain. Cek kesamaan hasil dengan `pytest tests/test_backend_parity.py` (tambahkan pasangan .pt/.engine jika perlu).

## 3. Yang WAJIB dikalibrasi ulang (rekaman gladi di forklift)

1. **Tanda pose kepala**: `python -m tools.check_pose` — kamera kabin dipasang sedikit di bawah mata dan menghadap ke atas, jadi pitch nol berbeda; kalibrasi 5 s menanganinya, tapi cek tanda `pitch_sign`.
2. **EAR / openness**: lampu IR, kacamata, dan bibir helm mengubah EAR. Ukur `population_ear_baseline` dari rekaman peraga; cek `closed_openness_max`, `closed_blink_min`, `closed_logic` (blendshape `eyeBlink*` di gambar NIR mungkin kurang andal → coba `openness_only`).
3. **eye_valid**: `eye_valid_pitch_down_max_deg` harus dipilih agar skenario 7 (pejam sambil sedikit menunduk) tetap D-02 dan skenario 9 (lihat pangkuan) tidak jadi D-02.
4. **D-09 kamera terhalang**: dengan lampu IR telapak tangan bisa membuat gambar terang dan rata. Ukur `std`, `lap_var`, `mean` saat lensa ditutup dan saat kabin gelap normal (overlay menampilkan ketiganya) → set `D09_camera_block`.
5. **D-04 / D-05**: sudut kamera berubah → `nod_drop_deg`, `pitch_down_deg`, `yaw_deg`. Lihat risiko K4 di `STATUS.md` (opsi `pitch_only_extra_hold_s`, `nod_count`).
6. **YOLO kabin**: fine-tune dengan frame NIR dari kamera kabin asli (rompi tanpa warna, strip reflektif).
7. **Kamera depan**: `python tools/calibrate_camera.py` (papan catur) → K/dist, ukur tinggi pasang `h_m` dan sudut `theta_deg`, uji di lantai bertanda 2–10 m (`docs/FRONT_CAMERA.md`).
8. **FPS MediaPipe di CPU Orin**: target ≥ 20 FPS. Di laptop ini ~9 ms/frame. Jika kurang di Jetson → MtDMS TensorRT (interface sinyal tetap, lihat `src/cabin/face.py:make_face_model`).
9. **Klip**: Orin Nano tidak punya hardware encoder → x264 software `ultrafast` di thread prioritas rendah (sudah begitu). Ukur dampaknya ke FPS.
10. **Waktu**: chrony ke NTP yang sama dengan Minerva/command center (`ts_event_ms` dipakai untuk K5).

Setelah semua di atas: jalankan replay harness pada rekaman gladi di forklift dan tempel laporannya sebelum mengklaim lulus.
