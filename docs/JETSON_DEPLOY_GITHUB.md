# Deploy ke Jetson dari GitHub (kamera kabin + depan)

Untuk anggota tim yang men-deploy langsung di Jetson tanpa laptop pengembang. Kode dan config ada di repo ini.
Bobot model **tidak** di-commit (aturan repo); semuanya ada di **GitHub Release** repo ini.

Jetson FL-01: Orin Nano, JetPack 6.2. Kabin = IMX219 NoIR CSI (`sensor-id=0`), depan = Logitech C930e USB.
Detektor dijalankan otomatis oleh `~/minerva/run_detector.sh` (crontab `@reboot`, milik tim MINERVA):
proses `python -m src.main --overlay configs/jetson.yaml` di `~/gmf-forklift`, dan di-restart 5 detik setelah keluar.
`~/gmf-forklift` di Jetson **bukan** git clone, tetapi salinan. Jangan menjalankan detektor kedua karena kamera
CSI hanya bisa dibuka satu proses.

## Isi release

| File | Tujuan di Jetson | Dipakai oleh |
|---|---|---|
| `cabin_best.onnx` | `models/weights/` | YOLO kabin: ponsel D-06, helm/rompi D-10 (cabin_v1) |
| `front_v2.onnx` | `models/weights/` | YOLO depan: person + rintangan + **kardus (`box`)**, F-01..F-03 |
| `front_best.onnx` | `models/weights/` | front_v1, cadangan jika front_v2 bermasalah |
| `face_detection_yunet_2023mar.onnx` | `models/face/` | detektor wajah YuNet (crop sebelum Face Landmarker) |
| `face_landmarker.task` | `models/mediapipe/` | MediaPipe Face Landmarker |

## Langkah di Jetson

```bash
# 1. Ambil kode (butuh akses ke repo privat: gh auth login, atau token HTTPS)
cd ~ && git clone https://github.com/emrsyah/gmf-forklift.git gmf-src   # sekali saja; berikutnya: cd ~/gmf-src && git pull

# 2. Cadangkan lalu salin kode ke folder yang dipakai detektor
cd ~/gmf-forklift && mkdir -p backups && tar -czf backups/pre-$(date +%Y%m%d-%H%M%S).tgz src configs tools
cp -r ~/gmf-src/src ~/gmf-src/configs ~/gmf-src/tools ~/gmf-forklift/
git -C ~/gmf-src rev-parse --short HEAD > ~/gmf-forklift/DEPLOYED_REV

# 3. Unduh model dari Release
cd ~/gmf-forklift && mkdir -p models/weights models/face models/mediapipe
gh release download models-2026-11-25 -R emrsyah/gmf-forklift -D /tmp/gmf-models --clobber
# tanpa gh: unduh file di halaman Releases lewat browser, lalu taruh di /tmp/gmf-models
cp /tmp/gmf-models/*.onnx models/weights/
mv models/weights/face_detection_yunet_2023mar.onnx models/face/
cp /tmp/gmf-models/face_landmarker.task models/mediapipe/

# 4. Build engine TensorRT FP16 DI Jetson (beberapa menit per model). Engine terikat versi TensorRT/GPU,
#    jadi jangan disalin dari mesin lain.
. .venv/bin/activate
for m in cabin_best front_v2; do
  /usr/src/tensorrt/bin/trtexec --onnx=models/weights/$m.onnx --saveEngine=models/weights/$m.trt --fp16
  python -m tools.wrap_trt_engine models/weights/$m.trt models/weights/$m.onnx models/weights/$m.engine
done

# 5. Cek import + config
python -c "from src.config import load_yaml; load_yaml('configs/jetson.yaml'); import cv2; cv2.FaceDetectorYN; print('ok')"

# 6. Restart detektor: hentikan dengan SIGINT (bukan kill -9); run_detector.sh menyalakannya lagi dengan kode baru
pkill -INT -f "src.main --overlay configs/jetson.yaml"
sleep 10 && tail -n 30 ~/minerva/logs/detector.log
```

Lihat hasilnya di browser: `http://<IP-Jetson>:8090` (kabin) dan `http://<IP-Jetson>:8090/stream/front` (depan).

## Setting kamera (semuanya di `configs/`)

- **`configs/jetson.yaml`**: overlay khusus Jetson.
  - Kabin: pipeline GStreamer IMX219, mode 1640x1232. Crop area pengemudi, auto-exposure hanya membaca area
    pengemudi, keluaran 960x720 GRAY8. YuNet crop menyala, YOLO kabin `cabin_best.engine`.
  - Depan: C930e dengan kontrol UVC (fokus tetap, 50 Hz) dan YOLO `front_v2.engine`.
  - Ada **opsi** mode sensor penuh 3280x2464 untuk wajah yang lebih besar. Opsi ini **belum diuji** di Jetson,
    jadi ukur FPS dulu sebelum dipakai.
- **`configs/front_camera.yaml`**: kelas rintangan, kelas bahaya F-03 (sekarang termasuk `box` = kardus), zona jarak.
- **`configs/thresholds.yaml`, `configs/events.yaml`**: ambang indikator D-01..D-10, kode/level/prioritas.

## Kembali ke versi sebelumnya

- Model depan lama: di `configs/jetson.yaml` set `front.yolo.weights: models/weights/front_best.engine`
  (build engine dari `front_best.onnx` dengan langkah 4).
- Kode lama: `cd ~/gmf-forklift && tar -xzf backups/pre-<waktu>.tgz`, lalu restart seperti langkah 6.

## Catatan

- Kamera kabin: fokuskan lensa dan arahkan ke wajah (wajah kira-kira 1/3 tinggi gambar, kamera tegak).
  Deteksi mata tertutup (D-01/D-02) gagal jika wajah kecil atau buram.
- Jangan mengubah layanan tim MINERVA (`minerva-modem.service`, `run_edge.sh`) tanpa koordinasi.
- `sudo` hanya jika memang perlu dan dengan izin pemilik perangkat.
