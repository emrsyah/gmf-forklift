# CLAUDE.md — Forklift Safety AI (UC2 GMF)

File ini adalah konteks utama untuk Claude Code di repo ini. Baca seluruhnya sebelum mengerjakan tugas apa pun.

---

## 1. Konteks proyek

Kita membangun **sistem AI keselamatan forklift** untuk proof of concept (PoC) di **GMF** (hangar perawatan pesawat). Sistem lengkapnya:

- **Edge:** NVIDIA Jetson Orin Nano di forklift uji, dengan dua kamera:
  - **Kamera kabin** (menghadap wajah pengemudi): kantuk, distraksi, ponsel, APD.
  - **Kamera depan**: orang dan rintangan di jalur forklift.
- **Posisi/zona:** BLE beacon + scanner (dikerjakan tim lain, bukan di repo ini).
- **Jaringan:** Ericsson Private 5G, modem Quectel RM520N-GL (hanya transport data).
- **Cloud:** platform Minerva (ingestion MQTT, alert management, penyimpanan klip, command center). Dikerjakan tim lain.

**Bagian kita = model AI + logika deteksi + event yang dikirim ke platform.**

Kelulusan PoC dinilai dengan **Logbook KPI UC2 — Forklift** dari GMF. Untuk AI, yang menentukan adalah **Lembar A** (20 skenario kondisi pengemudi). Dua KPI di antaranya **penggugur** (lihat bagian 4).

---

## 2. Fase kerja

### Fase 1 — SEKARANG: model di laptop

Jetson dan kamera final belum datang. Fokus fase ini: **bangun dan latih semua model semaksimal mungkin** sampai semua kapabilitas di bagian 4 ada dan teruji.

- **Kamera kabin:** webcam laptop (menghadap wajah).
- **Kamera depan:** webcam laptop atau video rekaman untuk sementara; nanti bisa dicoba kamera HP (stream RTSP/HTTP dari aplikasi IP camera ke OpenCV).
- **Inferensi:** PyTorch / ONNX Runtime di laptop (CPU atau GPU yang ada).
- **Alert:** suara beep di laptop + overlay di layar, plus event JSON ke file / MQTT lokal.

Disadari: webcam laptop berbeda jauh dari kamera NIR di kabin forklift (sudut, cahaya, IR, getaran). Tidak apa-apa untuk fase ini, **asalkan kode dibuat agar mudah dipindah dan dikalibrasi ulang**.

### Fase 2 — NANTI: Jetson Orin Nano

Setelah Jetson dan kamera datang:

- Ganti sumber kamera ke CSI/USB lewat GStreamer.
- Export model ke **TensorRT FP16** (engine di-build di Jetson itu sendiri).
- **Kalibrasi ulang** threshold dengan kamera dan posisi pasang yang sebenarnya.
- Integrasi buzzer GPIO dan MQTT ke Minerva.

**Aturan desain agar Fase 2 mudah:** sumber kamera, backend inferensi, dan output alert harus berupa **interface yang bisa diganti lewat config**, bukan ditulis langsung di logika.

---

## 3. Dokumen referensi

Simpan versi Markdown/PDF dari dokumen ini di folder `docs/` (export dari Claude Docs ke Markdown). Jika ada konflik, urutan prioritas: **Logbook KPI** > **Technical Spec** > **Spesifikasi & Strategi Data** > **Design Document**.

| File di `docs/` | Isi | Dipakai untuk |
|---|---|---|
| `Logbook_KPI_UC2_-_Forklift.xlsx` | KPI, ambang lulus, 20 skenario Lembar A, 10 skenario zona Lembar B, daftar periksa | **Sumber kebenaran kriteria lulus** |
| `technical_spec.md` — *Technical Spec — Pembangunan Model AI Forklift Safety* | Struktur repo, toolchain, algoritma indikator D-01..D-10, config, training, export, runtime, event schema, replay harness, milestone | **Panduan implementasi utama** |
| `model_spec_data.md` — *Model AI Forklift Safety — Spesifikasi & Strategi Data* | Daftar model, kandidat model pretrained, dataset publik, strategi data tanpa data primer, pemetaan 20 skenario (bagian 4A) | Pilihan model dan data |
| `design_document.md` / `AI_Forklift_Safety_System_Design.pdf` | Desain jangka panjang (termasuk load stability dan tip-over yang ditunda ke fase berikutnya), riset 2025–2026 | Latar belakang; jangan implementasikan bagian "[Later phase]" |

Link asli (butuh login claude.ai, tidak bisa dibuka Claude Code):
- Design Document: https://claude.ai/artifact/LoaWufSbXJKJbpVRFcdLXq
- Spesifikasi & Strategi Data: https://claude.ai/artifact/H4dkbx8BjWagcxaUFAxoy6
- Technical Spec: https://claude.ai/artifact/YVasfw451g5SbHL2ahv6Bs

---

## 4. Kapabilitas WAJIB (harus ada dan teruji)

### 4.1 KPI yang terkait AI

| KPI | Ambang lulus | Arti untuk kita |
|---|---|---|
| UC2-K3 Deteksi kondisi pengemudi | ≥ 80% dari 20 skenario (≥ 16) | Target internal **≥ 19/20** |
| UC2-K4 Peringatan diterima — **PENGGUGUR** | ≥ 95% | Setiap alert harus punya **kode D-xx dan level yang tepat**. Kode salah = gagal |
| UC2-K5 Waktu penyampaian — **PENGGUGUR** | Rata-rata kabin ≤ 2 s, command center ≤ 5 s | Deteksi cepat; event diberi timestamp di edge (`ts_event_ms`) |
| UC2-K7 Rekaman kejadian | ≥ 95% rekaman bisa dibuka | Snapshot + klip MP4 yang bisa diputar, dengan waktu dan jenis alert |

### 4.2 Level alert (dari Logbook)

- **A-02 (sedang):** buzzer sekali di kabin + notifikasi command center. Dipicu **D-01, D-05, D-06, D-10**.
- **A-03 (berat):** buzzer berulang sampai kondisi selesai + notifikasi prioritas tertinggi. Dipicu **D-02, D-04, D-09** (dan pelanggaran zona, bukan bagian kita).
- **Tidak ada eskalasi otomatis antar kode** selama uji KPI.

### 4.3 Dua puluh skenario Lembar A — semuanya harus terdeteksi dengan kode yang benar

| No | Kode | Peragaan | Durasi | Level | Indikator |
|---|---|---|---|---|---|
| 1 | D-01 | Pejam mata lambat berulang | 60 s | A-02 | PERCLOS 30 s / kedip lambat |
| 2 | D-01 | Sama, kepala tegak menghadap depan | 60 s | A-02 | sama |
| 3 | D-01 | Sama, peraga lain | 60 s | A-02 | sama |
| 4 | D-01 | Sama, **cahaya redup** | 60 s | A-02 | sama (uji di ruangan redup) |
| 5 | D-02 | Pejam mata terus | 3 s | A-03 | microsleep ≥ 1,5 s |
| 6 | D-02 | Pejam mata terus, peraga lain | 3 s | A-03 | sama |
| 7 | D-02 | Pejam mata sambil **sedikit menunduk** | 3 s | A-03 | sama; harus tetap D-02, bukan D-04 |
| 8 | D-05 | Menoleh ke kanan | 3 s | A-02 | pandangan lepas ≥ 1,5 s |
| 9 | D-05 | **Menunduk ke pangkuan** | 3 s | A-02 | pitch turun, mata terbuka; jangan jadi D-02/D-04 |
| 10 | D-05 | Menoleh ke kiri | 3 s | A-02 | pandangan lepas ≥ 1,5 s |
| 11 | D-05 | **Menoleh bergantian kanan-kiri tiap 1 s** | 30 s | A-02 | **akumulasi** ≥ 50% dari 6 s |
| 12 | D-06 | Ponsel posisi mengetik | 3 s | A-02 | YOLO `phone`; jangan jadi D-05 |
| 13 | D-06 | Ponsel di telinga | 3 s | A-02 | YOLO `phone` |
| 14 | D-06 | Ponsel setinggi kemudi, **sebagian tertutup tangan** | 3 s | A-02 | YOLO `phone` |
| 15 | D-10 | Tanpa helm, rompi terpasang | 15 s | A-02 | YOLO `head_no_helmet` |
| 16 | D-10 | **Tanpa rompi**, helm terpasang | 15 s | A-02 | YOLO `torso` tanpa `vest` |
| 17 | D-10 | Tanpa helm dan tanpa rompi | 15 s | A-02 | kirim **satu** alert D-10 |
| 18 | D-04 | Kepala tunduk perlahan lalu tegak, diulang | 30 s | A-03 | anggukan ≥ 2 kali / 30 s |
| 19 | D-04 | **Kepala miring ke bahu, bertahan** | 5 s | A-03 | roll > 25° ≥ 1,5 s |
| 20 | D-09 | **Lensa kamera ditutup telapak tangan** | 15 s | A-03 | kamera terhalang ≥ 1 s (tanpa wajah) |

Jeda antarskenario minimal 30 detik. Minimal satu peraga **bukan anggota tim kalibrasi**, jadi sistem harus jalan untuk wajah yang belum pernah dilihat.

### 4.4 Prioritas kode (jika beberapa indikator aktif bersamaan)

`D-09 > D-02 > D-04 > D-06 > D-01 > D-05 > D-10` — kirim hanya kode paling atas.

### 4.5 Kamera depan (belum diuji di Lembar A, tetap dibangun)

- Deteksi `person` dan rintangan: `forklift`, `pallet_jack`, `gse_vehicle`, `aircraft_part`, `ladder_stand`, `obstacle`.
- Tracking (ByteTrack) + arah gerak (mendekat/menjauh).
- Estimasi jarak dari titik kaki + kalibrasi kamera.
- Event usulan (belum disepakati GMF): F-01 orang di zona peringatan + mendekat (A-02), F-02 orang di zona bahaya (A-03), F-03 pesawat/GSE di zona bahaya (A-03).

---

## 5. Ringkasan teknis

Detail lengkap ada di `docs/technical_spec.md`. Ringkasnya:

- **Model 1+2 (wajah):** MediaPipe Face Landmarker (478 landmark, blendshape `eyeBlinkLeft/Right`, `jawOpen`, transformation matrix untuk pose kepala). Running mode `VIDEO`, `num_faces=1`.
- **Sinyal per frame:** `openness` (EAR / baseline), `eye_closed`, `eye_valid`, `yaw/pitch/roll` (relatif ke baseline), `mouth_open`, `face_present`. Median filter 3 frame.
- **Kalibrasi:** pengemudi menatap depan ~5 s di awal → `ear_baseline` dan pose nol; ada fallback baseline populasi.
- **Model 3 (YOLO kabin):** YOLO11n, kelas `phone`, `helmet`, `head_no_helmet`, `vest`, `torso`. `no_vest` = `torso` tanpa `vest` (IoU < 0,3).
- **Model 4+5 (depan):** YOLO11n + ByteTrack.
- **D-09 kamera terhalang:** dari frame mentah (std, variansi Laplacian, mean), bukan model. Dengan lampu IR, telapak tangan bisa membuat gambar **terang dan rata**, jadi deteksi berdasarkan keseragaman, bukan hanya gelap.
- **Semua ambang** di `configs/thresholds.yaml`, pemetaan kode/level/prioritas di `configs/events.yaml` (isi awal ada di Technical Spec bagian 4).

### Event schema (JSON)

```json
{
  "event_id": "laptop-20261108-000123",
  "device_id": "laptop-dev",
  "phase": "start",
  "code": "D02",
  "level": "A-03",
  "ts_event_ms": 1794130000123,
  "ts_condition_start_ms": 1794129998610,
  "confidence": 0.91,
  "snapshot_path": "events/2026-11-08/laptop-20261108-000123.jpg",
  "clip_path": "events/2026-11-08/laptop-20261108-000123.mp4",
  "model_versions": { "cabin": "cabin_v1", "face": "mediapipe-x.y" }
}
```

---

## 6. Struktur repo

```
forklift-ai/
├── CLAUDE.md
├── docs/                    # dokumen referensi (bagian 3)
├── configs/
│   ├── runtime.yaml         # sumber kamera, backend, output alert
│   ├── thresholds.yaml
│   └── events.yaml
├── src/
│   ├── capture/             # webcam, file video, URL (HP), nanti GStreamer Jetson
│   ├── backends/            # pytorch / onnxruntime / tensorrt (interface sama)
│   ├── cabin/               # face, signals, calibration, yolo cabin, indicators
│   ├── front/               # yolo depan, tracker, distance
│   ├── decision/            # prioritas, cooldown, hysteresis, event lifecycle
│   ├── io/                  # alert (beep laptop / GPIO nanti), MQTT/file, snapshot, klip
│   ├── ui/                  # overlay debug: sinyal, indikator aktif, level
│   └── main.py
├── tools/
│   └── recorder.py          # rekam skenario + tandai kebenaran dengan hotkey
├── training/
│   ├── datasets/            # data.yaml per model (data asli TIDAK di-commit)
│   ├── scripts/             # download, konversi kelas, split, augmentasi, auto-label
│   └── train_cabin.py, train_front.py
├── eval/
│   ├── scenarios/           # video + truth.csv
│   └── replay_eval.py
└── tests/                   # unit test indikator dengan sinyal sintetis
```

---

## 7. Aturan kerja untuk Claude Code

1. **Kriteria lulus = bagian 4.** Setiap fitur harus bisa ditelusuri ke kode D-xx atau KPI. Jangan menambah fitur di luar scope tanpa ditanya.
2. **Tidak ada angka ajaib di kode.** Semua threshold, FPS, path, dan pemetaan kode lewat file di `configs/`.
3. **Satu pipeline untuk live dan replay.** Sumber kamera dan file video harus lewat interface `capture` yang sama, dengan timestamp asli, supaya hasil replay sama dengan live.
4. **Interface sinyal wajah tetap.** Output modul wajah (`openness`, `eye_valid`, `yaw/pitch/roll`, dst.) harus sama apa pun modelnya (MediaPipe sekarang, MtDMS mungkin nanti).
5. **Tidak pernah mengklaim "lulus" tanpa laporan replay harness.** Setelah mengubah threshold atau model, jalankan `eval/replay_eval.py` dan tampilkan hasilnya.
6. **Split data per orang dan per sesi**, bukan acak per frame. Test set berisi orang yang tidak ada di training.
7. **Simulasi NIR di laptop:** sediakan opsi `--simulate-nir` (grayscale 3 channel + CLAHE) di runtime dan di training, supaya model dan threshold sudah terbiasa dengan gambar mirip kamera NIR.
8. **Unit test untuk setiap indikator** dengan deret sinyal sintetis (misalnya mata tertutup 1,4 s → tidak aktif; 1,6 s → aktif; menoleh bergantian 1 s → D-05 aktif lewat akumulasi).
9. **Jangan commit** dataset, video rekaman, wajah orang, bobot model besar, atau engine TensorRT. Gunakan `.gitignore` dan simpan di storage terpisah.
10. **Lisensi:** banyak dataset driver monitoring hanya untuk riset non-komersial. Catat lisensi setiap dataset di `training/datasets/LICENSES.md`.
11. **Privasi:** rekaman wajah hanya dari orang yang setuju direkam. Jangan unggah ke layanan eksternal.
12. **Kerjakan per milestone** (bagian 8), satu milestone per sesi kerja, dan cek kriteria "selesai jika" sebelum lanjut.

---

## 8. Daftar pekerjaan Fase 1 (laptop)

Kerjakan berurutan. Setiap item punya kriteria selesai.

### L1. Setup lingkungan
- Python 3.10+, venv, `opencv-python`, `mediapipe`, `ultralytics`, `onnxruntime`, `numpy`, `pyyaml`, `paho-mqtt`, `pytest`.
- **Selesai jika:** `python -m src.main --source webcam` membuka webcam dan menampilkan FPS.

### L2. Capture + recorder
- Interface `capture` untuk: webcam (index), file video, URL stream (HP).
- `tools/recorder.py`: rekam video webcam; tekan tombol angka/hotkey untuk menandai **awal dan akhir skenario + kode D-xx**; simpan `.mp4` + baris di `truth.csv` (`video,start_s,end_s,expected_code,expected_level,peraga,cahaya`).
- **Selesai jika:** satu sesi berisi 20 skenario bisa direkam dan `truth.csv` terisi otomatis.

### L3. Pipeline wajah + sinyal + kalibrasi
- MediaPipe Face Landmarker, ROI, opsi `--simulate-nir`, perhitungan sinyal per frame, kalibrasi 5 detik + baseline populasi.
- Overlay debug: nilai `openness`, `yaw/pitch/roll`, status mata.
- **Selesai jika:** overlay stabil, dan kalibrasi berhasil untuk minimal 3 orang berbeda.

### L4. Indikator wajah + keputusan + alert
- D-01, D-02, D-04 (anggukan + roll), D-05 (terus-menerus + akumulasi), D-09.
- Modul keputusan: prioritas kode, cooldown A-02 (20 s), A-03 berulang, hysteresis tutup 1,5 s, event `start`/`end`.
- Alert laptop: beep sekali (A-02), beep berulang (A-03); event JSON ke file dan (opsional) Mosquitto lokal.
- Unit test indikator dengan sinyal sintetis.
- **Selesai jika:** skenario 1–11 dan 18–20 dapat dipicu manual di depan webcam dengan kode dan level yang benar.

### L5. Data + training YOLO kabin
- Skrip download/konversi dataset publik (PPE helmet/head/vest, phone), `class_map.yaml`, konversi grayscale+CLAHE, split per orang.
- Rekam data tim di webcam: pakai/lepas **helm proyek dan rompi** (siapkan properti fisik), ponsel di berbagai posisi termasuk tertutup tangan.
- Auto-label (Grounding DINO atau YOLO-World) → koreksi di CVAT/Label Studio.
- Training YOLO11n sesuai Technical Spec bagian 5; integrasikan ke D-06 dan D-10.
- **Selesai jika:** mAP50 ≥ 0,85 dan recall `phone`, `head_no_helmet`, `torso` tanpa `vest` ≥ 0,90 di test set orang luar training.

### L6. Replay harness
- `eval/replay_eval.py` memutar video dengan timestamp asli lewat pipeline yang sama, mencocokkan event dengan `truth.csv`.
- Laporan: tabel 20 skenario (terdeteksi, kode benar, level benar, waktu deteksi), skor K3, % kode+level benar, confusion matrix antar kode, alarm palsu per jam pada video "normal".
- **Selesai jika:** laporan Markdown/HTML keluar otomatis dari satu perintah.

### L7. Tuning sampai lolos gerbang
- Rekam minimal 3 orang (1 di luar tim), semua 20 skenario, termasuk cahaya redup.
- Tuning `thresholds.yaml` hanya dengan set validasi; laporkan angka akhir dari set test.
- **Gerbang Fase 1:**
  - K3 ≥ 18/20 di orang luar tim.
  - Kode + level benar 100% untuk skenario yang terdeteksi.
  - Waktu deteksi skenario 3 detik (D-02, D-05, D-06) ≤ 2,0 s.
  - Alarm palsu < 1 per jam pada video penggunaan normal (orang duduk, bicara, minum, melihat sekitar wajar).

### L8. Snapshot dan klip (persiapan K7)
- Ring buffer ~10 s; saat event `start` simpan snapshot JPG dan klip MP4 (10 s sebelum + 5 s sesudah), H.264 `yuv420p`, `-movflags +faststart`, di thread terpisah.
- **Selesai jika:** 15 event acak, semua klip bisa diputar di browser dan berisi waktu + kode.

### L9. Kamera depan (paralel, prioritas lebih rendah)
- YOLO11n COCO untuk `person` + ByteTrack + arah gerak, dengan webcam atau stream HP.
- Siapkan pipeline data untuk kelas hangar (Apron Dataset, LOCO, sintetis) dan training sesuai Technical Spec bagian 6.
- Estimasi jarak titik kaki dengan kalibrasi kamera (papan catur); uji di lantai bertanda jarak 2–10 m.
- **Selesai jika:** orang terdeteksi dan dilacak stabil; error jarak tercatat.

### L10. Siap pindah ke Jetson
- Export ONNX semua model YOLO; backend `tensorrt` disiapkan (dipakai di Jetson).
- Dokumentasi `docs/JETSON_MIGRATION.md`: apa yang harus dikalibrasi ulang.
- **Selesai jika:** semua model bisa jalan lewat backend ONNX Runtime di laptop dengan hasil sama seperti PyTorch.

---

## 9. Yang pasti berubah saat pindah ke Jetson (jangan di-hardcode)

| Hal | Laptop (Fase 1) | Jetson (Fase 2) |
|---|---|---|
| Kamera kabin | Webcam RGB, depan wajah | NoIR + lampu IR, lensa ~60–90°, wajah sampai dada |
| Sumber kamera | OpenCV index / URL | GStreamer `nvarguscamerasrc` / `v4l2src` |
| Backend | PyTorch / ONNX Runtime | TensorRT FP16 (engine di-build di Jetson) |
| MediaPipe | CPU laptop | CPU Jetson (ukur FPS; jika kurang → MtDMS TensorRT) |
| Alert | Beep laptop | Buzzer GPIO |
| Event | File / Mosquitto lokal | MQTT ke Minerva via 5G |
| Encode klip | ffmpeg | x264 software (Orin Nano tidak punya hardware encoder) |
| Threshold | Hasil tuning laptop | **Kalibrasi ulang** dengan rekaman gladi di forklift |

---

## 10. Pertanyaan terbuka (jangan diasumsikan diam-diam)

1. Titik mulai pengukuran K5: usulan = `ts_event_ms` dibuat di edge. Belum disepakati GMF.
2. Apakah uji dilakukan saat kendaraan bergerak. Untuk sekarang `require_moving: false`.
3. Broker MQTT, format topik, dan skema final dari tim Minerva.
4. Kode dan level event kamera depan (F-01..F-03) belum disepakati HSE GMF.
5. Kebijakan privasi dan retensi klip wajah pengemudi.

Jika sebuah tugas bergantung pada salah satu poin ini, buat sebagai opsi di config dan sebutkan asumsinya di ringkasan pekerjaan.

---

## 11. Definisi selesai Fase 1

- Semua 20 skenario Lembar A punya indikator yang jalan, kode dan level yang benar.
- Gerbang L7 terpenuhi berdasarkan laporan replay harness, bukan perkiraan.
- Snapshot dan klip MP4 dibuat untuk setiap event.
- Kamera depan mendeteksi dan melacak orang.
- Kode siap dipindah ke Jetson hanya dengan mengganti config dan backend.
