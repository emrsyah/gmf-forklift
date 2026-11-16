# STATUS — kerja semalam 14 Nov 2026

> Bukan klaim lulus. Semua angka di bawah berasal dari unit test sintetis, video sintetis, atau dataset publik.
> Gerbang L7 hanya bisa dinilai dari replay rekaman gladi 20 skenario dengan peraga luar tim.

## Pagi ini (urut, ~15 menit)

1. `.venv/Scripts/python -m tools.check_pose` — 30 detik, konfirmasi saja. Tanda pitch **sudah diverifikasi semalam** pada 486 wajah publik: korelasi pitch vs posisi hidung di bawah garis mata = −0,815 → menunduk = pitch negatif (`pitch_sign: 1` benar).
2. `.venv/Scripts/python -m src.main --source webcam --no-yolo` — mulai **tanpa YOLO kabin**, karena siapa pun di depan laptop tanpa helm proyek + rompi akan langsung memicu D-10 (benar menurut aturan, tapi menutupi kode lain). Tatap depan 5 s (kalibrasi), lalu coba: pejam 2 s (D-02 A-03), menoleh 2 s (D-05), lihat pangkuan (D-05), tutup lensa (D-09), miring ke bahu (D-04), angguk 2x (D-04). Event + klip di `events/`. Setelah itu coba tanpa `--no-yolo` dengan helm/rompi dan ponsel.
3. Putuskan opsi risiko K4 di bawah (`pitch_only_extra_hold_s`, `nod_count`).
4. RunPod: kunci API ditolak (401). Buat kunci baru (izin read/write), lalu ikuti `training/runpod/README.md` untuk training cabin penuh di GPU.
5. `.venv/Scripts/python -m pytest` → harus 133 passed.

**Diputuskan 14 Nov:** `D04_head_drop.nod_count: 3` dan `nod_active_hold_s: 5.0` aktif (risiko K4 #2 dan #7). Lembar 20 skenario sintetis tetap 20/20; skenario 18 terdeteksi di anggukan ke-3 (~8,9 s dari 30 s). `pitch_only_extra_hold_s` (risiko #1) masih 0, tunggu rekaman gladi.

## YOLO kabin cabin_v1 (RunPod RTX 4090, 14 Nov 2026) — split TEST publik, BUKAN gerbang L5

~26 ribu gambar publik (train 23.254 / val 1.358 / test 1.520), 150 epoch, imgsz 640, ~3,5 jam, ~$3,2.
Terpasang di `models/weights/cabin_best.*`; cabin_v0 di `models/weights/archive/`. Laporan: `training/runs/cabin_v1/gate_report.md` (tidak di-commit).

| Kelas | mAP50 v0 → v1 | Recall@0.35 v1 |
|---|---|---|
| all | 0,633 → **0,799** | |
| phone | 0,25 → **0,528** | 0,42 |
| helmet | 0,82 → **0,924** | 0,86 |
| head_no_helmet | 0,61 → **0,781** | 0,64 |
| vest | 0,70 → **0,871** | 0,81 |
| torso | 0,79 → **0,889** | 0,80 (no_vest 0,71) |

Masih di bawah gerbang (mAP50 0,85, recall 0,90). Ponsel tetap terlemah: data publik tidak punya ponsel dari sudut kabin.
`D06_phone.conf` kembali ke 0,35 (spec). Inferensi 640 di CPU laptop ~64 ms (tiap 2 frame).
Pelajaran RunPod: pod di-stop dengan volume lokal tidak bisa di-resume saat GPU host dipakai orang → pakai **network volume** untuk training berikutnya (atau "Start Pod using CPUs" untuk mengambil file).

## Uji webcam live 14 Nov 2026 (P0, satu orang — bukan gerbang)

- `check_pose`: 7/7 OK, pitch menunduk −23 / mendongak +35 (tanda benar).
- Live `--no-yolo`: D-02, D-05 (menoleh + bergantian), D-04 (angguk/miring), D-01 benar. **D-04 palsu** di awal dari 2 lirikan ke bawah (risiko K4 #2 terbukti nyata) + ekor buzzer ~20 s (risiko #7).
- Rekaman diagnostik terpandu (`tools.recorder --guide configs/guides/diag_eyes_cover_lap.yaml`) menemukan 3 masalah, sudah diperbaiki lewat config (bukan nilai spec, diberi komentar di `thresholds.yaml`, diuji di `tests/test_webcam_findings.py`):

| Masalah | Sebelum | Perbaikan | Sesudah (replay rekaman yang sama) |
|---|---|---|---|
| Pejam santai: openness berayun 0,21–0,34 di sekitar ambang 0,30 | D-02 2,7 s setelah mata tertutup (gerbang 2,0 s) | hysteresis `closed_openness_exit 0.45`, `closed_blink_exit 0.40` | D-02 1,5 s |
| Telapak tangan + cahaya bocor: std ~18, lap_var 10–50 | D-09 tidak pernah aktif | `std_max 25`, `lap_var_max 60` (scene normal std ~65–80, lap ~600–950) | D-09 ~1,5 s setelah tangan menutup |
| Lihat pangkuan dalam: MediaPipe kehilangan wajah | D-05 tidak pernah aktif | `face_lost_offroad_hold_s 5.0` (tidak berlaku saat kamera terhalang) | D-05 1,5 s setelah menunduk |
| Lihat pangkuan biasa: pitch berayun −24..−28 | run D-05 putus-sambung | `hysteresis_deg 5` (hanya run kontinu; akumulasi tetap ambang mentah supaya anggukan lambat tidak jadi D-05) | stabil |

- Bug yang ditemukan: `configs/scenarios.yaml` `no:` terbaca boolean (YAML 1.1) → recorder crash. Diperbaiki + tes.
- Kedip normal (cepat) memang tidak dihitung tertutup (openness min ~0,33–0,38). Itu disengaja; D-01/D-02 hanya butuh pejam ≥ 0,4 s.
- Semua angka ini dari **satu orang, webcam laptop**. Wajib kalibrasi ulang di Jetson + kamera NIR.

Tidak ada yang di-commit (repo sudah `git init`). `.gitignore` menahan data, video, wajah, dan bobot.

## Ringkasan per milestone

| # | Milestone | Status | Bukti | Menunggu kamu |
|---|---|---|---|---|
| L1 | Setup | ✅ selesai | `python -m src.main --source webcam --fps-only` → 30 FPS 640×480 | — |
| L2 | Capture + recorder | ✅ kode selesai | webcam/file/URL/GStreamer lewat satu interface; recorder menulis mp4 + `.ts.csv` (timestamp asli) + `truth.csv` | rekam 1 sesi 20 skenario |
| L3 | Wajah + sinyal + kalibrasi | ✅ kode selesai | MediaPipe 1.2.3 VIDEO mode, ~9 ms/frame CPU; tanda roll/yaw diverifikasi pada foto publik (rotasi ±20° → roll ±20°, mirror membalik yaw) | **cek tanda pitch** (`python -m tools.check_pose`, 30 detik); kalibrasi 3 orang |
| L4 | Indikator + keputusan + alert | ✅ kode + test | 7 indikator, prioritas/cooldown/hysteresis/A-03 berulang, beep, JSONL + MQTT opsional; tes skenario sintetis | picu skenario 1–11, 18–20 manual di webcam |
| L5 | Data + training YOLO kabin | ✅ pipeline; 🟡 cabin_v0 (CPU) | dataset publik tanpa token → grayscale+CLAHE, split per grup, augmentasi; lihat bagian "YOLO kabin" | training penuh di GPU (RunPod) + data kabin asli untuk gerbang L5 |
| L6 | Replay harness | ✅ selesai | `python -m eval.replay_eval` → `report.md/.json/.html`; tes end-to-end D-09 dengan MediaPipe asli lulus | video gladi asli |
| L7 | Tuning sampai lolos gerbang | ⛔ butuh rekaman | — | rekaman 3 orang (1 luar tim) |
| L8 | Snapshot + klip | ✅ selesai | klip 10 s + 5 s, H.264 yuv420p Constrained Baseline, faststart (moov sebelum mdat), teks waktu + kode | buka 15 klip acak di browser |
| L9 | Kamera depan | ✅ kode + test (agen) | person + ByteTrack + jarak titik kaki + arah; 4 orang ID stabil di klip sintetis, ~30 FPS | kalibrasi papan catur + uji lantai 2–10 m |
| L10 | Siap Jetson | ✅ ONNX + parity | backend ORT murni identik dengan Ultralytics-ONNX, cocok dengan PyTorch; `docs/JETSON_MIGRATION.md` | build engine di Jetson |

## Latency di laptop ini (Ryzen 7 8845HS, CPU saja)

- Pipeline kabin tanpa YOLO: median ~11 ms/frame (~90 FPS).
- YOLO11n 640 ONNX Runtime CPU: ~48 ms. Model kabin cabin_v0 (416): ~15 ms; tiap 2 frame → pipeline penuh rata-rata ~19 ms/frame (~50 FPS).

## Risiko K4 yang ditemukan tes sintetis (keputusan untukmu)

Default = nilai Technical Spec apa adanya. Opsi mitigasi ada di `configs/thresholds.yaml`, keduanya diuji di `tests/test_scenarios_synthetic.py`.

1. **Skenario 18 (anggukan) bisa terkirim D-05 duluan** jika anggukan lambat dan dalam (pitch < −25° lebih dari 1,5 s), karena D-05 aktif sebelum anggukan ke-2 menyalakan D-04. Mitigasi: `D05_distraction.pitch_only_extra_hold_s: 1.0` (D-05 dari menunduk saja butuh 2,5 s). Efek samping: skenario 9 (lihat pangkuan) terdeteksi ~2,5 s, bukan ~1,5 s (masih di dalam 3 s skenario, tapi di atas gerbang 2,0 s).
2. **Alarm palsu D-04**: dua kali melirik dasbor (pitch −20° selama 0,6 s) dalam 30 s memenuhi definisi "anggukan" spec. Mitigasi: `D04_head_drop.nod_count: 3` (skenario 18 tetap terdeteksi).
3. **Skenario 1–4 (D-01)**: satu pejaman > 1,5 s akan terbaca D-02 (prioritas lebih tinggi). Tidak ada mitigasi otomatis — arahkan peraga agar kedipan lambat ≤ 1,4 s.
4. **Skenario 18 "perlahan"**: jika satu anggukan (turun → kembali) > `nod_max_s` 2,0 s, tidak dihitung anggukan. Ukur saat gladi.
5. **Skenario 8/10**: jika menoleh terlalu jauh MediaPipe kehilangan wajah → D-05 tidak menghitung frame itu. Ukur saat gladi.
6. **Skenario 12 (ponsel posisi mengetik)**: dengan recall ponsel cabin_v0 ~0,2, ponsel sering tidak terdeteksi, padahal kepala menunduk → **D-05 terkirim (kode salah = gagal K4)**, bukan sekadar terlewat. Data ponsel-di-tangan dari sudut kabin adalah prioritas pertama pemakaian GPU.
7. **D-04 setelah skenario 18**: sesuai spec, D-04 tetap aktif selama ≥ 2 anggukan masih di jendela 30 s → buzzer A-03 berbunyi ~20 s setelah anggukan berhenti. Opsi `D04_head_drop.nod_active_hold_s: 5.0` menutupnya ~5 s setelah anggukan terakhir (diuji).
8. **Jeda antarskenario harus kembali netral** (helm + rompi terpakai, menghadap depan). Tes urutan 20 skenario berturut-turut (`tests/test_sequence_and_calibration.py`) lulus 20/20 kode+level dengan jeda netral 30 s; tanpa kembali netral (mis. langsung dari "tanpa rompi" ke "tanpa helm dan rompi") event D-10 masih terbuka sehingga skenario 17 tidak mendapat alert baru.
9. **Kalibrasi**: ditolak jika pose tidak lurus (|roll| > 15°, |yaw| > 25°, |pitch| > 35°) atau tidak stabil; setelah 3 kali gagal → baseline populasi (tekan `c` untuk ulang). Mencegah kalibrasi terkunci saat kepala miring (skenario 19). Untuk replay, setiap video harus diawali ~5 s menatap depan.

## Asumsi yang saya buat (cek)

- Rumus EAR dan syarat `eye_valid` di PDF Technical Spec terpotong. Dipakai: EAR standar (Soukupová) dengan indeks dari spec; `eye_valid` = wajah ada, pitch ≥ −25°, |yaw| ≤ 45°.
- `eye_closed` = openness < 0,30 **dan** blink > 0,5 (spec). Opsi `closed_logic` jika blendshape kurang andal di NIR.
- Modul keputusan: satu event terbuka pada satu waktu; kode berprioritas lebih tinggi memotong (end + start); kode lebih rendah menunggu sampai event terbuka ditutup (hysteresis 1,5 s).
- `event_id` = `laptop-YYYYMMDD-HHMMSS-NNNN` (unik antar-run); event menambah field `t_video_s` (dipakai harness) dan `zone_id: null`.
- `ts_event_ms` = timestamp frame saat ambang terlampaui (pertanyaan terbuka #1 K5).
- Logbook `.xlsx` tidak ada di repo → 20 skenario diambil dari CLAUDE.md 4.3 (`configs/scenarios.yaml`).
- Kode kamera depan F-01..F-03 = usulan, dimatikan (`front_events_enabled: false`).

## YOLO kabin (L5)

- Sumber (semua tanpa token, lisensi di `training/datasets/LICENSES.md`): `jhboyo/ppe-dataset` (helm/kepala/rompi), `VincentGOURBIN/ppe-detection` (CC BY 4.0), `keremberke/construction-safety` (CC BY 4.0), COCO `cell phone` (anotasi CC BY 4.0, sebagian gambar Flickr non-komersial).
- `torso` tidak ada di dataset publik → dibuat dari keypoint `yolo11n-pose` (bahu–pinggul) dan dari kotak no-vest; `head_no_helmet` yang tidak dilabel sumber diisi dari keypoint wajah. Semua kotak buatan ditandai pseudo di `manifest.csv`.
- Semalam: dataset `cabin_v0` = 1.000 gambar/sumber (3.398 gambar; train 2.716 + 2.716 augmentasi, val 323, test 359), training YOLO11n **di CPU** imgsz 416, 30 epoch → `models/weights/cabin_best.onnx`. Hasil di bagian "Hasil training" di bawah.
- Training penuh (semua data, imgsz 640, 150 epoch) siap untuk GPU: `training/runpod/README.md`.

### Hasil training cabin_v0 (CPU, 30 epoch, ~3,5 jam) — split TEST publik, BUKAN gerbang L5

| Kelas | Precision | Recall | mAP50 | Recall @conf 0,35 |
|---|---|---|---|---|
| phone | 0,56 | 0,19 | **0,25** | 0,20 |
| helmet | 0,83 | 0,74 | 0,82 | 0,75 |
| head_no_helmet | 0,72 | 0,54 | 0,61 | 0,58 |
| vest | 0,66 | 0,68 | 0,70 | 0,61 |
| torso | 0,82 | 0,72 | 0,79 | 0,70 |
| **semua** | | | **0,633** (gerbang 0,85) | no_vest 0,63 |

Laporan lengkap: `training/runs/cabin_v0/gate_report.md`. Bobot: `models/weights/cabin_best.pt/.onnx` (input 416).

- **phone lemah**: ponsel COCO kecil dan jauh; tidak ada ponsel di tangan dari sudut kabin. D-06 dengan model ini belum bisa diandalkan.
- **Perbaikan bug semalam**: model dilatih dengan gambar grayscale+CLAHE, tapi runtime awalnya memberi RGB jika `--simulate-nir` tidak dipakai → `phone` palsu di dasi. Sekarang input YOLO selalu diberi preprocessing yang sama dengan training (`cabin.yolo.preprocess: nir`).
- **`D06_phone.conf` 0,35 → 0,50**, di-tune pada split VALIDASI (recall 0,21 → 0,18, alarm palsu 35 → 17, precision 0,45 → 0,60). Alasannya tercatat di `thresholds.yaml`; kembalikan/tune ulang setelah model GPU.
- Latency pipeline kabin + YOLO kabin (tiap 2 frame) di laptop: rata-rata ~19 ms/frame (~50 FPS).
- **Gerbang L5 yang sebenarnya (recall ≥ 0,90 pada orang di luar training, kamera kabin) belum bisa diukur** — data publik adalah foto proyek dan COCO, bukan sudut kabin/NIR, dan tidak ada ponsel tertutup tangan.
- Disk C: tinggal ~2,7 GB. Cache yang bisa dihapus: `~/.cache/clip` (~340 MB), `data/weights/yolov8s-worldv2.pt`, `data/raw/` (setelah build).

## YOLO kamera depan (L9 bagian 2)

- Pipeline data + training siap (`training/scripts/build_front_dataset.py`, `training/train_front.py`, `training/runpod/run_train_front.sh`), 7 kelas: person, forklift, pallet_jack, gse_vehicle, aircraft_part, ladder_stand, obstacle. Split per sekuens/lokasi; augmentasi strip tiang mast.
- Sumber otomatis: COCO person, LOCO (CC0, unduh 769 MB sekali), CrowdHuman (mirror HF, non-komersial). **Apron Dataset harus didaftarkan manual** di wilddash.cc → `data/manual/apron/` (lisensi non-komersial, termasuk model hasil latih).
- **`ladder_stand` tidak punya sumber publik**, `aircraft_part` hanya pesawat utuh → perlu data sintetis/rekaman.
- Belum dilatih penuh (butuh GPU). Runtime kamera depan saat ini memakai COCO `yolo11n.pt` (person + alias kendaraan). Tracking butuh backend `ultralytics`/`tensorrt` (ORT murni tidak bisa tracking).

## RunPod

Kunci API yang diberikan dijawab **HTTP 401 (unauthorized)** oleh API GraphQL dan REST RunPod — mungkin terpotong saat ditempel, salah salin, atau izinnya terbatas. Kunci tidak disimpan di file mana pun. Sebaiknya **rotate kunci itu** (sudah tertempel di chat).

## Ponsel palsu D-06 + cabin_v2 (16 Nov 2026) — SEDANG BERJALAN

**A. Filter ponsel runtime (selesai, aktif):** `src/cabin/phone_filter.py`, config `D06_phone.filter` di `thresholds.yaml`,
dipanggil dari `yolo_cabin.interpret` / `pipeline.py` (hanya saat YOLO jalan). Tolak `phone` di luar wilayah
pengemudi (satuan = ukuran wajah), terlalu kecil/besar/pipih, di tengah wajah (mulut/dagu/janggut saat menghadap
kamera), atau tidak stabil (< 2 run YOLO berturut-turut di tempat yang sama). Wajah hilang (menunduk ke ponsel,
skenario 12) -> kotak wajah terakhir 5 s -> kotak kepala/torso YOLO -> lolos dengan conf 0,55 (tidak pernah
menolak hanya karena wajah hilang). `tests/test_phone_filter.py` 23 tes (geometri 12/13/14 lolos, D-06 < 2 s,
pola palsu ditolak). Log webcam lokal (cabin_v1): palsu di diag4 (kecil, tepi frame) dan di dalam wajah
tertolak; ponsel asli uji 15-16 Nov ada di dalam wilayah. Angka awal dari SATU orang/webcam -> kalibrasi ulang.

**C. cabin_v2 di RunPod:** pod `rdmkhjwcjr1zt5` (RTX 4090 secure, $0,89/jam, mulai ~11:28 UTC). Data = sumber v1
(split identik v1 lewat `split.anchor_sources`) + `openimages_phone` (maks 6.000) + `coco_lookalike` negatif
keras (maks 3.000, label parsial: pose-fill kepala/torso + buang gambar ber-helm/rompi menurut cabin_v1).
Fine-tune dari cabin_v1, 50 epoch. Pod **stop sendiri** (`guard.sh`: 10 menit setelah DONE, batas keras 5 jam
= 16:35 UTC); hasil di volume pod `/workspace/results_cabin_v2.tgz` (+ `.sha256`). Gerbang (D) BELUM dinilai:
`cabin_best.*` masih cabin_v1. Ambil hasil: lihat `training/runpod/README.md` / laporan agen (start pod, scp, cek
sha256, lalu terminate pod). Pod lama `rmchfltymromcj` tidak disentuh.
