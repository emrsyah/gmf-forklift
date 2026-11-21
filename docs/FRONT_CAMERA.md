# Kamera depan (L9): deteksi, tracking, arah gerak, jarak

Modul `src/front/` mendeteksi `person` dan rintangan (YOLO11n), melacaknya dengan ByteTrack,
menghitung jarak lantai dari titik kaki, dan menentukan apakah objek mendekat. Hasilnya berupa
**indikator kandidat** F01/F02/F03. Kode-kode ini **usulan** dan belum disepakati HSE GMF
(pertanyaan terbuka #4). `events.yaml: front_events_enabled: false`, jadi tidak ada alert yang dikirim.

| Kode (usulan) | Kondisi | Level (events.yaml) |
|---|---|---|
| F01 | `person` ≤ `warning_zone_m` **dan** mendekat lebih cepat dari `approach_speed_mps` | A-02 |
| F02 | `person` ≤ `danger_zone_m` | A-03 |
| F03 | `aircraft_part` / `gse_vehicle` ≤ `danger_zone_m` | A-03 |

Konfigurasi:

- `configs/runtime.yaml` → `front.yolo`: bobot, backend, `tracker`, `person_class`.
- `configs/thresholds.yaml` → `front`: `person_conf`, `approach_speed_mps`, `speed_window_s`, `warning_zone_m`, `danger_zone_m`.
- `configs/front_camera.yaml`: berisi K/distorsi, `mount.h_m`, `mount.theta_deg`, tinggi kelas untuk cadangan jarak, parameter motion, alias kelas, dan prioritas F.

## Menjalankan

```bash
python tools/run_front.py --source webcam:1                 # webcam kedua
python tools/run_front.py --source http://<ip-hp>:8080/video # aplikasi IP camera di HP
python tools/run_front.py --source video.mp4 --no-show --no-realtime --out out.mp4 --jsonl out.jsonl
```

Overlay menampilkan:

- `#id kelas jarak kecepatan`. Kecepatan negatif berarti mendekat.
- Akhiran `(c)` berarti kotak terpotong tepi bawah, `(h)` berarti jarak dari tinggi kotak.
- Panah merah berarti mendekat.
- Garis oranye/merah menandai batas zona peringatan/bahaya (perkiraan di sumbu tengah).
- Kandidat F-xx tampil di pojok kiri atas.

`--no-realtime` membuat file diproses secepat mungkin. Timestamp tetap diambil dari video, jadi
kecepatan (m/s) tetap benar.

## 1. Kalibrasi intrinsik (papan catur)

1. Buat papan: `python tools/calibrate_camera.py --make-board data/calib/board_9x6_25mm.png --pattern 9x6 --square-mm 25`.
   Cetak A4 landscape **100% / actual size** (bukan "fit to page"), rekatkan pada papan yang rata dan kaku
   (kardus tebal/akrilik; kertas melengkung = K salah). Ukur garis 50 mm dan sisi kotak dengan penggaris;
   pakai sisi kotak **hasil ukur** di `--square-mm`. `--pattern` adalah jumlah sudut **dalam**: papan 10×7 kotak = `9x6`.
2. Untuk webcam, setelan kamera (resolusi, MJPG, `autofocus: false`, `focus`) otomatis diambil dari
   `runtime.yaml` `front.source`, sama dengan `tools/run_front.py`, dan dicatat di `camera.capture`.
   Jika fokus atau resolusi diubah setelah kalibrasi, kalibrasi ulang.
3. Ambil 15–30 view. Papan harus menutupi seluruh area frame, termasuk tepi dan sudut: jendela menampilkan
   area yang sudah tertutup (hijau) dan sel 3×3 yang belum (kotak merah). Variasikan kemiringan papan ±30–45°
   dan jaraknya. View ditolak otomatis jika papan masih bergerak (blur) atau terlalu mirip view sebelumnya.
   Setelah kalibrasi, view dengan error jauh di atas median dibuang lalu kalibrasi diulang sekali.

```bash
# live: SPASI/c ambil view, a = ambil otomatis tiap 1 s, u = batalkan view terakhir, q = selesai & kalibrasi
python tools/calibrate_camera.py --source webcam:1 --pattern 9x6 --square-mm 25 --save-views data/calib/front
# dari folder gambar
python tools/calibrate_camera.py --images "data/calib/front/*.png" --pattern 9x6 --square-mm 25
# lensa sangat lebar (>~120°)
python tools/calibrate_camera.py --images "data/calib/front/*.png" --fisheye
```

Tool ini menulis blok `camera:` di `configs/front_camera.yaml` (`calibrated: true`, `K`, `dist`,
`image_size`, `rms_px`) dan **mempertahankan** blok lain. Komentar YAML hilang setelah ditulis
ulang. Hasil yang baik punya RMS < 0,5 px; jika > 1 px, ulangi kalibrasi. Selama kamera belum
dikalibrasi, overlay menampilkan "KAMERA BELUM DIKALIBRASI".

## 2. Ukur pemasangan (`mount`)

- `h_m`: tinggi pusat lensa dari lantai, dalam meter. Ukur dengan meteran.
- `theta_deg`: sudut tunduk kamera di bawah horizontal (positif = menunduk). Ada dua cara:
  - Pakai aplikasi inklinometer di HP yang ditempel sejajar badan kamera.
  - **Lebih akurat:** taruh penanda di lantai, tepat di depan kamera, pada jarak `d` yang diketahui (misalnya 5 m). Baca baris piksel `v` penanda di frame (setelah kalibrasi), lalu hitung:
    `theta = atan(h / d) - atan((v - cy) / fy)`. Nilai `cy` dan `fy` diambil dari K hasil kalibrasi. Jika lensa berdistorsi, pakai `v` yang sudah di-undistort.

Rumus jarak yang dipakai (Tech Spec bag. 6):
`angle = theta + atan(y_n)`, `distance_m = h / tan(angle)`, dengan `y_n` = koordinat
ternormalisasi titik kaki setelah undistort. Kesalahan 1° pada `theta` di h = 2 m menggeser
jarak 10 m sekitar ±0,9 m. Karena itu, `theta` adalah parameter paling sensitif.

Ada dua cadangan untuk titik kaki yang tidak andal:

- Kotak menyentuh tepi bawah (kaki terpotong): jarak = `min(titik tepi bawah, fy·H/tinggi_kotak)`. Keduanya adalah batas atas jarak sebenarnya, sehingga orang yang sangat dekat tetap masuk zona bahaya.
- Titik kaki di atas horizon: jarak dari tinggi kotak (`distance.class_height_m`, person = 1,65 m).

## 3. Uji error jarak (lantai bertanda 2–10 m)

1. Pasang kamera pada `h_m` dan `theta_deg` yang sudah diukur. Tandai lantai dengan lakban di 2, 3, 4, 5, 6, 8, dan 10 m, diukur dari titik lantai tepat di bawah lensa sepanjang sumbu maju. Untuk uji lateral, tambahkan tanda ±1,5 m di 4 m dan 8 m.
2. Peraga berdiri dengan ujung sepatu tepat di garis, menghadap kamera, selama ~3 s per titik. Rekam satu video untuk semua titik.
3. Jalankan `python tools/run_front.py --source rekaman.mp4 --no-show --no-realtime --jsonl jarak.jsonl`, lalu ambil median `distance_m` track orang tersebut selama ia diam di setiap tanda.
4. Isi tabel di bawah. Target awal yang diusulkan: error ≤ 10% sampai 6 m dan ≤ 15% sampai 10 m. Target ini belum disepakati.
5. Jika error naik sebanding dengan jarak, periksa `theta_deg` lebih dulu, lalu `h_m`. Jika error berbeda antara kiri dan kanan, periksa kalibrasi distorsi.

| Tanggal | Kamera / resolusi | h_m | theta_deg | Jarak benar (m) | Lateral (m) | Estimasi median (m) | Error (m) | Error (%) | Metode (foot/clipped/height) | Catatan (cahaya, sepatu, pose) |
|---|---|---|---|---|---|---|---|---|---|---|
| | | | | 2 | 0 | | | | | |
| | | | | 3 | 0 | | | | | |
| | | | | 4 | 0 | | | | | |
| | | | | 4 | +1,5 | | | | | |
| | | | | 5 | 0 | | | | | |
| | | | | 6 | 0 | | | | | |
| | | | | 8 | 0 | | | | | |
| | | | | 8 | -1,5 | | | | | |
| | | | | 10 | 0 | | | | | |

Untuk uji arah gerak, peraga berjalan mendekat dari 10 m ke 2 m, lalu menjauh. F01 harus muncul
hanya saat mendekat di dalam zona peringatan. Catat waktu sampai `approaching` menyala (dibatasi
oleh `speed_window_s` dan `motion.min_span_s`).

## Batasan saat ini

- Bobot COCO `yolo11n.pt` hanya punya `person`. Kelas hangar (`forklift`, `pallet_jack`, `ladder_stand`, `obstacle`) belum ada. `aircraft_part`/`gse_vehicle` dipetakan sementara dari COCO `airplane`/`truck`/`bus`/`car` (`classes.aliases`) sampai bobot `front_best` dilatih (Tech Spec bag. 6).
- Lantai diasumsikan datar dan forklift tidak miring. Saat forklift mengerem, `theta` efektif berubah.
- Kecepatan adalah kecepatan **relatif** terhadap kamera. Saat forklift bergerak maju, orang yang diam pun terbaca "mendekat". Ini memang perilaku yang diinginkan untuk F01.
- Belum ada debounce awal kandidat. Hysteresis dan cooldown ditangani modul keputusan.
- Fase 2 (Jetson): kalibrasi ulang K, `h_m`, dan `theta_deg` pada kamera dan posisi pasang yang sebenarnya.
