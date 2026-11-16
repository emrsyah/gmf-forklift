# Uji dengan kamera HP + simulasi kamera Jetson

Tujuan: memakai **HP sebagai kamera** sambil **meniru kamera yang akan dibeli** untuk forklift (Jetson Orin
Nano: kamera kabin NoIR + lampu IR, HFOV ~60–90°; kamera depan warna). Dengan begitu kita bisa menilai apakah
kandidat kamera (8 MP IMX219, 12 MP IMX477/IMX708, varian IR) cukup bagus **sebelum** membeli.

Profil kamera ada di `configs/camera_profiles.yaml`. Simulasi berjalan tepat setelah capture
(`src/capture/camera_sim.py`), sehingga live, file video, dan replay memakai jalur yang sama, dengan
timestamp asli yang tidak diubah.

## 1. Apa yang disimulasikan dan apa yang tidak

| Disimulasikan | Tidak bisa disimulasikan |
|---|---|
| Sudut pandang (HFOV): crop tengah frame HP | Pantulan kulit, mata, dan kacamata pada 850 nm (IR) |
| Resolusi dan aspek keluaran (mis. 1640×1232) | Ruangan gelap yang diterangi lampu IR (kamera HP tidak melihat IR) |
| Tampilan NIR mono, vignette lampu IR, bloom | Rolling shutter dan auto-exposure kamera target |
| Noise sensor dan cahaya redup (modifier `redup`) | Kualitas lensa sebenarnya (ketajaman, flare) |
| Distorsi barrel lensa lebar (perkiraan) | Getaran forklift yang sebenarnya (modifier `forklift` hanya meniru) |
| Batas FPS, kompresi JPEG/MJPEG (modifier `stream`) | |

Akibatnya, simulasi ini cukup untuk menilai **geometri** (apakah wajah cukup besar, apakah kepala tetap
dalam frame saat menoleh, berapa jarak deteksi orang) dan **ketahanan model** terhadap gambar mono dan
noise. Simulasi ini **tidak** menggantikan uji dengan kamera NIR asli di kabin. Threshold tetap harus
dikalibrasi ulang di Jetson (CLAUDE.md bagian 9).

Model memperkecil gambar sebelum inferensi (wajah: `cabin.face.max_side_px` = 640, YOLO: `imgsz` 640).
Resolusi 8 MP atau 12 MP **tidak** membuat model melihat lebih banyak piksel. Yang berpengaruh adalah
HFOV (seberapa besar wajah dan objek di frame) serta kualitas gambar sebelum diperkecil.

## 2. Menyambungkan HP ke laptop

HP dan laptop harus berada di **Wi-Fi yang sama**. Hotspot HP juga bisa dipakai, tetapi jangan memakai
Wi-Fi tamu yang memblokir koneksi antarperangkat.

### Android: aplikasi "IP Webcam" (disarankan)

1. Pasang **IP Webcam** (Pavel Khlebovich) dari Play Store.
2. Atur resolusi di *Video preferences → Video resolution*: **1280×960** atau **1440×1080** (aspek
   **4:3**, lihat bagian 3). Atur *Quality* ~70–80. Format bawaan adalah MJPEG, yang paling cocok untuk
   OpenCV.
3. Tekan *Start server*. Layar HP akan menampilkan alamat seperti `http://192.168.1.23:8080`.
4. URL untuk OpenCV adalah **`http://192.168.1.23:8080/video`**. Cek dulu di browser laptop.
5. Jalankan:

```bash
.venv\Scripts\python -m src.main --source http://192.168.1.23:8080/video --camera-profile imx219_77ir
```

Untuk lensa ultrawide (0,5×), pilih kamera ultrawide di pengaturan IP Webcam jika HP mendukungnya.
Jika tidak, gunakan bagian 2 "alternatif".

### iPhone (dan Android lain)

- **DroidCam** (iOS/Android) atau **Iriun Webcam**: pasang aplikasi di HP dan klien di Windows. Klien
  membuat **webcam virtual** di Windows. Pakai sebagai indeks webcam, misalnya
  `--source 1` (laptop biasanya `0`, webcam virtual `1` atau `2`; coba berurutan).
- DroidCam juga menyediakan stream IP. URL yang umum dipakai adalah `http://IP-HP:4747/video`
  (**perkiraan**: cek alamat yang tertera di aplikasi dan buka dulu di browser). Jika browser bisa
  menampilkannya, OpenCV biasanya juga bisa.
- **Resolusi saat dipakai sebagai webcam:** `src.main` meminta resolusi dari `cabin.source.width/height`
  di `configs/runtime.yaml` (bawaan 640×480). Itu terlalu kecil untuk profil 1640×1232 (upscale ~2,5×).
  Untuk jalur ini, ubah ke resolusi 4:3 yang didukung webcam virtual, misalnya `width: 1280`, `height: 960`.
  `tools/recorder.py` selalu meminta 1280×720 (16:9), sehingga profil 4:3 akan kena peringatan aspek
  dan geometrinya **berbeda** dengan saat live. Untuk merekam, pakai URL stream. Jalur URL tidak
  terpengaruh, karena resolusinya mengikuti pengaturan di aplikasi HP.
- Versi gratis aplikasi tertentu membatasi resolusi (mis. 480p). Untuk profil resolusi tinggi, simulasi
  akan memberi peringatan *upscale* (detail lebih sedikit dari kamera target).

### Kenapa tidak membuat aplikasi HP sendiri?

Tidak perlu. Aplikasi di atas sudah memberi stream MJPEG/HTTP atau webcam virtual yang bisa dibuka OpenCV,
dan semua simulasi kamera dilakukan di laptop. Aplikasi khusus baru berguna jika kita butuh hal yang tidak
diberikan aplikasi tersebut, misalnya mengunci exposure/fokus sambil mengirim timestamp sensor. Untuk
menilai kandidat kamera, hal itu belum diperlukan.

### Tips latensi

- Turunkan resolusi stream HP jika FPS turun. 1280×960 biasanya cukup untuk semua profil kabin.
- Pakai MJPEG (bawaan IP Webcam), bukan RTSP H.264. H.264 menambah buffer, sehingga latensi bisa 0,5–2 s.
- Router dekat dan pita 5 GHz. Matikan hemat daya / layar mati di HP.
- Overlay menampilkan FPS. Indikator memakai timestamp saat frame **diterima** laptop, jadi jeda jaringan
  menggeser waktu deteksi. Untuk mengukur waktu deteksi (KPI K5), pakai webcam USB atau rekaman.

## 3. Mengukur HFOV kamera HP

Simulasi meng-crop frame HP berdasarkan **HFOV HP**. Jika nilainya salah, ukuran wajah/objek dan jarak
kamera depan ikut salah. HFOV harus diukur pada **stream yang dipakai**, karena mode video, aspek, dan
stabilisasi elektronik mengubah HFOV (stabilisasi meng-crop ±10%). Matikan stabilisasi jika bisa.

**Aspek 4:3 lebih baik.** Kamera target 4:3 (IMX219 1640×1232) tidak bisa diisi penuh dari stream HP 16:9.
Simulasi akan memberi peringatan dan HFOV efektifnya menjadi lebih sempit.

### Cara A: meteran (5 menit)

1. Tempel HP menghadap tegak lurus ke dinding pada jarak `D` (mis. 2,00 m, diukur dari lensa).
2. Buka stream tanpa profil: `.venv\Scripts\python -m src.main --source <URL> --fps-only`.
3. Tandai di dinding titik paling kiri dan paling kanan yang masih terlihat di layar. Ukur lebarnya `L`.
4. `HFOV = 2 × atan(L / (2 × D))`. Contoh: L = 2,80 m, D = 2,00 m → HFOV = 2 × atan(0,70) = **70,0°**.

### Cara B: papan catur (lebih akurat)

```bash
.venv\Scripts\python tools\calibrate_camera.py --source http://IP-HP:8080/video --pattern 9x6 --square-mm 25 --out configs\phone_camera.yaml
```

`--out` wajib diisi agar `configs/front_camera.yaml` tidak tertimpa. Dari hasilnya,
`HFOV = 2 × atan(W / (2 × fx))`, dengan `W` = lebar frame dan `fx` = K[0][0].

### Menyimpan hasil

Isi `phone.hfov_deg` dan `phone.hfov_measured: true` di `configs/camera_profiles.yaml`, atau berikan per
jalan dengan `--phone-hfov 70`. Selama `hfov_measured: false` di YAML, overlay kamera depan tetap
menampilkan "BELUM DIKALIBRASI", walaupun `--phone-hfov` diberikan (karena angkanya bisa saja tebakan).

Jika profil **lebih lebar** dari HP (mis. `imx708_noir_wide_102` dengan HP 70°), simulasi tidak bisa
menirunya dan akan mencetak peringatan. Pakai lensa **ultrawide** HP (~100–120°), ukur HFOV-nya, lalu
jalankan dengan `--phone-hfov <nilai itu>`.

## 4. Memasang HP

### Kabin (meniru posisi kamera di forklift)

- Pasang **di atas dan di depan pengemudi**, kira-kira di posisi kamera asli (dekat atap kabin atau tiang A),
  **miring ke bawah** ~15–30° menghadap wajah. Jangan sejajar mata seperti webcam laptop.
- Jarak ke wajah sesuai rencana pemasangan, biasanya ~60–100 cm. Frame harus berisi wajah sampai dada
  (helm dan rompi harus terlihat untuk D-10).
- Pegangan HP harus kaku (penjepit/tripod). Getaran HP sungguhan menambah getaran simulasi.
- Untuk skenario cahaya redup (no. 4), kamera HP tetap butuh cahaya tampak. Tambahkan modifier `redup`
  untuk meniru noise; lihat juga keterbatasan di bagian 1.

### Depan

- Tinggi lensa **~1,2 m**, menunduk **~10°** (sesuaikan dengan `mount.h_m` dan `mount.theta_deg` di
  `configs/front_camera.yaml`). Ukur dengan meteran dan aplikasi inklinometer.
- Lantai diberi tanda 2, 4, 6, dan 8 m dari titik tepat di bawah lensa (docs/FRONT_CAMERA.md bagian 3).
- Dengan `--camera-profile`, K untuk jarak **diturunkan dari profil** (frame sudah di-crop dan diresize).
  Blok `camera` di `front_camera.yaml` diabaikan, sedangkan `mount` dan `distance` tetap dipakai.

```bash
.venv\Scripts\python tools\run_front.py --source http://IP-HP:8080/video --camera-profile imx477_m12_75 --phone-hfov 70
```

## 5. Protokol uji: apa yang dibandingkan antar profil

Langkah yang disarankan:

1. Rekam satu sesi **mentah** sekali, lalu bandingkan semua profil pada rekaman yang sama:

```bash
.venv\Scripts\python -m tools.recorder --peraga P1 --source http://IP-HP:8080/video --camera-profile imx219_77ir --save-raw
.venv\Scripts\python -m tools.compare_profiles --video eval\scenarios\<sesi>.mp4 --phone-hfov 70
```

   `compare_profiles` mencetak tabel per profil: % frame dengan wajah, % mata valid, median openness,
   jumlah deteksi YOLO dan rata-rata conf, event per kode, serta waktu simulasi/pipeline.
   Tanpa `--save-raw`, recorder menyimpan frame **simulasi** dan mencatat profilnya di
   `<video>.camera.json`. Rekaman itu bisa langsung diputar di `eval/replay_eval.py`. Untuk profil NIR,
   tambahkan `--simulate-nir` agar CLAHE sama seperti saat live.

2. Uji live per profil (`run.bat` → menu **8**). Amati:

| Aspek | Yang dilihat | Cukup jika (usulan, bukan KPI GMF) |
|---|---|---|
| Landmark wajah | Overlay stabil, tidak hilang saat menoleh ±45° atau menunduk | Wajah terdeteksi > 95% frame pada posisi normal |
| Mata / openness | Status mata benar saat berkedip dan memejam; openness ~1 saat terbuka | D-01/D-02 terpicu dengan benar (skenario 1–7) |
| Kepala di frame | Kepala tetap di dalam frame saat menoleh/miring (skenario 8–11, 18–19) | Tidak terpotong tepi frame |
| YOLO kabin | conf `phone`, `helmet`/`head_no_helmet`, `vest`/`torso` di overlay debug (`d`) | D-06/D-10 terpicu (skenario 12–17) |
| D-09 | Tutup lensa HP dengan telapak tangan | D-09 aktif ≤ 1 s |
| Kamera depan | Error jarak di 2/4/6/8 m (catat estimasi vs meteran) | Error < 10% sampai 6 m |
| FPS | FPS di overlay dan `sim ms` di compare_profiles | Pipeline tetap ≥ 15 FPS |

Kombinasi yang layak dicoba:

- `imx219_77ir` vs `imx708_noir_66`: 8 MP vs 12 MP pada HFOV hampir sama. Bedanya kecil jika semua
  gambar diperkecil ke 640.
- `imx219_77ir` vs `imx708_noir_wide_102` (pakai lensa ultrawide HP): apakah wajah terlalu kecil di
  lensa lebar?
- Tambahkan `+forklift` dan `+redup` untuk uji ketahanan, misalnya `imx219_77ir+forklift+redup`.

## 6. Referensi CLI

| Perintah | Opsi |
|---|---|
| `python -m src.main` | `--camera-profile NAMA[+modifier]`, `--phone-hfov DEG` (default `cabin.camera_profile` di runtime.yaml) |
| `python tools/run_front.py` | sama (default `front.camera_profile`) |
| `python -m tools.recorder` | sama + `--save-raw` |
| `python -m tools.compare_profiles` | `--video`, `--profiles a,b`, `--phone-hfov`, `--max-s`, `--no-yolo`, `--out` |
| `python -m src.capture.camera_sim --list [--role kabin\|depan]` | daftar profil |

`--camera-profile none` mematikan simulasi. Profil aktif ditampilkan di judul jendela
(`... [profil: NAMA]`). Jendela diperkecil sampai `ui.display_max_side` (hanya tampilan).
Profil dengan `nir: true` otomatis menyalakan `cabin.simulate_nir` (CLAHE di pipeline, tidak dua kali).
