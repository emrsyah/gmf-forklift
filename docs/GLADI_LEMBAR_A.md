# Gladi Lembar A — protokol rekaman untuk gerbang Fase 1

Tujuan: rekaman yang bisa **membuktikan** gerbang L7 (K3 ≥ 18/20 untuk peraga luar tim, kode + level benar 100%,
deteksi skenario 3 s ≤ 2,0 s, alarm palsu < 1 per jam). Video simulasi (`eval/scenarios/sim_*`) tidak cukup.
Tiga sumber miss di video simulasi berasal dari cara merekam, bukan dari detektor (bagian 3).

Privasi (CLAUDE.md aturan 11): hanya peraga yang setuju direkam. Video tidak di-commit dan tidak diunggah.

## 1. Persiapan

- Minimal **3 peraga**, **1 di luar tim kalibrasi**. Properti: helm proyek, rompi, ponsel.
- Kamera kabin di posisi pasang final (Jetson) atau webcam sejajar sedikit di bawah mata.
- Sesi **redup** untuk skenario 4 (lampu kabin dimatikan / malam).
- Rekam juga **≥ 15 menit penggunaan normal** per sesi (duduk, bicara, minum, menoleh wajar, menguap) untuk
  menghitung alarm palsu per jam.

## 2. Alur

1. **Kalibrasi**: di awal sesi peraga **menatap lurus, mata terbuka biasa, ≥ 15 detik**. Jangan memejam di menit
   pertama. Kalibrasi menolak mata terpejam dan memakai baseline adaptif jika gagal, tetapi kalibrasi yang
   berhasil tetap paling akurat (status di overlay: `terkalibrasi (EAR ...)`).
2. **Jeda ≥ 30 s antarskenario** (syarat Logbook). Jangan dipersingkat: D-01 memakai jendela PERCLOS 30 s,
   jadi pejaman di skenario sebelumnya masih menahan D-01 dan menutupi kode berikutnya (prioritas D-01 > D-05).
3. **Tandai waktu dari aksi yang sebenarnya**, bukan dari jadwal:
   - laptop: `python -m tools.recorder --peraga P2 --session gladi1_P2` → SPASI tepat saat peraga **mulai**
     beraksi dan tepat saat **selesai**; `n` untuk segmen normal.
   - forklift: pencatat menulis jam mulai/selesai → `python -m tools.truth_from_times --times sesi.csv
     --recordings recordings/cabin --date YYYY-MM-DD --out eval/scenarios/truth_gladi.csv`.
4. **Evaluasi**: `python -m eval.replay_eval --truth eval/scenarios/truth_gladi.csv`
   (tambahkan `--ignore-codes D10` hanya jika peraga memang tidak memakai APD di skenario non-D10).
   Tempel laporan `eval/reports/<waktu>/report.md` sebelum mengklaim lulus.

## 3. Brief peraga (dari kegagalan yang terukur di rekaman kita)

| No | Kode | Peragaan yang benar | Kesalahan yang sudah terjadi |
|---|---|---|---|
| 1–4 | D-01 | Kedip **lambat**, tiap pejaman **< 1,5 s** (hitung "satu"), terbuka 1–2 s, ulangi 60 s | Pejaman 1,6–3,6 s = mikrotidur → sistem benar mengirim **D-02** (prioritas lebih tinggi) → kode salah untuk Lembar A |
| 5–6 | D-02 | Pejam terus **3 s**, kepala tegak | Peraga memejam 4 s lebih awal dari tanda → event di luar jendela truth |
| 7 | D-02 | Pejam 3 s sambil **sedikit** menunduk (dagu turun ± 10–15°, **< 20°**) | Menunduk ~28° = sama persis dengan skenario 9 di kamera (mata tampak tertutup) |
| 8, 10 | D-05 | Menoleh jelas ke kanan/kiri (≥ 45°) **3 s** | — |
| 9 | D-05 | Menunduk ke pangkuan, **mata terbuka**, 3 s | — |
| 11 | D-05 | Menoleh kanan–kiri bergantian tiap 1 s selama 30 s | — |
| 12–14 | D-06 | Ponsel diangkat dan terlihat kamera 3 s (mengetik / di telinga / setinggi kemudi tertutup tangan) | Ponsel di bawah frame tidak terlihat sama sekali |
| 15–17 | D-10 | Helm/rompi dilepas **15 s**, badan menghadap kamera | — |
| 18 | D-04 | Tunduk **perlahan** (± 1–2 s turun-naik), tegak, ulangi ≥ 3 kali dalam 30 s | Anggukan > 2 s dulu tidak terhitung (batas kini 0,8–3,0 s) |
| 19 | D-04 | Kepala miring ke bahu (> 25°) **tahan 5 s** | — |
| 20 | D-09 | **Telapak tangan menutup lensa penuh** 15 s | Tutup setengah: fraksi di sekitar ambang (kini ada hysteresis) |

Catat setiap peragaan yang menyimpang (mis. "P3 pejam 2 s di no 2") di kolom `scenario` truth agar
hasilnya bisa dibaca dengan benar, dan **jangan** mengubah threshold untuk mengejar peragaan yang salah.

## 4. Pertanyaan terbuka untuk GMF

- **D-01 vs D-02**: berapa lama pejaman di skenario 1–4 yang dimaksud? Jika peraga GMF memejam ≥ 1,5 s, Logbook
  (D-02 ≥ 1,5 s, prioritas D-02 > D-01) membuat alert yang keluar **D-02**. Perlu disepakati sebelum uji K4.
- **Skenario 7**: batas "sedikit menunduk" (usulan < 20°).
