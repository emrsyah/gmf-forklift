# Lisensi dataset — model YOLO kamera depan (L9, Model 4)

Aturan kerja 10 (CLAUDE.md). Dicek ke sumber per 2026-11-14. Ini bukan nasihat hukum: sebelum model dipakai
di luar PoC internal, minta tim legal memeriksa ulang. Untuk kabin, lihat `LICENSES.md`.

## Dataset

| Sumber (`sources_front.yaml`) | URL | Isi | Lisensi | Komersial? | Akses | Terverifikasi |
|---|---|---|---|---|---|---|
| `coco_person` | label: https://github.com/ultralytics/assets/releases/download/v0.0.0/coco2017labels.zip ; gambar: http://images.cocodataset.org/ | COCO 2017, gambar berisi `person` (subset 5.000) | Anotasi CC BY 4.0; gambar = lisensi Flickr per gambar (campuran, **sebagian CC BY-NC / BY-NC-SA**) | Sebagian gambar **tidak** | otomatis, tanpa token | Ya (COCO terms of use) |
| `loco` | https://github.com/tum-fml/loco ; gambar https://go.mytum.de/239870 | 5.593 gambar gudang beranotasi (forklift, pallet_truck, pallet, SLC, stillage) | **CC0 1.0** (file LICENSE repo) | Ya | otomatis, tanpa token | Ya (LICENSE GitHub = CC0 1.0 Universal; berlaku untuk repo — README menyatakan dataset didedikasikan ke domain publik) |
| `crowdhuman` | https://huggingface.co/datasets/sshao0516/CrowdHuman (mirror) ; asal https://www.crowdhuman.org/ | 15.000 gambar train, orang dalam kerumunan | Kartu HF: **CC BY-NC 4.0**; terms Megvii: **hanya riset & pendidikan non-komersial, gambar tidak boleh didistribusikan** | **Tidak** | otomatis (mirror HF tidak di-gate) atau manual (Google Drive) | Sebagian: terms dibaca dari README mirror. Bahwa akun `sshao0516` adalah mirror resmi Megvii **belum diverifikasi** |
| `apron` | https://github.com/apronai/apron-dataset ; data https://wilddash.cc/aprondataset | 10.098 gambar dashcam apron, 43 kategori | Lisensi AIT: **non-komersial saja**; dilarang mendistribusikan data; **model hasil training ("derivative work") juga TIDAK boleh dipakai untuk tujuan komersial** | **Tidak** (termasuk model) | **manual**: registrasi + persetujuan | Ya (file LICENCE repo) |
| `rod_obstacles` | https://huggingface.co/datasets/ty-li/Obstacle-Detection-Dataset-YOLO | ROD: 24.326 gambar trotoar, 25 kelas; diambil hanya gambar berisi cone/barel/kursi/bangku/pot/tong (~4.900) untuk **uji simulasi** | Kartu HF: **MIT**. Tetapi isinya gabungan 26 dataset Roboflow Universe (lisensi per proyek, umumnya CC BY 4.0, belum dicek satu per satu) + foto penulis | Belum pasti (cek 26 sumber) | otomatis, tanpa token | Sebagian (kartu HF dibaca 2026-11-24; daftar sumber ada di paper) |
| `synthetic` | — (Isaac Sim / Omniverse Replicator, rekaman tim) | placeholder | milik sendiri (cek lisensi aset 3D yang dipakai) | tergantung aset | lokal | — |
| `roboflow_forklift` | Roboflow Universe (isi sendiri) | opsional | tergantung proyek (pilih CC BY 4.0 / Public Domain) | tergantung | butuh `ROBOFLOW_API_KEY` | — |

## Bobot model yang dipakai dalam pipeline

| Bobot | Dipakai untuk | Lisensi |
|---|---|---|
| `yolo11n.pt` (Ultralytics, COCO) | bobot awal training + **teacher-fill** (label pseudo person/kendaraan di LOCO, CrowdHuman, Roboflow) | AGPL-3.0 (atau Ultralytics Enterprise License) |

Label pseudo hasil teacher-fill adalah turunan model yang dilatih pada COCO (gambar Flickr campuran lisensi).

## Implikasi

- **PoC internal (riset/eksperimen):** semua sumber boleh dipakai sesuai syarat masing-masing.
- **Model yang dipakai komersial untuk GMF:** bobot yang dilatih dengan **Apron** tidak boleh dipakai komersial
  (lisensi Apron melarang penggunaan komersial atas derivative work, termasuk model). **CrowdHuman** dan sebagian
  gambar **COCO** juga non-komersial. Model final untuk produksi sebaiknya dilatih ulang hanya dari LOCO (CC0),
  data sintetis milik sendiri, rekaman tim/GMF, dan dataset berlisensi komersial — dengan meng-set `enabled: false`
  pada `apron` dan `crowdhuman` (dan memfilter COCO per lisensi gambar) di `sources_front.yaml`.
- Ultralytics AGPL-3.0 berlaku untuk kode dan bobot turunan; produk tertutup butuh lisensi Enterprise.
- Jangan unggah dataset atau gambar turunannya ke layanan eksternal (Apron & CrowdHuman melarang redistribusi).
