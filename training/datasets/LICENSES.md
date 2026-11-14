# Lisensi dataset dan bobot — model YOLO kabin (L5)

Aturan kerja 10 (CLAUDE.md): catat lisensi setiap dataset. Lisensi di bawah adalah **klaim pengunggah**
(kartu HuggingFace / file README Roboflow) per 2026-11-14. Kolom "Terverifikasi" = sudah dicek ke sumber
asli (upstream). Sebelum model dipakai di luar PoC internal, minta tim legal memeriksa ulang.

Semua sumber bisa diunduh **tanpa token** (diuji 2026-11-14 dengan `--limit 60`).

## Dataset

| Sumber (sources.yaml) | URL | Gambar tersedia | Lisensi (klaim) | Upstream | Terverifikasi | Dipakai |
|---|---|---|---|---|---|---|
| `ppe_jhboyo` | https://huggingface.co/datasets/jhboyo/ppe-dataset | 15.500 (YOLO: helmet, head, vest) | MIT (kartu HF) | ds1 = Kaggle andrewmvd/hard-hat-detection (Hard Hat Workers, MakeML); ds2 = Kaggle snehilsanyal/construction-site-safety-image-dataset-roboflow (Roboflow, CC BY 4.0) | **Tidak** — lisensi MIT dipasang oleh pengunggah ulang; lisensi upstream ds1 belum dicek | ya (maks 6.000 di build penuh) |
| `ppe_vincent` | https://huggingface.co/datasets/VincentGOURBIN/ppe-detection | 3.712 (= ~1.237 asli x 3 augmentasi) | CC BY 4.0 | Roboflow `vincentspace/ppe-detection-1-cniwr` v2 | Ya (README.dataset.txt + data.yaml: "License: CC BY 4.0") | ya |
| `construction_keremberke` | https://huggingface.co/datasets/keremberke/construction-safety-object-detection | 398 | CC BY 4.0 | Roboflow `roboflow-universe-projects/construction-site-safety` v1 (gambar dari beberapa proyek Universe + 2 video YouTube) | Ya (README.roboflow.txt: "License: CC BY 4.0"); gambar asal YouTube belum dicek | ya |
| `coco_phone` | label: https://github.com/ultralytics/assets/releases/download/v0.0.0/coco2017labels.zip ; gambar: http://images.cocodataset.org/ | 4.395 gambar dengan `cell phone` (setelah filter luas kotak >= 0,05%) | Anotasi COCO: CC BY 4.0. Gambar: lisensi Flickr per gambar (sebagian CC BY-NC / CC BY-NC-SA) | COCO 2017 (cocodataset.org/#termsofuse) | Ya (terms of use COCO) — **sebagian gambar non-komersial** | ya |
| `cellphone_mahek` | https://huggingface.co/datasets/MahekDharod/cellphone-detection-dataset | 57 | MIT (kartu HF) | tidak diketahui (foto WhatsApp/IMG) | Tidak | **tidak** (label tidak pas pada cek visual; `enabled: false`) |
| `roboflow_example` | Roboflow Universe (isi sendiri) | — | tergantung proyek | — | — | hanya jika `ROBOFLOW_API_KEY` ada |

Catatan:
- `ppe_jhboyo` ds2 dan `construction_keremberke` berasal dari keluarga dataset Roboflow yang sama; kemungkinan ada gambar
  ganda lintas sumber. `split.py` menggabungkan near-duplicate (dHash) lintas sumber sebelum split.
- Label `torso` dan sebagian `head_no_helmet` adalah **turunan otomatis** (lihat bagian bobot di bawah), bukan anotasi upstream.

## Bobot model yang dipakai dalam pipeline

| Bobot | Dipakai untuk | Lisensi |
|---|---|---|
| `yolo11n.pt` (Ultralytics) | bobot awal training | AGPL-3.0 (atau Ultralytics Enterprise License) |
| `yolo11n-pose.pt` (Ultralytics) | pose-fill: torso dari bahu-pinggul, kepala dari keypoint wajah | AGPL-3.0 |
| `yolov8s-worldv2.pt` + CLIP ViT-B/32 | `autolabel.py` (opsional) | AGPL-3.0 (Ultralytics), MIT (CLIP) |

## Implikasi

- **PoC internal:** semua sumber aktif boleh dipakai untuk eksperimen.
- **Produk komersial untuk GMF:** gambar COCO dengan lisensi Flickr NC, ds1 `ppe_jhboyo` yang lisensinya belum
  terverifikasi, dan lisensi AGPL-3.0 Ultralytics perlu ditinjau legal. Model final sebaiknya dilatih ulang dari data
  berlisensi jelas + rekaman sendiri (Spesifikasi & Strategi Data, "Penting soal lisensi").
