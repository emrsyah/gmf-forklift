# Gateway Jetson ↔ dashboard MINERVA

Kontrak ke dashboard **tidak berubah**: sama dengan `qa/uc2/jetson-scope.md` dan
`minerva-forklift/src/integrations/hardware.ts` di repo impactto. Dokumen ini menjelaskan cara proses lain
di Jetson (GPS/zona, UPS) ikut mengirim lewat gateway yang sama.

```
proses CV kabin (src.main --gateway) ─┐  ws://127.0.0.1:8788   ┌─► ws://<ip-jetson>:8787 ─► dashboard
proses GPS/zona (Resan) ──────────────┼──► gateway ────────────┤   (event, gnss, ups, frame, telemetry)
proses UPS ───────────────────────────┘   (src.gateway.server) └◄── shown / handled / replay
```

## Kenapa proses terpisah

- CV (MediaPipe, YOLO, kamera) paling mungkin crash atau restart. Kalau CV restart, koneksi dashboard,
  `seq`, replay, dan alarm zona tetap jalan.
- Setiap produsen bisa diuji sendiri; penghubungnya cuma JSON lewat localhost (latensi beberapa ms).

## Menjalankan

```bash
python -m src.gateway.server                      # config: runtime.yaml bagian `gateway`
python -m src.main --source webcam --gateway      # CV kabin mengirim ke ws://127.0.0.1:8788
```

Dashboard: Pengaturan → mode Langsung → endpoint Jetson FL-01 = `ws://<ip-jetson>:8787`.

## Yang diurus gateway

| Hal | Perilaku |
|---|---|
| `seq` | Diisi gateway, naik 1 per `event`, disimpan ke `gateway_state/seq.txt` (fsync) **sebelum** dikirim. Tidak mulai dari 1 setelah restart. |
| Replay | 500 event terakhir di memori + `gateway_state/events.jsonl`. `{type:'replay', fromSeq}` dijawab ke klien itu saja. |
| Duplikat | Event dengan `event_id` + `state` yang sama tidak diberi `seq` baru (kirim ulang setelah ack hilang). |
| `forkliftId` | Diisi `FL-01` (config) jika produsen tidak mengirim. |
| Kode | `D02` dinormalisasi ke `D-02`. |
| Telemetri | Tiap 2 s: `cpuPct, gpuPct, tempC, memUsedMb` (Jetson: `/sys`), `fpsCabin` dari status CV, plus usulan `camOk, cabinAlive, ntpSynced, seq`. |
| Log | `gateway_state/log/YYYY-MM-DD.jsonl`: semua event keluar, `shown`, `handled`, replay, info klip. |

## Protokol produsen (ws://127.0.0.1:8788)

Kirim JSON teks. Bentuknya sama dengan pesan ke dashboard, **tanpa `seq`**:

```json
{ "type": "event", "code": "G-EXCL", "level": "A-03", "state": "started",
  "t_deteksi": 1795000000000, "t_buzzer": 1795000000040,
  "zoneId": "EXCL-1", "zoneType": "exclusion", "event_id": "gps-000123", "ack_id": 7 }
```

- `event_id` (unik per kejadian, sama untuk `started` dan `ended`) disarankan supaya kirim ulang tidak dobel.
- `ack_id` opsional: gateway membalas `{type:'ack', ack_id}` setelah event disimpan. Hapus dari antrean
  lokal hanya setelah ack. Contoh klien Python: `src/io/gateway_sink.py`.
- `gnss`, `ups`, `frame` diteruskan apa adanya (tanpa `seq`, tanpa buffer).
- `{type:'status', source:'cabin'|'forward', fps, camera_ok, clock_synced}`: tanda hidup produsen, digabung
  ke telemetri. Status lebih tua dari 6 s dianggap mati (`fpsCabin: 0`, `cabinAlive: false`).

## Live dan bukti kejadian

- **Live kabin:** proses CV mengirim `frame` (JPEG ~480 px, 4 fps, base64) lewat gateway. Hanya frame
  terbaru yang dikirim; frame dibuang untuk dashboard yang buffer kirimnya masih penuh (`frame_drop_backlog_kb`),
  supaya alarm tidak antre di belakang gambar. Matikan di `sink.gateway.live.enabled` jika kebijakan privasi
  tidak mengizinkan wajah pengemudi live.
- **Snapshot/klip:** gateway menyajikan `events/` (hanya `.jpg`/`.mp4`, mendukung Range) di
  `http://<host>:8789/media/...`. Event `started` membawa `snapshotUrl` (D-09: frame terakhir sebelum kamera
  tertutup). Klip ditulis setelah kondisi selesai (10 s sebelum sampai selesai + 2 s, maks 60 s), lalu gateway
  menyiarkan `{type:'media', seq, clipUrl, clipDurationS}`. Host URL = host yang dipakai dashboard untuk tersambung.
- **Replay:** setelah putus, dashboard meminta `fromSeq = lastSeq+1`; info klip yang selesai saat terputus ikut
  dikirim ulang. Saat halaman baru dibuka/di-reload: `fromSeq: 0, maxAgeS: 120` (alarm 2 menit terakhir saja).
- Perubahan dashboard ada di branch lokal `ai-evidence-live` repo impactto (belum di-push; perlu review Resan).

## Buzzer

Untuk sekarang buzzer kabin tetap di proses CV: dibunyikan dulu, baru event dikirim (`t_buzzer`).
Buzzer tidak pernah menunggu gateway. Kalau buzzer zona juga dari pin GPIO yang sama, perlu disepakati
satu pemilik pin (usulan: gateway, dengan prioritas antar-alarm), sebelum modul GPS/zona digabung.

## Yang sudah diuji (laptop, 18 Nov 2026, belum lewat 5G, belum di Jetson)

- Rekaman `sim_P1c` → CV → gateway → dashboard impactto mode Langsung: toast A-02, panel A-03 merah,
  "Kondisi berakhir" untuk `ended`, `shown` diterima gateway.
- Gateway dimatikan ±12 s di tengah sesi lalu dinyalakan lagi: `seq` lanjut dari 12, 7 event yang terjadi
  selama mati terkirim setelah tersambung, dashboard meminta `replay fromSeq=12` dan menerima 8 event.
- Unit test: `tests/test_gateway.py`.
