"""Gateway Jetson <-> dashboard MINERVA (impactto, `qa/uc2/jetson-scope.md`).

Satu proses kecil yang terpisah dari proses CV:
- server WebSocket untuk dashboard (port 8787): kirim `event`/`gnss`/`ups`/`frame`/`telemetry`,
  terima `shown`/`handled`/`replay`;
- server WebSocket lokal untuk produsen (CV kabin, GPS/zona, UPS) di 127.0.0.1:8788;
- `seq` naik terus dan disimpan ke file (tidak mulai dari 1 setelah restart), buffer 500 event terakhir
  untuk `replay`, log JSONL semua pesan masuk/keluar.

Logika ada di `core.Gateway` (tanpa jaringan, bisa diuji unit); `server` hanya lapisan asyncio.
"""
