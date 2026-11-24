"""Kotak deteksi kamera depan -> agen edge MINERVA (inbox lokal) -> server -> dashboard.

Dashboard memutar video asli kamera depan (H.264 kamera, 720p 30 fps, tanpa encode ulang) dan menggambar
kotak ini di atasnya. Jauh lebih halus daripada video beranotasi yang di-encode ulang (dibatasi laju deteksi
dan CPU Jetson), dan uplink 5G hanya membawa JSON kecil (~10 pesan/s).

Thread sendiri, slot terbaru saja: agen edge lambat / mati tidak pernah menahan deteksi.
"""
from __future__ import annotations

import json
import threading
import time
import urllib.error
import urllib.request


class DetsPoster:
    def __init__(self, url: str, timeout_s: float = 1.0, retry_s: float = 5.0, opener=None):
        self.url = url
        self.timeout_s = float(timeout_s)
        self.retry_s = float(retry_s)
        self._open = opener or urllib.request.urlopen
        self._slot: dict | None = None
        self._cv = threading.Condition()
        self._stop = False
        self._pause_until = 0.0
        self.sent = 0
        self.errors = 0
        self._thread = threading.Thread(target=self._run, name="dets-post", daemon=True)
        self._thread.start()

    def push(self, msg: dict) -> None:
        if time.monotonic() < self._pause_until:
            return                          # agen edge tidak menjawab: jangan antre, coba lagi nanti
        with self._cv:
            self._slot = msg
            self._cv.notify()

    def _run(self) -> None:
        while True:
            with self._cv:
                while self._slot is None and not self._stop:
                    self._cv.wait(0.5)
                if self._stop:
                    return
                msg, self._slot = self._slot, None
            req = urllib.request.Request(self.url, data=json.dumps(msg, separators=(",", ":")).encode(),
                                         headers={"content-type": "application/json"}, method="POST")
            try:
                with self._open(req, timeout=self.timeout_s) as r:
                    r.read()
                self.sent += 1
            except (urllib.error.URLError, OSError, ValueError) as e:
                self.errors += 1
                self._pause_until = time.monotonic() + self.retry_s
                if self.errors <= 3 or self.errors % 100 == 0:
                    print(f"[depan] kirim kotak ke {self.url} gagal ({self.errors}x): {e}", flush=True)

    def close(self, timeout_s: float = 2.0) -> None:
        with self._cv:
            self._stop = True
            self._cv.notify()
        self._thread.join(timeout_s)
