"""Tampilan overlay lewat browser (MJPEG over HTTP) + tombol hotkey.

Untuk perangkat tanpa monitor/keyboard (Jetson di forklift, atau Jetson yang dikendalikan lewat SSH):
  python -m src.main --overlay configs/jetson.yaml --web 8090
lalu buka http://IP-JETSON:8090 di browser laptop (satu jaringan).

Hanya untuk uji/pengembangan di jaringan lokal: tanpa autentikasi, dan menampilkan wajah pengemudi.
Jangan diekspos ke internet (kebijakan privasi klip wajah = pertanyaan terbuka #5 di CLAUDE.md).
"""
from __future__ import annotations

import queue
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from urllib.parse import parse_qs, urlparse

import cv2
import numpy as np

from src.preprocess import resize_max_side

KEYS = {"c": "Kalibrasi ulang", "d": "Panel debug", "p": "Baseline populasi", "q": "Stop program"}

PAGE = """<!doctype html><html><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width,initial-scale=1"><title>{title}</title>
<style>
 body{{margin:0;background:#111;color:#eee;font-family:system-ui,sans-serif}}
 header{{display:flex;gap:8px;align-items:center;flex-wrap:wrap;padding:8px 12px;background:#1d1d1d}}
 h1{{font-size:15px;margin:0 12px 0 0;font-weight:600}}
 button{{background:#2d2d2d;color:#eee;border:1px solid #444;border-radius:6px;padding:7px 12px;
        font-size:14px;cursor:pointer}}
 button:hover{{background:#3a3a3a}} button.stop{{border-color:#a33}}
 #msg{{font-size:13px;color:#9c9}}
 main{{display:flex;justify-content:center;gap:8px;flex-wrap:wrap}}
 figure{{margin:0;flex:1 1 420px;max-width:{fig_max}}} figcaption{{font-size:13px;color:#aaa;padding:4px 0}}
 img{{width:100%;max-height:calc(100vh - 80px);object-fit:contain;background:#000}}
</style></head><body>
<header><h1>{title}</h1>{buttons}<span id="msg"></span></header>
<main>{figures}</main>
<script>
async function key(k,label){{
  if(k==='q' && !confirm('Hentikan program di perangkat?')) return;
  const r = await fetch('/key?k='+k,{{method:'POST'}});
  document.getElementById('msg').textContent = r.ok ? 'terkirim: '+label : 'gagal';
}}
</script></body></html>"""


class WebView:
    """Server MJPEG di thread latar. `publish(img)` dari loop utama; hotkey masuk ke `keys`."""

    def __init__(self, port: int, keys: queue.Queue, title: str = "Forklift Safety AI",
                 fps: float = 10.0, max_side: int | None = 960, jpeg_quality: int = 75, host: str = "0.0.0.0"):
        self.keys = keys
        self.title = title
        self.min_dt = 1.0 / fps if fps and fps > 0 else 0.0
        self.max_side = max_side
        self.quality = int(jpeg_quality)
        # kanal "cabin" = /stream (dipakai juga agen MINERVA), kanal lain = /stream/<nama>
        self._jpg: dict[str, bytes | None] = {"cabin": None}
        self._seq: dict[str, int] = {"cabin": 0}
        self._last_pub: dict[str, float] = {}
        self._labels: dict[str, str] = {"cabin": "Kamera kabin"}
        self._viewers: dict[str, int] = {}     # koneksi stream aktif per kanal; 0 -> tidak perlu encode JPEG
        self._cond = threading.Condition()
        view = self

        class Handler(BaseHTTPRequestHandler):
            def log_message(self, *a):   # jangan banjiri terminal
                pass

            def do_GET(self):
                path = urlparse(self.path).path
                if path == "/":
                    buttons = "".join(
                        f'<button class="{"stop" if k == "q" else ""}" onclick="key(\'{k}\',\'{v}\')">{v}</button>'
                        for k, v in KEYS.items())
                    chans = list(view._labels)
                    figures = "".join(
                        f'<figure><figcaption>{view._labels[c]}</figcaption>'
                        f'<img src="{view.stream_path(c)}" alt="{view._labels[c]}"></figure>' for c in chans)
                    body = PAGE.format(title=view.title, buttons=buttons, figures=figures,
                                       fig_max="100%" if len(chans) == 1 else "50%").encode()
                    self.send_response(200)
                    self.send_header("Content-Type", "text/html; charset=utf-8")
                    self.send_header("Content-Length", str(len(body)))
                    self.end_headers()
                    self.wfile.write(body)
                elif view.channel_for(path) is not None:
                    ch = view.channel_for(path)
                    self.send_response(200)
                    self.send_header("Cache-Control", "no-cache")
                    self.send_header("Content-Type", "multipart/x-mixed-replace; boundary=frame")
                    self.end_headers()
                    seen = -1
                    with view._cond:
                        view._viewers[ch] = view._viewers.get(ch, 0) + 1
                    try:
                        while True:
                            with view._cond:
                                view._cond.wait_for(lambda: view._seq[ch] != seen, timeout=5.0)
                                jpg, seen = view._jpg[ch], view._seq[ch]
                            if jpg is None:
                                continue
                            self.wfile.write(b"--frame\r\nContent-Type: image/jpeg\r\n")
                            self.wfile.write(f"Content-Length: {len(jpg)}\r\n\r\n".encode())
                            self.wfile.write(jpg)
                            self.wfile.write(b"\r\n")
                    except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                        return
                    finally:
                        with view._cond:
                            view._viewers[ch] = view._viewers.get(ch, 1) - 1
                            if view._viewers[ch] <= 0:
                                # tanpa penonton frame tidak diperbarui: jangan sajikan gambar basi ke penonton baru
                                view._jpg[ch] = None
                else:
                    self.send_error(404)

            def do_POST(self):
                u = urlparse(self.path)
                k = parse_qs(u.query).get("k", [""])[0]
                if u.path == "/key" and k in KEYS:
                    view.keys.put(k)
                    self.send_response(204)
                    self.end_headers()
                else:
                    self.send_error(400)

        self.server = ThreadingHTTPServer((host, int(port)), Handler)
        self.server.daemon_threads = True
        threading.Thread(target=self.server.serve_forever, daemon=True).start()

    @staticmethod
    def stream_path(channel: str) -> str:
        return "/stream" if channel == "cabin" else f"/stream/{channel}"

    def channel_for(self, path: str) -> str | None:
        for c in self._labels:
            if path == self.stream_path(c):
                return c
        return None

    def add_channel(self, channel: str, label: str) -> None:
        with self._cond:
            self._labels[channel] = label
            self._jpg.setdefault(channel, None)
            self._seq.setdefault(channel, 0)

    def viewers(self, channel: str = "cabin") -> int:
        return self._viewers.get(channel, 0)

    def wants(self, channel: str = "cabin") -> bool:
        """True jika frame berikutnya akan dikirim: ada penonton dan jeda `fps` sudah lewat.
        Pemanggil bisa melewati kerja mahal (dekode/rotasi/overlay) kalau False."""
        return (self._viewers.get(channel, 0) > 0
                and time.monotonic() - self._last_pub.get(channel, 0.0) >= self.min_dt)

    def publish(self, img: np.ndarray | None, channel: str = "cabin") -> None:
        if img is None or not self.wants(channel):    # tanpa penonton: jangan encode JPEG
            return
        self._last_pub[channel] = time.monotonic()
        small = resize_max_side(img, self.max_side) if self.max_side else img
        ok, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, self.quality])
        if not ok:
            return
        with self._cond:
            if channel not in self._labels:
                self._labels[channel] = channel
            self._jpg[channel] = buf.tobytes()
            self._seq[channel] = self._seq.get(channel, 0) + 1
            self._cond.notify_all()

    def close(self) -> None:
        self.server.shutdown()
        self.server.server_close()
