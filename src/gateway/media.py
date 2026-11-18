"""Server HTTP read-only untuk bukti kejadian: GET /media/<path> di bawah media root, hanya .jpg/.mp4.

Mendukung Range (browser memakai Range untuk memutar dan menggeser <video>). Tanpa listing direktori.
Hanya jaringan lokal/5G privat, tanpa login (sama dengan WebSocket dashboard).
"""
from __future__ import annotations

import re
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from urllib.parse import unquote, urlsplit

TYPES = {".jpg": "image/jpeg", ".jpeg": "image/jpeg", ".mp4": "video/mp4"}
_RANGE = re.compile(r"bytes=(\d*)-(\d*)$")
CHUNK = 256 * 1024


def resolve_media(root: Path, url_path: str) -> Path | None:
    """URL -> file di bawah root, atau None (path traversal, tipe lain, tidak ada)."""
    path = unquote(urlsplit(url_path).path)
    if not path.startswith("/media/"):
        return None
    try:
        root_r = root.resolve()
        f = (root_r / path[len("/media/"):]).resolve()
        f.relative_to(root_r)
    except (ValueError, OSError):
        return None
    if f.suffix.lower() not in TYPES or not f.is_file():
        return None
    return f


def make_handler(root: Path):
    class Handler(BaseHTTPRequestHandler):
        server_version = "forklift-media/1"

        def log_message(self, *args):          # tanpa log per request di terminal
            pass

        def _headers(self, code: int, ctype: str | None, length: int, extra: dict | None = None):
            self.send_response(code)
            if ctype:
                self.send_header("Content-Type", ctype)
            self.send_header("Content-Length", str(length))
            self.send_header("Accept-Ranges", "bytes")
            self.send_header("Access-Control-Allow-Origin", "*")
            # 404/416 jangan di-cache: file bisa muncul sebentar lagi (klip ditulis setelah kondisi selesai)
            self.send_header("Cache-Control", "private, max-age=3600" if code in (200, 206) else "no-store")
            for k, v in (extra or {}).items():
                self.send_header(k, v)
            self.end_headers()

        def _serve(self, body: bool):
            f = resolve_media(root, self.path)
            if f is None:
                self._headers(404, "text/plain", 0)
                return
            size = f.stat().st_size
            ctype = TYPES[f.suffix.lower()]
            start, end, code, extra = 0, size - 1, 200, {}
            rng = self.headers.get("Range")
            if rng:
                m = _RANGE.match(rng.strip())
                if not m or (not m.group(1) and not m.group(2)):
                    self._headers(416, None, 0, {"Content-Range": f"bytes */{size}"})
                    return
                if m.group(1):
                    start = int(m.group(1))
                    end = min(int(m.group(2)), size - 1) if m.group(2) else size - 1
                else:                            # bytes=-N (N byte terakhir)
                    start = max(size - int(m.group(2)), 0)
                if start > end or start >= size:
                    self._headers(416, None, 0, {"Content-Range": f"bytes */{size}"})
                    return
                code, extra = 206, {"Content-Range": f"bytes {start}-{end}/{size}"}
            self._headers(code, ctype, end - start + 1, extra)
            if not body:
                return
            with open(f, "rb") as fh:
                fh.seek(start)
                left = end - start + 1
                while left > 0:
                    chunk = fh.read(min(CHUNK, left))
                    if not chunk:
                        break
                    try:
                        self.wfile.write(chunk)
                    except (BrokenPipeError, ConnectionResetError, ConnectionAbortedError):
                        return
                    left -= len(chunk)

        def do_GET(self):
            self._serve(True)

        def do_HEAD(self):
            self._serve(False)

    return Handler


def start_media_server(root: Path, host: str, port: int) -> ThreadingHTTPServer:
    httpd = ThreadingHTTPServer((host, int(port)), make_handler(Path(root)))
    httpd.daemon_threads = True
    threading.Thread(target=httpd.serve_forever, daemon=True, name="media-http").start()
    return httpd
