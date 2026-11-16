"""Tampilan browser: halaman, stream MJPEG, dan tombol hotkey masuk ke antrean."""
import queue
import socket
import urllib.request

import numpy as np

from src.io.web_view import WebView


def _free_port() -> int:
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


def test_page_stream_and_keys():
    q: queue.Queue = queue.Queue()
    port = _free_port()
    v = WebView(port, q, title="uji", fps=0, max_side=64, host="127.0.0.1")
    try:
        base = f"http://127.0.0.1:{port}"
        html = urllib.request.urlopen(base + "/", timeout=5).read().decode()
        assert "/stream" in html and "Kalibrasi ulang" in html

        v.publish(np.full((120, 160, 3), 128, np.uint8))
        r = urllib.request.urlopen(base + "/stream", timeout=5)
        chunk = r.read(400)
        assert b"--frame" in chunk and b"\xff\xd8" in chunk   # header MJPEG + awal JPEG
        r.close()

        req = urllib.request.Request(base + "/key?k=c", method="POST")
        assert urllib.request.urlopen(req, timeout=5).status == 204
        assert q.get(timeout=2) == "c"

        bad = urllib.request.Request(base + "/key?k=x", method="POST")
        try:
            urllib.request.urlopen(bad, timeout=5)
            assert False, "kunci tak dikenal harus ditolak"
        except urllib.error.HTTPError as e:
            assert e.code == 400
        assert q.empty()
    finally:
        v.close()
