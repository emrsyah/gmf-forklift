"""Tampilan browser: halaman, stream MJPEG, dan tombol hotkey masuk ke antrean."""
import json
import queue
import socket
import time
import urllib.request

import cv2
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

        img = np.full((120, 160, 3), 128, np.uint8)
        v.publish(img)
        assert v._jpg["cabin"] is None                        # tanpa penonton: tidak di-encode
        r = urllib.request.urlopen(base + "/stream", timeout=5)
        t0 = time.monotonic()
        while v.viewers("cabin") == 0 and time.monotonic() - t0 < 5:
            time.sleep(0.01)
        assert v.wants("cabin")
        v.publish(img)
        chunk = r.read(400)
        assert b"--frame" in chunk and b"\xff\xd8" in chunk   # header MJPEG + awal JPEG
        r.close()
        t0 = time.monotonic()
        while v.viewers("cabin") > 0 and time.monotonic() - t0 < 5:
            v.publish(img)                                    # server baru tahu putus saat menulis
            time.sleep(0.05)
        assert v.viewers("cabin") == 0
        assert v._jpg["cabin"] is None                        # penonton berikutnya tidak dapat frame basi

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

        assert json.loads(urllib.request.urlopen(base + "/status", timeout=5).read()) == {}
        v.status = {"calibration": "terkalibrasi", "fps": 21.0}
        st = json.loads(urllib.request.urlopen(base + "/status", timeout=5).read())
        assert st == {"calibration": "terkalibrasi", "fps": 21.0}
    finally:
        v.close()


def _wait_viewers(v, ch, n):
    t0 = time.monotonic()
    while v.viewers(ch) != n and time.monotonic() - t0 < 5:
        time.sleep(0.01)
    assert v.viewers(ch) == n


def test_agent_request_limits_rate_and_width():
    port = _free_port()
    v = WebView(port, queue.Queue(), fps=10, max_side=960, host="127.0.0.1")
    try:
        assert v.parse_request("fps=2&w=320") == (2.0, 320)
        assert v.parse_request("fps=50") == (10.0, None)          # tidak bisa melebihi fps bawaan
        assert v.parse_request("w=2000&fps=x") == (10.0, None)    # lebih lebar dari max_side -> bawaan
        img = np.full((720, 960, 3), 90, np.uint8)
        base = f"http://127.0.0.1:{port}"
        agent = urllib.request.urlopen(base + "/stream?fps=2&w=320", timeout=5)
        _wait_viewers(v, "cabin", 1)
        v.publish(img)
        jpg = v._jpg["cabin"]
        assert cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR).shape[1] == 320
        assert not v.wants("cabin")                                # 2 fps: frame berikutnya baru 0,5 s lagi
        time.sleep(0.15)
        assert not v.wants("cabin")                                # (10 fps bawaan sudah boleh di sini)
        browser = urllib.request.urlopen(base + "/stream", timeout=5)
        _wait_viewers(v, "cabin", 2)
        assert v.wants("cabin")                                    # penonton browser: 10 fps, 960 px
        v.publish(img)
        assert cv2.imdecode(np.frombuffer(v._jpg["cabin"], np.uint8), cv2.IMREAD_COLOR).shape[1] == 960
        agent.close()
        browser.close()
    finally:
        v.close()
