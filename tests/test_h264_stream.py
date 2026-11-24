"""Publisher RTSP H.264: mulai di IDR + SPS/PPS, sambung ulang setelah putus, uji nyata dengan ffmpeg."""
import shutil
import socket
import subprocess
import time

import cv2
import pytest

from src.capture.uvch264 import nal_types, nal_units
from src.io.h264_stream import H264RtspPublisher

SPS = b"\x00\x00\x00\x01\x67\x42\xc0\x1f"
PPS = b"\x00\x00\x00\x01\x68\xce\x3c\x80"


def idr(i=0):
    return b"\x00\x00\x00\x01\x65" + bytes([i % 251 + 1]) * 8


def p(i=0):
    return b"\x00\x00\x00\x01\x41" + bytes([i % 251 + 1]) * 4


class FakeConn:
    made: list = []

    def __init__(self, fail_after=None):
        self.data = bytearray()
        self.fail_after = fail_after
        self.writes = 0
        FakeConn.made.append(self)

    def alive(self):
        return True

    def write(self, b):
        if self.fail_after is not None and self.writes >= self.fail_after:
            raise BrokenPipeError("putus")
        self.writes += 1
        self.data += b

    def close(self):
        pass


def _wait(cond, t=3.0):
    t0 = time.monotonic()
    while not cond() and time.monotonic() - t0 < t:
        time.sleep(0.01)


def test_starts_at_idr_and_reconnects_with_cached_sps():
    FakeConn.made = []
    plan = iter([2, None])                   # sambungan 1 putus setelah 2 tulis
    asked = []
    pub = H264RtspPublisher({"retry_s": 0}, connect=lambda: FakeConn(next(plan)),
                            on_restart=lambda: asked.append(1))
    pub.push(p(0), 0)                        # sebelum IDR: dibuang
    pub.push(SPS + PPS + idr(1), 33)
    pub.push(p(2), 66)
    pub.push(p(3), 100)                      # tulis ke-3 gagal -> putus
    pub.push(p(4), 133)                      # menunggu IDR
    pub.push(idr(5), 166)                    # IDR tanpa SPS -> diawali SPS/PPS tersimpan
    pub.push(p(6), 200)
    pub.close()
    first, second = FakeConn.made
    assert bytes(first.data) == SPS + PPS + idr(1) + p(2)
    assert bytes(second.data) == SPS + PPS + idr(5) + p(6)
    assert pub.connects == 2 and asked, "kamera diminta IDR setelah putus"


def test_on_demand_pushes_only_while_leased_and_restarts_at_idr():
    FakeConn.made = []
    asked = []
    pub = H264RtspPublisher({"retry_s": 0, "on_demand": True}, connect=lambda: FakeConn(),
                            on_restart=lambda: asked.append(1))
    pub.push(SPS + PPS + idr(0), 0)          # tanpa sewa: tidak ada sambungan
    _wait(lambda: pub.idle == 1)
    assert FakeConn.made == []
    pub.want(True, 5)
    pub.push(p(1), 33)                       # disewa, tapi bukan IDR: menunggu
    pub.push(SPS + PPS + idr(2), 66)
    pub.push(p(3), 100)
    _wait(lambda: pub.sent == 2)
    pub.want(False)                          # sewa dihentikan -> sambungan ditutup
    pub.push(p(4), 133)
    _wait(lambda: pub.idle == 2)
    assert not pub.connected
    pub.want(True, 5)
    pub.push(p(5), 166)
    pub.push(idr(6), 200)                    # mulai lagi di IDR, SPS/PPS dari cache
    pub.close()
    first, second = FakeConn.made
    assert bytes(first.data) == SPS + PPS + idr(2) + p(3)
    assert bytes(second.data) == SPS + PPS + idr(6)
    assert asked, "kamera diminta IDR saat mulai lagi"


def test_lease_expires():
    pub = H264RtspPublisher({"retry_s": 0, "on_demand": True}, connect=lambda: FakeConn())
    pub.want(True, 0.05)
    assert pub.wanted()
    time.sleep(0.08)
    assert not pub.wanted()
    assert H264RtspPublisher({"retry_s": 0}, connect=lambda: FakeConn()).wanted()


def test_no_lease_news_fails_open():
    # agen mati / agen lama: tanpa kabar sewa selama lease_orphan_s -> dorong terus
    pub = H264RtspPublisher({"retry_s": 0, "on_demand": True, "lease_orphan_s": 0.05}, connect=lambda: FakeConn())
    assert not pub.wanted()
    time.sleep(0.08)
    assert pub.wanted()
    pub.want(False)                              # kabar baru (off) -> berhenti lagi
    assert not pub.wanted()
    pub.close()


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg tidak ada")
def test_real_rtsp_roundtrip(tmp_path):
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    port = s.getsockname()[1]
    s.close()
    raw = tmp_path / "src.h264"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc=size=320x240:rate=15",
                    "-frames:v", "60", "-pix_fmt", "yuv420p", "-c:v", "libx264", "-profile:v", "baseline",
                    "-x264-params", "keyint=15:min-keyint=15:aud=1:scenecut=0", "-f", "h264", str(raw)],
                   check=True)
    data = raw.read_bytes()
    starts = [st for typ, st, _ in nal_units(data) if typ == 9] + [len(data)]
    aus = [data[a:b] for a, b in zip(starts, starts[1:])]
    out = tmp_path / "got.mkv"
    url = f"rtsp://127.0.0.1:{port}/front"
    # ffmpeg sebagai server RTSP penerima (pengganti MediaMTX di uji)
    # -listen_timeout (detik) wajib di ffmpeg baru: default -1 dikali 1000 -> di luar rentang (bug ffmpeg)
    rx = subprocess.Popen(["ffmpeg", "-v", "error", "-y", "-rtsp_flags", "listen", "-listen_timeout", "30",
                           "-i", url, "-c", "copy", str(out)], stderr=subprocess.PIPE)
    time.sleep(1.0)
    pub = H264RtspPublisher({"url": url, "retry_s": 0.5})
    for k, au in enumerate(aus):
        pub.push(au, 66 * k, 5 in nal_types(au))
        time.sleep(1 / 15)
    pub.close()
    try:
        rx.wait(20)
    except subprocess.TimeoutExpired:
        rx.kill()
        pytest.fail("penerima RTSP tidak selesai")
    assert pub.sent == len(aus), (pub.sent, pub.dropped)
    cap = cv2.VideoCapture(str(out))
    n = 0
    while cap.read()[0]:
        n += 1
    cap.release()
    assert n >= len(aus) - 2                  # frame sampai dan bisa didekode (bukan cuma sebagian)
