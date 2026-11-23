"""Perekam H.264: segmen mulai di IDR + SPS/PPS, satu baris ts.csv per frame, resync setelah frame dibuang."""
import shutil
import subprocess
import threading
import time
from pathlib import Path

import cv2
import pytest

from src.capture.uvch264 import nal_types, nal_units
from src.io.h264_recorder import H264Recorder

SPS = b"\x00\x00\x00\x01\x67\x42\xc0\x1f"
PPS = b"\x00\x00\x00\x01\x68\xce\x3c\x80"


def idr(i=0):
    return b"\x00\x00\x00\x01\x65" + bytes([i % 251 + 1]) * 8


def p(i=0):
    return b"\x00\x00\x00\x01\x41" + bytes([i % 251 + 1]) * 4


class FakeSink:
    store: dict = {}

    def __init__(self, path, gate: threading.Event | None = None):
        self.path, self.gate = path, gate
        FakeSink.store[path] = bytearray()

    def write(self, data):
        if self.gate is not None:
            self.gate.wait(5)
        FakeSink.store[self.path] += data

    def close(self):
        pass


def _rec(tmp_path, **kw):
    cfg = {"dir": str(tmp_path), "segment_s": kw.pop("segment_s", 300), "min_free_gb": 0,
           "max_queue": kw.pop("max_queue", 60), "prefix": "front"}
    FakeSink.store = {}
    gate = kw.pop("gate", None)
    return H264Recorder(cfg, "t", sink_factory=lambda path: FakeSink(path, gate), **kw)


def _rows(path: Path) -> list[int]:
    lines = Path(str(path) + ".ts.csv").read_text().splitlines()
    assert lines[0] == "frame,ts_ms"
    return [int(x.split(",")[1]) for x in lines[1:]]


def test_waits_for_decodable_idr_and_writes_every_frame(tmp_path):
    r = _rec(tmp_path)
    t = 1_794_130_000_000
    r.push(p(1), t)                       # sebelum IDR: dibuang
    r.push(idr(2), t + 33)                # IDR tanpa SPS yang pernah dilihat: belum bisa didekode
    r.push(p(3), t + 66)
    r.push(SPS + PPS + idr(4), t + 100)   # segmen dimulai di sini
    for k in range(5):
        r.push(p(10 + k), t + 133 + 33 * k)
    r.close()
    assert len(r.segments) == 1
    data = bytes(FakeSink.store[r.segments[0]])
    assert data.startswith(SPS + PPS + idr(4))
    assert _rows(r.segments[0]) == [t + 100] + [t + 133 + 33 * k for k in range(5)]
    assert r.frames == 6


def test_new_segment_on_idr_after_segment_s_with_cached_sps(tmp_path):
    asked = []
    r = _rec(tmp_path, segment_s=1, on_segment=lambda: asked.append(1))
    t = 1_794_130_000_000
    au_ts = []
    for k in range(40):                   # 100 ms per frame, IDR (tanpa SPS) tiap 7 frame
        ts = t + 100 * k
        if k == 0:
            au = SPS + PPS + idr(k)
        elif k % 7 == 0:
            au = idr(k)
        else:
            au = p(k)
        r.push(au, ts)
        au_ts.append(ts)
    r.close()
    assert len(r.segments) >= 3
    for seg in r.segments:
        data = bytes(FakeSink.store[seg])
        assert [x for x, _, _ in nal_units(data)][:3] == [7, 8, 5]   # setiap segmen bisa didekode sendiri
    assert asked, "kamera diminta IDR saat segmen jatuh tempo"
    all_rows = [ts for seg in r.segments for ts in _rows(seg)]
    assert all_rows == au_ts                                          # tidak ada frame hilang/dobel


def test_queue_overflow_resyncs_at_next_idr(tmp_path):
    gate = threading.Event()
    r = _rec(tmp_path, max_queue=1, gate=gate)
    t = 1_794_130_000_000
    r.push(SPS + PPS + idr(0), t)         # diambil penulis, tertahan di write
    time.sleep(0.2)
    r.push(p(1), t + 33)                  # masuk antrean (kapasitas 1)
    r.push(p(2), t + 66)                  # penuh -> dibuang, resync
    r.push(p(3), t + 100)                 # bukan IDR -> dibuang
    gate.set()
    time.sleep(0.2)
    r.push(idr(4), t + 133)               # IDR -> rekaman lanjut
    time.sleep(0.2)                       # antrean berkapasitas 1: biarkan penulis mengambil IDR dulu
    r.push(p(5), t + 166)
    r.close()
    data = bytes(FakeSink.store[r.segments[0]])
    assert p(2) not in data and p(3) not in data
    assert _rows(r.segments[0]) == [t, t + 33, t + 133, t + 166]
    assert r.dropped == 2


def test_ffmpeg_cmd_rotation_old_and_new_style(monkeypatch):
    from src.io import h264_recorder as m
    monkeypatch.setitem(m._DISPLAY_ROTATION, "ff-old", False)
    monkeypatch.setitem(m._DISPLAY_ROTATION, "ff-new", True)
    old = m.ffmpeg_cmd("ff-old", Path("x.mp4"), 270)
    new = m.ffmpeg_cmd("ff-new", Path("x.mp4"), 270)
    assert "rotate=270" in old and "-display_rotation:v:0" not in old       # 4.4: tag searah jarum jam
    i = new.index("-display_rotation:v:0")
    assert new[i + 1] == "90" and i < new.index("-i")                       # baru: opsi input, berlawanan
    assert "-display_rotation:v:0" not in m.ffmpeg_cmd("ff-new", Path("x.mp4"), 0)


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg tidak ada")
def test_real_ffmpeg_mp4_frame_count_and_rotation(tmp_path):
    raw = tmp_path / "src.h264"
    subprocess.run(["ffmpeg", "-v", "error", "-y", "-f", "lavfi", "-i", "testsrc=size=640x360:rate=15",
                    "-frames:v", "45", "-pix_fmt", "yuv420p", "-c:v", "libx264", "-profile:v", "baseline", "-bf", "0",
                    "-x264-params", "keyint=15:min-keyint=15:aud=1:scenecut=0", "-f", "h264", str(raw)],
                   check=True)
    data = raw.read_bytes()
    # pisah per access unit di setiap AUD (NAL 9)
    starts = [s for typ, s, _ in nal_units(data) if typ == 9] + [len(data)]
    aus = [data[a:b] for a, b in zip(starts, starts[1:])]
    assert len(aus) == 45 and 5 in nal_types(aus[0])
    cfg = {"dir": str(tmp_path / "rec"), "segment_s": 300, "min_free_gb": 0, "prefix": "front"}
    r = H264Recorder(cfg, "t", rotate=270)
    t = int(time.time() * 1000)
    for k, au in enumerate(aus):
        r.push(au, t + 66 * k)
        time.sleep(1 / 30)
    r.close()
    assert r.stopped_reason is None and len(r.segments) == 1
    seg = r.segments[0]
    cap = cv2.VideoCapture(str(seg))
    n, shape, first = 0, None, None
    while True:
        ok, img = cap.read()
        if not ok:
            break
        n, shape = n + 1, img.shape
        first = img if first is None else first
    cap.release()
    assert n == len(_rows(seg)) == 45          # gerbang replay: frame video = baris ts.csv
    assert shape[:2] == (640, 360)             # metadata rotasi -> potret seperti tampilan langsung
    # arah putaran harus sama dengan front_preview.rotate(img, 270) yang dipakai tampilan langsung
    from src.io.front_preview import rotate
    ok, src0 = cv2.VideoCapture(str(raw)).read()
    want = rotate(src0, 270)
    assert cv2.absdiff(first, want).mean() < 5
    assert cv2.absdiff(first, rotate(src0, 90)).mean() > 20
