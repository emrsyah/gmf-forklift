"""Deteksi depan di Jetson: ByteTrack di atas backend tanpa .track() + worker di thread pratinjau."""
import time

import numpy as np
import pytest

from src.backends import Detection

pytest.importorskip("ultralytics")

from src.backends.tracking import ByteTrackDetector  # noqa: E402
from src.capture import Frame  # noqa: E402
from src.io.front_preview import FrontDetectWorker  # noqa: E402


class _MovingPerson:
    """Detektor palsu: satu orang bergeser 5 px per frame + satu mobil (kelas lain)."""
    names = ["person", "bicycle", "car"]

    def __init__(self):
        self.i = 0

    def predict(self, img, conf=0.25):
        x = 100 + 5 * self.i
        self.i += 1
        return [Detection(0, "person", 0.9, (x, 100, x + 50, 250)),
                Detection(2, "car", 0.8, (400, 300, 600, 420))]


def test_bytetrack_keeps_one_id_for_moving_person():
    bt = ByteTrackDetector(_MovingPerson(), "bytetrack.yaml", frame_rate=10)
    img = np.zeros((720, 720, 3), np.uint8)
    ids = []
    for _ in range(15):
        out = bt.track(img, conf=0.25, classes=[0])
        assert all(d.cls_name == "person" for d in out)       # filter kelas sebelum tracking
        ids += [d.track_id for d in out]
    assert ids and len(set(ids)) == 1
    last = out[0].xyxy
    assert abs(last[0] - (100 + 5 * 14)) < 6                   # kotak mengikuti deteksi terakhir


def test_bytetrack_empty_frames_and_reset():
    class _None:
        names = ["person"]

        def predict(self, img, conf=0.25):
            return []

    bt = ByteTrackDetector(_None(), "bytetrack.yaml", frame_rate=10)
    img = np.zeros((10, 10, 3), np.uint8)
    assert bt.track(img) == []
    bt.reset()
    assert bt.track(img) == []


class _Web:
    def __init__(self):
        self.pub = []

    def wants(self, ch):
        return True

    def publish(self, img, ch):
        self.pub.append((img.shape, ch))


class _Pipe:
    def __init__(self, fail=False):
        self.fail = fail
        self.n = 0

    def process(self, frame):
        self.n += 1
        if self.fail:
            raise RuntimeError("boom")
        from src.front.pipeline import FrontResult
        return FrontResult(frame.ts_ms, frame.index)

    def draw(self, img, res, fps=None):
        return img


def _wait(cond, s=3.0):
    t = time.monotonic() + s
    while not cond() and time.monotonic() < t:
        time.sleep(0.02)
    return cond()


def test_worker_rotates_detects_and_publishes():
    web, pipe = _Web(), _Pipe()
    w = FrontDetectWorker(pipe, fps=50, rotate_deg=270, web=web)
    try:
        w.offer(Frame(np.zeros((720, 1280, 3), np.uint8), 1000, 1))
        assert _wait(lambda: web.pub)
        assert web.pub[0] == ((1280, 720, 3), "front")          # diputar 270 sebelum deteksi
        assert w.last is not None and w.last.ts_ms == 1000
    finally:
        w.close()


def test_worker_survives_model_errors():
    web, pipe = _Web(), _Pipe(fail=True)
    w = FrontDetectWorker(pipe, fps=50, web=web)
    try:
        for i in range(3):
            w.offer(Frame(np.zeros((8, 8, 3), np.uint8), i, i))
            time.sleep(0.05)
        assert _wait(lambda: w.errors >= 2)
        assert w._thread.is_alive() and not web.pub
    finally:
        w.close()


def test_worker_caps_rate():
    web, pipe = _Web(), _Pipe()
    w = FrontDetectWorker(pipe, fps=5, web=web)
    try:
        t_end = time.monotonic() + 1.0
        i = 0
        while time.monotonic() < t_end:                        # kamera 100 fps
            w.offer(Frame(np.zeros((8, 8, 3), np.uint8), i, i))
            i += 1
            time.sleep(0.01)
        assert 3 <= pipe.n <= 7                                 # ~5 per detik, frame lain dilewati
    finally:
        w.close()
