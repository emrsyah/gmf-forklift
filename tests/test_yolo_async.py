"""YOLO async (frame tidak menunggu YOLO) dan pencarian wajah yang dijarangkan saat wajah hilang."""
import threading

import numpy as np

from src.backends import Detection, Detector
from src.cabin.face import FaceObs
from src.cabin.face_crop import CropFace
from src.cabin.pipeline import CabinPipeline
from src.capture import Frame
from src.config import Config


class FakeFace:
    def process(self, img, ts, off=(0, 0)):
        return FaceObs(present=False)


class SlowDetector(Detector):
    """predict() menunggu `gate` -> bisa dibuktikan frame tidak ikut menunggu."""
    names = ["phone", "helmet", "head_no_helmet", "vest", "torso"]

    def __init__(self):
        self.gate = threading.Event()
        self.calls = 0

    def predict(self, image_bgr, conf=0.25):
        self.calls += 1
        assert self.gate.wait(5)
        h, w = image_bgr.shape[:2]
        return [Detection(0, "phone", 0.9, (w * 0.4, h * 0.4, w * 0.6, h * 0.6))]


def _pipe(max_fps=0):
    cfg = Config.load()
    y = cfg.runtime["cabin"]["yolo"]
    y["async"], y["async_max_fps"] = True, max_fps
    cfg.runtime["cabin"]["face"]["crop"]["enabled"] = False
    det = SlowDetector()
    return CabinPipeline(cfg.runtime, cfg.thresholds, cfg.events, face_model=FakeFace(), detector=det), det


def _frame(i):
    return Frame(np.full((120, 160, 3), 100, np.uint8), 1000 + i * 33, i)


def test_frames_do_not_wait_for_yolo_and_result_arrives_later():
    p, det = _pipe()
    st = [p.process(_frame(i)) for i in range(5)]       # YOLO tertahan: frame tetap jalan
    assert all(s.yolo is None for s in st) and det.calls == 1   # satu job, tidak ditumpuk
    det.gate.set()
    p._yolo_pending.result(timeout=5)
    s = p.process(_frame(5))
    assert s.yolo is not None and s.yolo.dets[0].cls_name == "phone"   # filter ponsel butuh hit berulang
    assert "yolo" in s.timings_ms


def test_async_max_fps_limits_submissions():
    p, det = _pipe(max_fps=5)                           # >= 200 ms antar job
    det.gate.set()
    for i in range(30):                                 # 30 frame x 33 ms ~ 1 s
        p.process(_frame(i))
        if p._yolo_pending is not None:
            p._yolo_pending.result(timeout=5)
    assert 4 <= det.calls <= 6


class FailingDetector(SlowDetector):
    def predict(self, image_bgr, conf=0.25):
        self.calls += 1
        raise RuntimeError("cuda error")


def test_worker_exception_does_not_kill_loop():
    cfg = Config.load()
    y = cfg.runtime["cabin"]["yolo"]
    y["async"], y["async_max_fps"] = True, 0
    cfg.runtime["cabin"]["face"]["crop"]["enabled"] = False
    det = FailingDetector()
    p = CabinPipeline(cfg.runtime, cfg.thresholds, cfg.events, face_model=FakeFace(), detector=det)
    for i in range(6):
        st = p.process(_frame(i))
        if p._yolo_pending is not None:
            p._yolo_pending.exception(timeout=5)
    assert st.yolo is None and det.calls >= 2 and p.yolo_status.startswith("error")


class CountDet:
    def __init__(self):
        self.calls = 0

    def detect(self, img):
        self.calls += 1
        return []


def test_lost_face_search_throttled_after_grace():
    det = CountDet()
    cf = CropFace(FakeFace(), det, {"lost_search_every": 3, "lost_search_after_s": 0.5})
    img = np.zeros((120, 160, 3), np.uint8)
    for i in range(16):                                 # 33 ms/frame: 0..495 ms tiap frame dicari (16 frame)
        assert cf.process(img, i * 33).present is False
    assert det.calls == 16
    for i in range(16, 25):                             # sesudah 0,5 s: tiap 3 frame
        cf.process(img, i * 33)
    assert det.calls == 16 + 3


def test_lost_search_default_every_frame():
    det = CountDet()
    cf = CropFace(FakeFace(), det, {})
    img = np.zeros((120, 160, 3), np.uint8)
    for i in range(5):
        cf.process(img, i * 33)
    assert det.calls == 5
