"""YOLO paralel dengan MediaPipe dan sumber GRAY8 harus memberi hasil identik dengan jalur lama."""
import threading

import cv2
import numpy as np
import pytest

from src.backends import Detection, Detector
from src.cabin.face import FaceObs
from src.cabin.pipeline import CabinPipeline
from src.capture import Frame, GStreamerSource
from src.config import Config


class FakeFace:
    def __init__(self):
        self.shapes = []

    def process(self, img, ts, off=(0, 0)):
        self.shapes.append(img.shape)
        return FaceObs(present=False)


class SpyDetector(Detector):
    names = ["phone", "helmet", "head_no_helmet", "vest", "torso"]

    def __init__(self):
        self.inputs, self.threads = [], set()

    def predict(self, image_bgr, conf=0.25):
        self.inputs.append(image_bgr.copy())
        self.threads.add(threading.current_thread().name)
        h, w = image_bgr.shape[:2]
        return [Detection(0, "phone", 0.9, (w * 0.4, h * 0.4, w * 0.6, h * 0.6))]


def _run(parallel: bool, frames):
    cfg = Config.load()
    cfg.runtime["cabin"]["yolo"]["parallel"] = parallel
    cfg.runtime["cabin"]["yolo"]["every_n_frames"] = 1
    spy = SpyDetector()
    p = CabinPipeline(cfg.runtime, cfg.thresholds, cfg.events, face_model=FakeFace(), detector=spy)
    out = [p.process(f) for f in frames]
    return out, spy


def test_parallel_yolo_matches_sequential():
    rng = np.random.default_rng(1)
    frames = [Frame(rng.integers(0, 255, (120, 160, 3), np.uint8), 1000 + i * 50, i) for i in range(6)]
    seq, s1 = _run(False, frames)
    par, s2 = _run(True, frames)
    assert all(np.array_equal(a, b) for a, b in zip(s1.inputs, s2.inputs))
    assert [o.yolo.phone_conf for o in seq] == [o.yolo.phone_conf for o in par]
    assert [o.active for o in seq] == [o.active for o in par]
    assert any(t.startswith("yolo") for t in s2.threads)              # benar-benar di thread pekerja
    assert "yolo_wait" in par[0].timings_ms and "yolo" in par[0].timings_ms


@pytest.mark.parametrize("simulate_nir", [True, False])
def test_gray_frames_reach_models_as_3_channels(simulate_nir):
    cfg = Config.load()
    cfg.runtime["cabin"]["simulate_nir"] = simulate_nir
    face, spy = FakeFace(), SpyDetector()
    p = CabinPipeline(cfg.runtime, cfg.thresholds, cfg.events, face_model=face, detector=spy)
    gray = np.random.default_rng(2).integers(0, 255, (120, 160), np.uint8)
    st = p.process(Frame(gray, 1000, 0))
    assert face.shapes[0][2] == 3 and spy.inputs[0].shape[2] == 3 and st.image.ndim == 3
    # sama dengan jalur BGR lama: abu-abu 3 channel yang identik
    st_bgr = CabinPipeline(cfg.runtime, cfg.thresholds, cfg.events, face_model=FakeFace(),
                           detector=SpyDetector()).process(Frame(cv2.cvtColor(gray, cv2.COLOR_GRAY2BGR), 1000, 0))
    assert np.array_equal(st.image, st_bgr.image)


def test_gstreamer_gray_limited_range_is_stretched(monkeypatch):
    class FakeCap:
        def __init__(self, *a):
            pass

        def isOpened(self):  # noqa: N802
            return True

        def get(self, *_):
            return 30.0

        def read(self):
            return True, np.array([[16, 126, 235, 0, 255]], np.uint8)

        def release(self):
            pass

    monkeypatch.setattr(cv2, "VideoCapture", FakeCap)
    f = GStreamerSource("fake ! appsink", gray_range="limited").read()
    assert f.image.tolist() == [[0, 128, 255, 0, 255]]
    assert GStreamerSource("fake ! appsink").read().image.tolist() == [[16, 126, 235, 0, 255]]
