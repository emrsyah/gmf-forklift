"""Detektor + crop sebelum Face Landmarker: pemetaan koordinat, crop ditahan, cadangan full frame."""
import numpy as np

from src.cabin.face import FaceObs
from src.cabin.face_crop import CropFace, crop_square, square_crop_box

CFG = {"margin": 2.0, "out_px": 200, "min_face_px": 10, "hold_shift": 0.12, "hold_scale": 0.15}


class FakeDet:
    def __init__(self, boxes):
        self.boxes = boxes
        self.calls = 0

    def detect(self, img):
        self.calls += 1
        return list(self.boxes)


class FakeLandmarker:
    """Menemukan 'wajah' = piksel putih di gambar input; landmark = 4 sudut kotak putih."""

    def __init__(self):
        self.inputs = []

    def process(self, img, ts_ms, offset=(0, 0), scale_to=None):
        self.inputs.append(img.shape)
        g = img if img.ndim == 2 else img[..., 0]
        ys, xs = np.nonzero(g > 200)
        if len(xs) == 0:
            return FaceObs(present=False)
        xy = np.array([[xs.min(), ys.min()], [xs.max(), ys.min()], [xs.min(), ys.max()], [xs.max(), ys.max()]],
                      dtype=np.float32) + np.array(offset, dtype=np.float32)
        return FaceObs(True, xy, None, {}, None, 1.0)


def frame_with_face(x1, y1, x2, y2, w=960, h=720):
    img = np.zeros((h, w, 3), np.uint8)
    img[y1:y2, x1:x2] = 255
    return img


def test_landmarks_mapped_back_to_frame_coordinates():
    img = frame_with_face(400, 600, 460, 660)                 # wajah kecil di bawah
    lm = FakeLandmarker()
    cf = CropFace(lm, FakeDet([(400, 600, 460, 660, 0.9)]), CFG)
    obs = cf.process(img, 1000, offset=(5, 7))
    assert obs.present and cf.last_source == "detector"
    assert lm.inputs[0][:2] == (200, 200)                      # Face Landmarker melihat crop 200 px
    mn, mx = obs.landmarks_px.min(axis=0), obs.landmarks_px.max(axis=0)
    assert np.allclose(mn, [405, 607], atol=1.5) and np.allclose(mx, [464, 666], atol=1.5)


def test_face_cut_at_bottom_edge_is_padded():
    img = frame_with_face(400, 680, 470, 720)                  # dagu di luar gambar
    crop, x0, y0 = crop_square(img, 435, 700, 140, 140)
    assert crop.shape[:2] == (140, 140) and y0 + 140 > 720     # crop melewati tepi bawah
    assert crop[-1].max() == 0                                 # bagian luar gambar = hitam


def test_tracking_skips_detector_and_holds_crop():
    det = FakeDet([(400, 300, 460, 360, 0.9)])
    cf = CropFace(FakeLandmarker(), det, CFG)
    cf.process(frame_with_face(400, 300, 460, 360), 0)
    first = cf.last_crop_box
    cf.process(frame_with_face(403, 302, 463, 362), 33)       # geser 3 px
    assert det.calls == 1 and cf.last_source == "track"
    assert cf.last_crop_box == first                           # crop ditahan (pelacakan MediaPipe stabil)


def test_big_move_recenters_crop():
    det = FakeDet([(100, 300, 160, 360, 0.9)])
    cf = CropFace(FakeLandmarker(), det, CFG)
    cf.process(frame_with_face(100, 300, 160, 360), 0)
    first = cf.last_crop_box
    cf.process(frame_with_face(130, 300, 190, 360), 33)       # geser 30 px > 12% sisi crop
    # crop frame N memakai landmark frame N-1 (seperti pelacakan MediaPipe) -> pindah satu frame kemudian
    cf.process(frame_with_face(130, 300, 190, 360), 66)
    assert cf.last_crop_box != first
    assert abs((cf.last_crop_box[0] + cf.last_crop_box[2]) / 2 - 160) < 2


def test_lost_face_triggers_detector_again():
    det = FakeDet([(400, 300, 460, 360, 0.9)])
    cf = CropFace(FakeLandmarker(), det, CFG)
    cf.process(frame_with_face(400, 300, 460, 360), 0)
    assert not cf.process(np.zeros((720, 960, 3), np.uint8), 33).present
    cf.process(frame_with_face(400, 300, 460, 360), 66)
    assert det.calls == 2


def test_no_detection_falls_back_to_full_frame():
    full = FakeLandmarker()
    cf = CropFace(FakeLandmarker(), FakeDet([]), CFG, inner_full=full)
    obs = cf.process(frame_with_face(10, 10, 30, 30), 0)
    assert obs.present and cf.last_source == "full" and full.inputs[0][:2] == (720, 960)


def test_no_detection_without_fallback_is_absent():
    cf = CropFace(FakeLandmarker(), FakeDet([]), CFG)
    assert cf.process(frame_with_face(10, 10, 30, 30), 0).present is False


def test_tiny_detections_ignored():
    cf = CropFace(FakeLandmarker(), FakeDet([(10, 10, 15, 15, 0.99)]), CFG)
    assert cf.process(frame_with_face(10, 10, 15, 15), 0).present is False


def test_square_crop_box():
    cx, cy, side = square_crop_box((0, 0, 40, 80), 1.5)
    assert (cx, cy, side) == (20, 40, 120)
