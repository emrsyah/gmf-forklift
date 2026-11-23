"""Pose cadangan PnP 5 titik: konvensi tanda, selisih yang dipelajari, sinyal pose-only, D-05."""
import math

import cv2
import numpy as np

from src.cabin.calibration import Calibration
from src.cabin.face import FaceObs
from src.cabin.face_crop import _FACE5_3D, CropFace, PoseFallback, pose_from_5pts
from src.cabin.indicators import IndicatorInput, build_indicators
from src.cabin.signals import FaceSignals, SignalExtractor
from src.config import Config

WH = (960, 720)


def project(yaw_deg=0.0, pitch_deg=0.0, roll_deg=0.0, dist=600.0):
    """5 titik gambar dari model 3D yang diputar (konvensi: yaw>0 hidung ke kanan gambar, pitch<0 menunduk)."""
    y, p, r = (math.radians(v) for v in (yaw_deg, pitch_deg, roll_deg))
    ry = cv2.Rodrigues(np.array([0.0, -y, 0.0]))[0]      # hidung (-z) ke +x
    rx = cv2.Rodrigues(np.array([-p, 0.0, 0.0]))[0]      # pitch<0 -> hidung ke +y (bawah gambar)
    rz = cv2.Rodrigues(np.array([0.0, 0.0, -r]))[0]
    rot = ry @ rx @ rz
    f = max(WH)
    k = np.array([[f, 0, WH[0] / 2], [0, f, WH[1] / 2], [0, 0, 1]], dtype=float)
    pts, _ = cv2.projectPoints(_FACE5_3D, cv2.Rodrigues(rot)[0], np.array([0, 0, dist]), k, None)
    return pts.reshape(5, 2)


def test_pnp_sign_convention():
    y, p, r = pose_from_5pts(project(yaw_deg=30), WH)
    assert 25 < y < 35 and abs(p) < 5
    y, p, r = pose_from_5pts(project(pitch_deg=-20), WH)
    assert -25 < p < -15 and abs(y) < 5
    assert abs(pose_from_5pts(project(roll_deg=20), WH)[2]) > 15


def test_offset_learned_and_untrusted_axes_nan():
    fb = PoseFallback({"pose_fallback": True, "pose_fallback_min_samples": 3})
    fb.offset, fb.samples = np.array([5.0, 0.0, 0.0]), 3       # mis. MediaPipe +5 deg dibanding PnP
    o = fb.estimate(project(yaw_deg=30), WH, 0.9)
    assert o.pose_only and 33 < o.yaw_raw < 40 and math.isnan(o.pitch_raw) and not math.isnan(o.roll_raw)


class _Det:
    def __init__(self, pts):
        self.pts = pts

    def detect(self, img):
        x1, y1 = self.pts.min(axis=0) - 20
        x2, y2 = self.pts.max(axis=0) + 20
        return [(x1, y1, x2, y2, 0.9, self.pts)]


class _NoLandmarks:
    def process(self, img, ts, off=(0, 0), scale_to=None):
        return FaceObs(present=False)


def test_cropface_returns_pose_only_when_landmarker_fails():
    cf = CropFace(_NoLandmarks(), _Det(project(yaw_deg=40)), {"pose_fallback": True, "pose_fallback_min_samples": 0})
    o = cf.process(np.zeros((720, 960, 3), np.uint8), 0)
    assert o.present and o.pose_only and o.yaw_raw > 30 and cf.last_source == "pose"


def test_not_used_before_enough_samples():
    cf = CropFace(_NoLandmarks(), _Det(project(yaw_deg=40)), {"pose_fallback": True})
    assert cf.process(np.zeros((720, 960, 3), np.uint8), 0).present is False


def test_signals_pose_only():
    se = SignalExtractor({}, Calibration({}))
    s = se.update(FaceObs(True, yaw_raw=35.0, pitch_raw=float("nan"), roll_raw=2.0, pose_only=True), 0)
    assert s.face_present and s.pose_only and not s.eye_valid and math.isnan(s.pitch) and s.yaw > 30


def _sig(ts, yaw=0.0, pitch=0.0, pose_only=False, present=True):
    return FaceSignals(ts_ms=ts, face_present=present, eye_valid=not pose_only, yaw=yaw, pitch=pitch,
                       pose_only=pose_only)


def _run(seq):
    th = Config.load().thresholds
    d05 = build_indicators(th)["D05"]
    act = False
    for s in seq:
        act = d05.update(IndicatorInput(ts_ms=s.ts_ms, sig=s, phone_conf=0.0, no_helmet=False, no_vest=False,
                                        yolo_valid=False, blocked_frame=False, moving=True)) or act
    return act


def test_d05_turn_with_pose_only_frames():
    seq = [_sig(i * 33, yaw=0) for i in range(30)]
    seq += [_sig(1000 + i * 33, yaw=45, pitch=float("nan"), pose_only=True) for i in range(70)]   # 2.3 s menoleh
    assert _run(seq) is True


def test_d05_lap_look_hold_through_pose_only_frames():
    seq = [_sig(i * 33, yaw=0) for i in range(30)]
    seq += [_sig(1000 + i * 33, pitch=-35) for i in range(10)]                     # menunduk (landmark ada)
    seq += [_sig(1330 + i * 33, yaw=2, pitch=float("nan"), pose_only=True) for i in range(60)]   # landmark hilang
    assert _run(seq) is True


def test_no_d05_when_pose_only_looking_ahead():
    seq = [_sig(i * 33, yaw=0) for i in range(30)]
    seq += [_sig(1000 + i * 33, yaw=3, pitch=float("nan"), pose_only=True) for i in range(90)]
    assert _run(seq) is False
