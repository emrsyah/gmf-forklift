"""Gerbang pengemudi: kode wajah/YOLO hanya dikirim saat wajah pengemudi (cukup besar, di area kursi) terlihat."""
from types import SimpleNamespace

import numpy as np

from src.cabin.presence import DriverPresence

WH = (960, 540)
CFG = {"enabled": True, "min_face_h_frac": 0.12, "x_range": [0.2, 0.8], "enter_s": 2.0, "hold_s": 10.0}


def face(h_frac, cx, cy=0.4):
    h = h_frac * WH[1]
    x, y = cx * WH[0], cy * WH[1]
    pts = np.array([[x - h / 2, y - h / 2], [x + h / 2, y + h / 2]])
    return SimpleNamespace(present=True, landmarks_px=pts, eye_pts=None)


NONE = SimpleNamespace(present=False, landmarks_px=None, eye_pts=None)
ACTIVE = {"D05": {}, "D10": {}, "D09": {}}


def run(p, obs, t0, t1, dt=100):
    for t in range(t0, t1, dt):
        p.update(obs, WH, t)
    return p.present


def test_disabled_never_gates():
    p = DriverPresence(None)
    assert p.update(NONE, WH, 0) and p.filter(ACTIVE) == ACTIVE


def test_driver_needs_enter_time_then_holds():
    p = DriverPresence(CFG)
    assert not run(p, face(0.25, 0.5), 0, 1900)          # < 2 s: belum
    assert run(p, face(0.25, 0.5), 1900, 2200)           # >= 2 s: ada
    assert run(p, NONE, 2200, 11_000)                    # menoleh / menunduk: masih ada (hold 10 s)
    assert not run(p, NONE, 11_000, 13_000)              # turun dari forklift


def test_far_and_side_faces_are_not_the_driver():
    for obs in (face(0.06, 0.45), face(0.14, 0.87)):     # orang jauh di kantor / orang di samping forklift
        p = DriverPresence(CFG)
        assert not run(p, obs, 0, 5000)
        out = p.filter(ACTIVE)
        assert set(out) == {"D09"}                         # D-09 tidak digerbang


def test_pose_only_face_uses_eye_distance():
    p = DriverPresence(CFG)
    eyes = SimpleNamespace(present=True, landmarks_px=None, eye_pts=np.array([[450.0, 200.0], [510.0, 200.0]]))
    g = p.face_geom(eyes, WH)
    assert abs(g[0] - 2.5 * 60 / 540) < 1e-6 and p.qualifies(g)
