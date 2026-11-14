"""Uji geometri jarak titik kaki dengan proyeksi OpenCV (bukan rumus kita sendiri yang dibalik)."""
import math

import cv2
import numpy as np
import pytest

from src.config import CONFIG_DIR, load_yaml
from src.front.distance import (METHOD_CLIPPED, METHOD_FOOT, CameraModel,
                                DistanceEstimator)

W, H = 1280, 720
K = np.array([[914.0, 0.0, 640.0], [0.0, 914.0, 360.0], [0.0, 0.0, 1.0]])
CAM_H = 1.5
THETA_DEG = 12.0


def _rt(theta_deg):
    """Kamera di (0,0,0) menunduk theta. Titik dunia (X samping, Y=+h ke bawah, Z maju)."""
    t = math.radians(theta_deg)
    R = np.array([[1, 0, 0], [0, math.cos(t), -math.sin(t)], [0, math.sin(t), math.cos(t)]])
    rvec, _ = cv2.Rodrigues(R)
    return rvec, np.zeros(3)


def _project(pts_xz, Kp, dist, fisheye=False, h=CAM_H, theta=THETA_DEG):
    obj = np.array([[x, h, z] for x, z in pts_xz], dtype=np.float64).reshape(-1, 1, 3)
    rvec, tvec = _rt(theta)
    if fisheye:
        img, _ = cv2.fisheye.projectPoints(obj, rvec, tvec, Kp, dist)
    else:
        img, _ = cv2.projectPoints(obj, rvec, tvec, Kp, dist)
    return img.reshape(-1, 2)


def _est(dist, fisheye=False, image_size=(W, H)):
    return DistanceEstimator(CameraModel(K, dist, fisheye, image_size), CAM_H, THETA_DEG,
                             class_height_m={"person": 1.65}, bottom_margin_px=2,
                             min_ray_angle_deg=0.5, max_distance_m=40.0)


GROUND = [(x, z) for z in (3.0, 4.0, 6.0, 8.0, 10.0) for x in (-1.5, 0.0, 1.5)]


@pytest.mark.parametrize("dist", [np.zeros(5), np.array([-0.25, 0.08, 0.001, -0.002, -0.01])])
def test_pinhole_known_geometry(dist):
    est = _est(dist)
    uv = _project(GROUND, K, dist)
    for (x, z), (u, v) in zip(GROUND, uv):
        assert 0 <= u < W and 0 <= v < H
        fwd, lat = est.ground_point(u, v, W, H)
        assert fwd == pytest.approx(z, rel=0.01)
        assert lat == pytest.approx(x, abs=0.02)


def test_fisheye_known_geometry():
    D = np.array([0.08, -0.02, 0.005, -0.001])
    est = _est(D, fisheye=True)
    uv = _project(GROUND, K, D, fisheye=True)
    for (x, z), (u, v) in zip(GROUND, uv):
        fwd, lat = est.ground_point(u, v, W, H)
        assert fwd == pytest.approx(z, rel=0.01)
        assert lat == pytest.approx(x, abs=0.02)


def test_k_scaled_to_frame_size():
    est = _est(np.zeros(5), image_size=(W, H))   # dikalibrasi 1280x720
    w2, h2 = 640, 360
    K2 = K.copy()
    K2[:2] *= 0.5
    uv = _project([(0.0, 5.0)], K2, np.zeros(5))
    fwd, _ = est.ground_point(*uv[0], w2, h2)
    assert fwd == pytest.approx(5.0, rel=0.01)


def test_estimate_uses_foot_point_for_full_box():
    est = _est(np.zeros(5))
    (u, v), = _project([(0.0, 6.0)], K, np.zeros(5))
    box = (u - 40, v - 250, u + 40, v)
    e = est.estimate(box, "person", W, H)
    assert e.method == METHOD_FOOT
    assert e.distance_m == pytest.approx(6.0, rel=0.01)


def test_point_above_horizon_is_none():
    est = _est(np.zeros(5))
    v_horizon = K[1, 2] - K[1, 1] * math.tan(math.radians(THETA_DEG))
    assert est.ground_point(640, v_horizon - 20, W, H) is None
    assert est.estimate((600, 0, 680, v_horizon - 20), "chair", W, H) is None   # tanpa tinggi kelas


def test_height_fallback():
    est = _est(np.zeros(5))
    box_h = K[1, 1] * 1.65 / 5.0
    assert est.height_distance(box_h, "person", W, H) == pytest.approx(5.0, rel=1e-6)
    assert est.height_distance(box_h, "pallet_jack", W, H) is None


def test_close_person_with_feet_cut_off_is_in_danger_zone():
    """Orang ~1 m dari kamera, kaki & kepala di luar frame: harus tetap <= danger_zone_m."""
    danger = load_yaml(CONFIG_DIR / "thresholds.yaml")["front"]["danger_zone_m"]
    est = _est(np.zeros(5))
    e = est.estimate((500, 0, 780, H - 1), "person", W, H)
    assert e.method == METHOD_CLIPPED
    assert e.distance_m <= danger


def test_from_config_file_loads():
    cfg = load_yaml(CONFIG_DIR / "front_camera.yaml")
    est = DistanceEstimator.from_config(cfg)
    assert est.h_m > 0
    e = est.estimate((600, 300, 680, 600), "person", 1280, 720)
    assert e is not None and e.distance_m > 0
