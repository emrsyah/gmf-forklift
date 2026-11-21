"""tools/calibrate_camera.py: papan cetak, deteksi diperkecil, kalibrasi + buang outlier pada gambar sintetis."""
from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools"))
import calibrate_camera as cc  # noqa: E402

COLS, ROWS, SQ = 9, 6, 0.025
W, H = 1280, 720
K_TRUE = np.array([[900.0, 0, 640.0], [0, 900.0, 360.0], [0, 0, 1]])


def _board_texture(px: int = 40) -> np.ndarray:
    img = np.full(((ROWS + 3) * px, (COLS + 3) * px), 255, np.uint8)   # 1 kotak margin putih tiap sisi
    for r in range(ROWS + 1):
        for c in range(COLS + 1):
            if (r + c) % 2 == 0:
                img[(r + 1) * px:(r + 2) * px, (c + 1) * px:(c + 2) * px] = 0
    return img


def _render(rvec, tvec, px: int = 40) -> np.ndarray:
    """Gambar papan dilihat kamera K_TRUE tanpa distorsi (homografi bidang papan -> gambar)."""
    tex = _board_texture(px)
    # titik tekstur (piksel) -> koordinat papan (m); sudut dalam pertama di tekstur = (2px, 2px)
    src = np.array([[0, 0], [tex.shape[1], 0], [tex.shape[1], tex.shape[0]], [0, tex.shape[0]]], np.float64)
    obj = np.c_[(src - 2 * px) / px * SQ, np.zeros(4)]
    dst, _ = cv2.projectPoints(obj, rvec, tvec, K_TRUE, None)
    Hm = cv2.getPerspectiveTransform(src.astype(np.float32), dst.reshape(-1, 2).astype(np.float32))
    out = cv2.warpPerspective(tex, Hm, (W, H), flags=cv2.INTER_LINEAR, borderValue=180)
    return cv2.GaussianBlur(out, (3, 3), 0)


def _views():
    rng = np.random.default_rng(0)
    views = []
    for i in range(16):
        rvec = np.array([rng.uniform(-0.5, 0.5), rng.uniform(-0.5, 0.5), rng.uniform(-0.2, 0.2)])
        tvec = np.array([rng.uniform(-0.15, 0.05), rng.uniform(-0.1, 0.0), rng.uniform(0.45, 0.8)])
        views.append(_render(rvec, tvec))
    return views


@pytest.fixture(scope="module")
def corners():
    pts = [cc.find_corners(g, COLS, ROWS, detect_max_side=640) for g in _views()]
    return [p for p in pts if p is not None]


def test_detection_on_downscaled_image_is_subpixel(corners):
    assert len(corners) >= 12
    full = [cc.find_corners(g, COLS, ROWS) for g in _views()[:4]]
    for a, b in zip(corners[:4], full):
        if b is not None:
            assert np.abs(a - b).max() < 0.3          # deteksi diperkecil + subpix penuh ~ sama dengan penuh


def test_calibration_recovers_k(corners):
    objp = cc.board_points(COLS, ROWS, SQ)
    rms, K, dist, used, bad, _ = cc.calibrate_robust(objp, corners, (W, H), False, 2.5, 0.5, 10)
    assert rms < 0.5
    assert abs(K[0, 0] - 900) / 900 < 0.02 and abs(K[1, 1] - 900) / 900 < 0.02
    assert abs(K[0, 2] - 640) < 15 and abs(K[1, 2] - 360) < 15


def test_bad_view_is_rejected(corners):
    objp = cc.board_points(COLS, ROWS, SQ)
    broken = corners[0].copy()
    broken[::2] += 6.0                                  # sudut salah deteksi
    pts = corners + [broken]
    _, _, _, used, bad, _ = cc.calibrate_robust(objp, pts, (W, H), False, 2.5, 0.5, 10)
    assert bad == [len(pts) - 1] and len(used) == len(corners)


def test_coverage_cells():
    cov = cc.Coverage((W, H))
    assert cov.percent == 0 and not any(cov.cells_3x3())
    cov.add(np.array([[0, 0], [W - 1, 0], [W - 1, H - 1], [0, H - 1]], float))
    assert cov.percent > 95 and all(cov.cells_3x3())


def test_make_board_detectable_and_scaled(tmp_path):
    p = cc.make_board(tmp_path / "b.png", COLS, ROWS, 25.0, dpi=150)
    img = cv2.imread(str(p), cv2.IMREAD_GRAYSCALE)
    assert img.shape == (round(210 * 150 / 25.4), round(297 * 150 / 25.4))
    c = cc.find_corners(img, COLS, ROWS)
    assert c is not None
    spacing_px = np.linalg.norm(np.diff(c.reshape(ROWS, COLS, 2), axis=1), axis=2).mean()
    assert abs(spacing_px - 25.0 * 150 / 25.4) < 1.0
    with pytest.raises(ValueError):
        cc.make_board(tmp_path / "x.png", COLS, ROWS, 40.0)


def test_source_config_uses_runtime_front_settings():
    rt = {"front": {"source": {"type": "file", "index": 1, "width": 1280, "height": 720, "fourcc": "MJPG",
                               "autofocus": False, "focus": 0, "path": "x.mp4", "realtime": True}}}
    cfg = cc.source_config("webcam:2", rt)
    assert cfg["type"] == "webcam" and cfg["index"] == 2 and cfg["fourcc"] == "MJPG" and cfg["autofocus"] is False
    assert "path" not in cfg
    assert cc.source_config("vid.mp4", rt) == {"type": "file", "path": "vid.mp4", "realtime": False}
