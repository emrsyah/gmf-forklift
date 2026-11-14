"""Preprocessing gambar yang sama untuk runtime dan training (simulasi NIR, ROI)."""
from __future__ import annotations

import cv2
import numpy as np

_clahe_cache: dict[tuple, cv2.CLAHE] = {}


def _clahe(clip: float, tile: int) -> cv2.CLAHE:
    key = (float(clip), int(tile))
    if key not in _clahe_cache:
        _clahe_cache[key] = cv2.createCLAHE(clipLimit=clip, tileGridSize=(tile, tile))
    return _clahe_cache[key]


def simulate_nir(img_bgr: np.ndarray, clip: float = 2.0, tile: int = 8) -> np.ndarray:
    """Grayscale 3 channel + CLAHE (Tech Spec bag. 3 dan 5). Juga dipakai untuk frame NIR mono asli."""
    gray = img_bgr if img_bgr.ndim == 2 else cv2.cvtColor(img_bgr, cv2.COLOR_BGR2GRAY)
    eq = _clahe(clip, tile).apply(gray)
    return cv2.cvtColor(eq, cv2.COLOR_GRAY2BGR)


def crop_roi(img: np.ndarray, roi: list[float] | None) -> tuple[np.ndarray, tuple[int, int]]:
    """Crop ROI relatif [x, y, w, h] (0..1). Mengembalikan crop dan offset (x0, y0) piksel."""
    if not roi:
        return img, (0, 0)
    h, w = img.shape[:2]
    x, y, rw, rh = roi
    x0, y0 = int(x * w), int(y * h)
    x1, y1 = min(w, int((x + rw) * w)), min(h, int((y + rh) * h))
    return img[y0:y1, x0:x1], (x0, y0)


def resize_max_side(img: np.ndarray, max_side: int) -> np.ndarray:
    h, w = img.shape[:2]
    s = max(h, w)
    if s <= max_side:
        return img
    k = max_side / s
    return cv2.resize(img, (int(w * k), int(h * k)), interpolation=cv2.INTER_AREA)
