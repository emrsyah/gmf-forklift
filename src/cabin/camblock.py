"""D-09: deteksi kamera terhalang dari frame mentah (tanpa model).

Dengan lampu IR, telapak tangan yang menutup lensa bisa membuat gambar TERANG dan rata,
jadi kita menilai keseragaman (std + variansi Laplacian rendah), bukan hanya kegelapan.
"""
from __future__ import annotations

import cv2
import numpy as np

from src.preprocess import resize_max_side


def frame_stats(img_bgr: np.ndarray, downscale_px: int = 160) -> dict:
    small = resize_max_side(img_bgr, downscale_px)
    g = small if small.ndim == 2 else cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
    return {
        "mean": float(g.mean()),
        "std": float(g.std()),
        "lap_var": float(cv2.Laplacian(g, cv2.CV_64F).var()),
    }


def is_blocked(stats: dict, cfg: dict) -> bool:
    uniform = stats["std"] < cfg.get("std_max", 10) and stats["lap_var"] < cfg.get("lap_var_max", 20)
    too_dark = stats["mean"] < cfg.get("mean_min", 15)
    too_bright = stats["mean"] > cfg.get("mean_max", 240)
    return bool(uniform or too_dark or too_bright)
