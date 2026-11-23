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


class PartialBlock:
    """Lensa tertutup SEBAGIAN: frame dibagi grid; sel 'tertutup' jika teksturnya (variansi Laplacian) turun jauh
    dibanding baseline sel itu sendiri, atau sel sangat gelap/terang. Baseline hanya diperbarui saat wajah
    terlihat (kamera jelas tidak terhalang), jadi dinding polos yang memang rata tidak dihitung (sel tidak
    informatif). Keluaran: fraksi sel informatif yang tertutup. Semua angka dari D09_camera_block.partial.
    D-09 tetap mensyaratkan wajah tidak terdeteksi (indikator), jadi tangan yang menutup sebagian lensa tapi
    wajah masih terbaca tidak memicu alert."""

    def __init__(self, cfg: dict | None, mean_min: float = 15, mean_max: float = 240):
        c = cfg or {}
        self.enabled = bool(c.get("enabled", False))
        self.rows, self.cols = (int(v) for v in c.get("grid", [4, 4]))
        self.downscale_px = int(c.get("downscale_px", 160))
        self.drop_ratio = float(c.get("drop_ratio", 0.25))
        self.min_base_lap = float(c.get("min_base_lap", 40.0))
        self.min_cells = int(c.get("min_cells", 4))
        self.frac_min = float(c.get("frac_min", 0.6))
        # hysteresis: setelah tertutup, lepas hanya jika fraksi < frac_exit (null = tanpa hysteresis)
        fe = c.get("frac_exit")
        self.frac_exit = self.frac_min if fe is None else float(fe)
        self._on = False
        self.alpha = float(c.get("baseline_alpha", 0.05))
        self.warmup = int(c.get("warmup_frames", 15))
        self.mean_min, self.mean_max = float(mean_min), float(mean_max)
        self._base: np.ndarray | None = None
        self._n = 0
        self.frac = 0.0

    def cell_stats(self, img_bgr: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        small = resize_max_side(img_bgr, self.downscale_px)
        g = small if small.ndim == 2 else cv2.cvtColor(small, cv2.COLOR_BGR2GRAY)
        lap = cv2.Laplacian(g, cv2.CV_64F)
        h, w = g.shape
        ys = np.linspace(0, h, self.rows + 1).astype(int)
        xs = np.linspace(0, w, self.cols + 1).astype(int)
        lv = np.zeros((self.rows, self.cols))
        mn = np.zeros((self.rows, self.cols))
        for i in range(self.rows):
            for j in range(self.cols):
                lv[i, j] = lap[ys[i]:ys[i + 1], xs[j]:xs[j + 1]].var()
                mn[i, j] = g[ys[i]:ys[i + 1], xs[j]:xs[j + 1]].mean()
        return lv, mn

    def update(self, img_bgr: np.ndarray, face_present: bool) -> bool:
        """True jika fraksi sel tertutup >= frac_min."""
        self.frac = 0.0
        if not self.enabled:
            return False
        lv, mn = self.cell_stats(img_bgr)
        if face_present:
            self._base = lv.copy() if self._base is None else (1 - self.alpha) * self._base + self.alpha * lv
            self._n += 1
        if self._base is None or self._n < self.warmup:
            self._on = False
            return False
        informative = self._base >= self.min_base_lap
        if informative.sum() < self.min_cells:
            self._on = False
            return False
        covered = informative & ((lv <= self.drop_ratio * self._base) | (mn < self.mean_min) | (mn > self.mean_max))
        self.frac = float(covered.sum() / informative.sum())
        self._on = self.frac >= (self.frac_exit if self._on else self.frac_min)
        return self._on
