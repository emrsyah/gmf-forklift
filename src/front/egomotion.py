"""Arah gerak forklift (maju / diam / mundur) dari kamera depan, tanpa kabel ke forklift.

Saat forklift MAJU, pemandangan di kamera depan MEMBESAR (titik menjauh dari pusat ekspansi); saat MUNDUR,
MENGECIL. Optical flow Lucas-Kanade pada titik sudut (di luar kotak deteksi: orang/kendaraan yang bergerak
sendiri diabaikan) -> transformasi similarity (geser + putar + skala, RANSAC) antar dua frame -> laju skala
v = (skala - 1) / dt per detik. Belok dan guncangan menjadi geser/putar, bukan skala, dan ukuran ini tidak
bergantung pada orientasi pemasangan kamera (uji 2026-11-25: kamera depan sempat terpasang miring 90 derajat;
ukuran "gerak vertikal lantai" salah membaca maju-sambil-belok sebagai mundur).

Status `reversing` dipakai kabin untuk tidak menghitung D-05 (menoleh ke belakang/spion saat mundur = benar).
Semua ambang di thresholds.yaml `front_motion`.
"""
from __future__ import annotations

import threading
import time
from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class EgoSample:
    ts_ms: int
    v: float | None        # laju skala pemandangan per detik (+ = maju/membesar, - = mundur); None = tak terukur
    n_points: int


class EgoMotion:
    def __init__(self, cfg: dict):
        self.width = int(cfg.get("width", 320))
        self.ransac_px = float(cfg.get("ransac_px", 1.5))
        self.max_corners = int(cfg.get("max_corners", 150))
        self.min_points = int(cfg.get("min_points", 15))
        self.box_margin = float(cfg.get("box_margin_frac", 0.1))
        self.max_dt_s = float(cfg.get("max_dt_s", 0.5))
        self._prev: tuple[np.ndarray, int] | None = None

    def update(self, img: np.ndarray, ts_ms: int, boxes=()) -> EgoSample:
        h0, w0 = img.shape[:2]
        s = self.width / float(w0)
        g = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        g = cv2.resize(g, (self.width, max(1, int(round(h0 * s)))), interpolation=cv2.INTER_AREA)
        prev, self._prev = self._prev, (g, ts_ms)
        if prev is None or prev[0].shape != g.shape:
            return EgoSample(ts_ms, None, 0)
        pg, pts_ms = prev
        dt = (ts_ms - pts_ms) / 1000.0
        if dt <= 0 or dt > self.max_dt_s:
            return EgoSample(ts_ms, None, 0)
        mask = np.full_like(pg, 255)
        for x1, y1, x2, y2 in boxes:                      # objek yang bisa bergerak sendiri
            mx, my = (x2 - x1) * self.box_margin, (y2 - y1) * self.box_margin
            mask[max(0, int((y1 - my) * s)):int((y2 + my) * s) + 1, max(0, int((x1 - mx) * s)):int((x2 + mx) * s) + 1] = 0
        p0 = cv2.goodFeaturesToTrack(pg, self.max_corners, 0.01, 7, mask=mask)
        if p0 is None or len(p0) < self.min_points:
            return EgoSample(ts_ms, None, 0 if p0 is None else len(p0))
        p1, st, _ = cv2.calcOpticalFlowPyrLK(pg, g, p0, None, winSize=(21, 21), maxLevel=3)
        pb, stb, _ = cv2.calcOpticalFlowPyrLK(g, pg, p1, None, winSize=(21, 21), maxLevel=3)
        ok = (st.ravel() == 1) & (stb.ravel() == 1) & (np.linalg.norm((pb - p0).reshape(-1, 2), axis=1) < 1.0)
        if ok.sum() < self.min_points:
            return EgoSample(ts_ms, None, int(ok.sum()))
        a, b = p0.reshape(-1, 2)[ok], p1.reshape(-1, 2)[ok]
        M, inl = cv2.estimateAffinePartial2D(a, b, method=cv2.RANSAC, ransacReprojThreshold=self.ransac_px)
        n_in = int(inl.sum()) if inl is not None else 0
        if M is None or n_in < self.min_points:
            return EgoSample(ts_ms, None, n_in)
        scale = float(np.hypot(M[0, 0], M[1, 0]))
        return EgoSample(ts_ms, (scale - 1.0) / dt, n_in)


class ReverseState:
    """Histeresis waktu di atas EgoSample: mundur jika v <= -enter_v selama enter_s; selesai jika
    v > -exit_v selama exit_s. Dibaca thread kabin lewat `reversing_at(ts)` (data basi = bukan mundur)."""

    def __init__(self, cfg: dict):
        self.enter_v = float(cfg.get("reverse_enter_v", 0.05))
        self.exit_v = float(cfg.get("reverse_exit_v", 0.02))
        self.enter_ms = float(cfg.get("reverse_enter_s", 0.7)) * 1000
        self.exit_ms = float(cfg.get("reverse_exit_s", 1.5)) * 1000
        self.window_ms = float(cfg.get("median_window_s", 1.0)) * 1000
        self.stale_ms = float(cfg.get("stale_s", 1.5)) * 1000
        self.reversing = False
        self.v = None                      # median v di jendela (None = belum ada sampel terukur)
        self._win: list[tuple[int, float]] = []
        self._cand_since: int | None = None
        self._last_ts: int | None = None
        self._last_wall_ms: int | None = None
        self._lock = threading.Lock()

    def update(self, smp: EgoSample) -> bool:
        with self._lock:
            self._last_ts = smp.ts_ms
            self._last_wall_ms = int(time.time() * 1000)
            if smp.v is not None:
                self._win.append((smp.ts_ms, smp.v))
            self._win = [(t, v) for t, v in self._win if smp.ts_ms - t <= self.window_ms]
            if not self._win:
                self.v = None
                return self.reversing
            self.v = float(np.median([v for _, v in self._win]))
            want = self.v <= -self.enter_v if not self.reversing else self.v <= -self.exit_v
            if want != self.reversing:
                if self._cand_since is None:
                    self._cand_since = smp.ts_ms
                need = self.enter_ms if not self.reversing else self.exit_ms
                if smp.ts_ms - self._cand_since >= need:
                    self.reversing, self._cand_since = want, None
            else:
                self._cand_since = None
            return self.reversing

    def reversing_now(self) -> bool:
        """Untuk thread lain (jam dinding): status mundur, False jika kamera depan berhenti memperbarui."""
        with self._lock:
            if self._last_wall_ms is None or int(time.time() * 1000) - self._last_wall_ms > self.stale_ms:
                return False
            return self.reversing
