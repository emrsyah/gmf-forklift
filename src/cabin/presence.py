"""Gerbang "pengemudi ada di kursi" (thresholds `driver_presence`).

Uji lapangan 2026-11-25 (sampel 54 snapshot): banyak alert keluar saat TIDAK ada pengemudi: kursi kosong dengan
orang di samping forklift, kamera dipegang / tergeletak di meja kantor, orang berdiri jauh. Indikator wajah dan
YOLO tetap dihitung, tetapi kode di `gated_codes` tidak dikirim sampai ada wajah pengemudi:
  - tinggi kotak wajah >= min_face_h_frac x tinggi frame (pengemudi dekat kamera; orang jauh wajahnya kecil),
  - pusat wajah di dalam x_range / y_range (posisi kursi pengemudi di gambar),
  - terlihat >= enter_s (orang lewat sesaat tidak dihitung).
Setelah itu pengemudi dianggap ada sampai hold_s tanpa wajah yang memenuhi syarat (menunduk ke pangkuan,
menoleh jauh, mata tertutup tetap terhitung). D-09 tidak digerbang: lensa ditutup = tidak ada wajah.
"""
from __future__ import annotations

import numpy as np


class DriverPresence:
    def __init__(self, cfg: dict | None):
        c = cfg or {}
        self.enabled = bool(c.get("enabled", False))
        self.min_face_h_frac = float(c.get("min_face_h_frac", 0.0))
        self.x_range = c.get("x_range")
        self.y_range = c.get("y_range")
        self.enter_ms = float(c.get("enter_s", 0.0)) * 1000
        self.hold_ms = float(c.get("hold_s", 10.0)) * 1000
        self.gated = set(c.get("gated_codes", ["D01", "D02", "D04", "D05", "D06", "D10"]))
        # pose cadangan (tanpa landmark): tinggi wajah ~ eye_to_face_h x jarak antar mata
        self.eye_to_face_h = float(c.get("eye_to_face_h", 2.5))
        self.present = not self.enabled
        self.last_face: tuple[float, float, float] | None = None   # (h_frac, cx, cy) untuk overlay/log
        self._seen_since: int | None = None
        self._last_ok_ms: int | None = None

    def face_geom(self, obs, frame_wh) -> tuple[float, float, float] | None:
        """(tinggi wajah / tinggi frame, pusat x 0..1, pusat y 0..1) atau None."""
        if not obs.present:
            return None
        w, h = frame_wh
        if obs.landmarks_px is not None and len(obs.landmarks_px):
            xy = np.asarray(obs.landmarks_px, dtype=float)
            x0, y0 = xy.min(axis=0)
            x1, y1 = xy.max(axis=0)
            return (y1 - y0) / h, (x0 + x1) / 2 / w, (y0 + y1) / 2 / h
        if obs.eye_pts is not None:
            e = np.asarray(obs.eye_pts, dtype=float)
            d = float(np.hypot(*(e[0] - e[1])))
            cx, cy = e.mean(axis=0)
            return self.eye_to_face_h * d / h, cx / w, cy / h
        return None

    def qualifies(self, g) -> bool:
        if g is None:
            return False
        hf, cx, cy = g
        if hf < self.min_face_h_frac:
            return False
        if self.x_range is not None and not (self.x_range[0] <= cx <= self.x_range[1]):
            return False
        if self.y_range is not None and not (self.y_range[0] <= cy <= self.y_range[1]):
            return False
        return True

    def update(self, obs, frame_wh, ts_ms: int) -> bool:
        if not self.enabled:
            return True
        g = self.face_geom(obs, frame_wh)
        self.last_face = g
        if self.qualifies(g):
            if self._seen_since is None:
                self._seen_since = ts_ms
            if ts_ms - self._seen_since >= self.enter_ms:
                self._last_ok_ms = ts_ms
        else:
            self._seen_since = None
        self.present = self._last_ok_ms is not None and ts_ms - self._last_ok_ms <= self.hold_ms
        return self.present

    def filter(self, active: dict) -> dict:
        if self.present:
            return active
        return {k: v for k, v in active.items() if k not in self.gated}
