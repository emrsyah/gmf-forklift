"""Live kamera kabin ke dashboard: JPEG kecil (pesan `frame`, base64) dengan fps terbatas.

Bukan video asli: cukup untuk melihat kondisi kabin dari command center. Frame selalu boleh dibuang
(hanya yang terbaru dikirim), jadi tidak pernah menunda event alarm.
Privasi: ini wajah pengemudi secara live (CLAUDE.md pertanyaan terbuka #5) -> bisa dimatikan lewat config.
"""
from __future__ import annotations

import base64

import cv2
import numpy as np

from src.preprocess import resize_max_side


class LiveFrames:
    def __init__(self, cfg: dict | None, camera: str = "cabin"):
        cfg = cfg or {}
        self.enabled = bool(cfg.get("enabled", False))
        self.period_ms = 1000.0 / max(float(cfg.get("fps", 4)), 0.1)
        self.max_side = int(cfg.get("max_side", 480))
        self.quality = int(cfg.get("jpeg_quality", 60))
        self.overlay = bool(cfg.get("overlay", False))      # True = gambar dengan overlay debug
        self.camera = camera
        self._last: float | None = None

    def due(self, ts_ms: float) -> bool:
        return self.enabled and (self._last is None or ts_ms - self._last >= self.period_ms * 0.95)

    def encode(self, img: np.ndarray, ts_ms: float) -> dict:
        self._last = ts_ms
        small = resize_max_side(img, self.max_side)
        ok, buf = cv2.imencode(".jpg", small, [cv2.IMWRITE_JPEG_QUALITY, self.quality])
        b64 = base64.b64encode(buf.tobytes()).decode("ascii") if ok else ""
        return {"type": "frame", "camera": self.camera, "jpegB64": b64, "ts_ms": int(ts_ms)}
