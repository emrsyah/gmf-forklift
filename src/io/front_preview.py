"""Pratinjau + rekaman kamera depan di thread sendiri (tanpa deteksi).

Dipakai di Jetson selama model depan (TensorRT YOLO + ByteTrack) belum dipasang: kamera depan tetap
terlihat di tampilan web (:8090, panel kedua) dan direkam untuk pengumpulan data. Thread terpisah supaya
deteksi kabin tidak ikut melambat. Kalau kamera dicabut, dicoba buka lagi tiap `retry_s`.
"""
from __future__ import annotations

import threading
import time

import cv2
import numpy as np

_ROT = {0: None, 90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}


def rotate(img: np.ndarray, deg: int) -> np.ndarray:
    code = _ROT.get(int(deg) % 360)
    if int(deg) % 360 not in _ROT:
        raise ValueError(f"rotate harus 0/90/180/270, bukan {deg}")
    return img if code is None else cv2.rotate(img, code)


class FrontPreview:
    def __init__(self, source_cfg: dict, cfg: dict, web=None, recorder=None):
        self.source_cfg = source_cfg
        self.rotate = int(cfg.get("rotate", 0))
        self.retry_s = float(cfg.get("retry_s", 5))
        self.web = web
        self.recorder = recorder
        self.frames = 0
        self.connected = False
        self._stop = threading.Event()
        if web is not None:
            web.add_channel("front", cfg.get("label", "Kamera depan"))
        self._thread = threading.Thread(target=self._run, name="front-preview", daemon=True)
        self._thread.start()

    def _run(self) -> None:
        from src.capture import make_source
        while not self._stop.is_set():
            try:
                src = make_source(self.source_cfg)
            except Exception as e:  # kamera tidak ada / sedang dipakai
                if self.connected or self.frames == 0:
                    print(f"[depan] kamera tidak bisa dibuka: {e}; coba lagi tiap {self.retry_s:.0f} s", flush=True)
                self.connected = False
                self._stop.wait(self.retry_s)
                continue
            self.connected = True
            print("[depan] kamera depan tersambung", flush=True)
            with src:
                for fr in src:
                    if self._stop.is_set():
                        break
                    img = rotate(fr.image, self.rotate)
                    if self.recorder is not None:
                        self.recorder.push(img, fr.ts_ms)
                    if self.web is not None:
                        self.web.publish(img, "front")
                    self.frames += 1
            if not self._stop.is_set():
                print("[depan] kamera depan terputus", flush=True)
                self.connected = False
                self._stop.wait(self.retry_s)

    def close(self, timeout_s: float = 3.0) -> None:
        self._stop.set()
        self._thread.join(timeout_s)
        if self.recorder is not None:
            self.recorder.close()
