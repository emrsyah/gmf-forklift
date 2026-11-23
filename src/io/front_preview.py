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
    def __init__(self, source_cfg: dict, cfg: dict, web=None, recorder=None, h264_recorder=None):
        self.source_cfg = source_cfg
        self.rotate = int(cfg.get("rotate", 0))
        self.retry_s = float(cfg.get("retry_s", 5))
        self.web = web
        self.recorder = recorder            # MJPG (cuplik ~fps); cadangan kalau kamera tidak mengirim H.264
        self.h264_recorder = h264_recorder  # H.264 dari kamera, tanpa encode ulang (src.capture.uvch264)
        self._src = None
        self._mode: str | None = None
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
            self._src = src
            with src:
                for fr in src:
                    if self._stop.is_set():
                        break
                    img = rotate(fr.image, self.rotate)
                    au = fr.meta.get("h264")
                    if self.h264_recorder is not None and au is not None:
                        self._log_mode("H.264 dari kamera")
                        self.h264_recorder.push(au, fr.ts_ms, fr.meta.get("h264_idr"))
                    elif self.recorder is not None:
                        self._log_mode("MJPG")
                        self.recorder.push(img, fr.ts_ms)
                    if self.web is not None:
                        self.web.publish(img, "front")
                    self.frames += 1
            self._src = None
            if not self._stop.is_set():
                print("[depan] kamera depan terputus", flush=True)
                self.connected = False
                self._stop.wait(self.retry_s)

    def _log_mode(self, mode: str) -> None:
        if mode != self._mode:
            self._mode = mode
            print(f"[depan] rekaman: {mode}", flush=True)

    def request_idr(self) -> None:
        """Dipanggil perekam H.264 saat segmen jatuh tempo: minta keyframe agar segmen cepat berganti."""
        src = self._src
        if src is not None and hasattr(src, "request_idr"):
            src.request_idr()

    def close(self, timeout_s: float = 3.0) -> None:
        self._stop.set()
        self._thread.join(timeout_s)
        for rec in (self.recorder, self.h264_recorder):
            if rec is not None:
                rec.close()
