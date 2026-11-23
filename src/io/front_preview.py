"""Pratinjau + rekaman kamera depan di thread sendiri (tanpa deteksi).

Dipakai di Jetson selama model depan (TensorRT YOLO + ByteTrack) belum dipasang: kamera depan tetap
terlihat di tampilan web (:8090, panel kedua) dan direkam untuk pengumpulan data. Thread terpisah supaya
deteksi kabin tidak ikut melambat. Kalau kamera dicabut, dicoba buka lagi tiap `retry_s`.

Hemat CPU: kalau kamera mengirim H.264 (src.capture.uvch264), aliran itu langsung diteruskan ke sink H.264
(rekaman, streaming) tanpa dekode. JPEG hanya didekode + diputar saat benar-benar dipakai: ada penonton web
(dibatasi web.fps) atau perekam MJPG cadangan mengambil frame itu.
"""
from __future__ import annotations

import threading

import cv2
import numpy as np

_ROT = {0: None, 90: cv2.ROTATE_90_CLOCKWISE, 180: cv2.ROTATE_180, 270: cv2.ROTATE_90_COUNTERCLOCKWISE}


def rotate(img: np.ndarray, deg: int) -> np.ndarray:
    code = _ROT.get(int(deg) % 360)
    if int(deg) % 360 not in _ROT:
        raise ValueError(f"rotate harus 0/90/180/270, bukan {deg}")
    return img if code is None else cv2.rotate(img, code)


def preview_source_cfg(source_cfg: dict) -> dict:
    """Sumber untuk pratinjau: kalau mode H.264 aktif, JPEG tidak didekode di sumber (dekode saat perlu)."""
    cfg = dict(source_cfg)
    if (cfg.get("h264_mux") or {}).get("enabled", False):
        cfg["h264_mux"] = dict(cfg["h264_mux"], decode=False)
    return cfg


class FrontPreview:
    def __init__(self, source_cfg: dict, cfg: dict, web=None, recorder=None, h264_sinks=None,
                 source_factory=None):
        self.source_cfg = preview_source_cfg(source_cfg)
        self.rotate = int(cfg.get("rotate", 0))
        self.retry_s = float(cfg.get("retry_s", 5))
        self.web = web
        self.recorder = recorder            # MJPG (cuplik ~fps); cadangan kalau kamera tidak mengirim H.264
        self.h264_sinks = [s for s in (h264_sinks or []) if s is not None]   # rekaman / streaming H.264
        self.frames = 0
        self.decoded = 0                    # berapa frame yang benar-benar didekode (diagnosis CPU)
        self.connected = False
        self._src = None
        self._mode: str | None = None
        if source_factory is None:
            from src.capture import make_source
            source_factory = make_source
        self._make_source = source_factory
        self._stop = threading.Event()
        if web is not None:
            web.add_channel("front", cfg.get("label", "Kamera depan"))
        self._thread = threading.Thread(target=self._run, name="front-preview", daemon=True)
        self._thread.start()

    def _handle(self, fr) -> None:
        from src.capture.uvch264 import decode_frame
        au = fr.meta.get("h264")
        alive = [s for s in self.h264_sinks if getattr(s, "stopped_reason", None) is None]
        if au is not None:
            for s in alive:
                s.push(au, fr.ts_ms, fr.meta.get("h264_idr"))
        # rekaman H.264 berjalan -> MJPG tidak perlu; kamera tanpa H.264 atau perekam H.264 berhenti -> MJPG
        h264_recording = au is not None and any(getattr(s, "records", False) for s in alive)
        if h264_recording:
            self._log_mode("H.264 dari kamera")
        need_rec = not h264_recording and self.recorder is not None and self.recorder.wants(fr.ts_ms)
        need_web = self.web is not None and self.web.wants("front")
        if need_rec or need_web:
            img = decode_frame(fr)
            if img is None:
                return
            self.decoded += 1
            img = rotate(img, self.rotate)
            if need_rec:
                self._log_mode("MJPG")
                self.recorder.push(img, fr.ts_ms)
            if need_web:
                self.web.publish(img, "front")

    def _run(self) -> None:
        while not self._stop.is_set():
            try:
                src = self._make_source(self.source_cfg)
            except Exception as e:  # kamera tidak ada / sedang dipakai
                if self.connected or self.frames == 0:
                    print(f"[depan] kamera tidak bisa dibuka: {e}; coba lagi tiap {self.retry_s:.0f} s", flush=True)
                self.connected = False
                self._stop.wait(self.retry_s)
                continue
            self.connected = True
            self._src = src
            print("[depan] kamera depan tersambung", flush=True)
            with src:
                for fr in src:
                    if self._stop.is_set():
                        break
                    self._handle(fr)
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
        """Dipanggil sink H.264 saat segmen jatuh tempo: minta keyframe agar segmen cepat berganti."""
        src = self._src
        if src is not None and hasattr(src, "request_idr"):
            src.request_idr()

    def close(self, timeout_s: float = 3.0) -> None:
        self._stop.set()
        self._thread.join(timeout_s)
        for rec in [self.recorder, *self.h264_sinks]:
            if rec is not None:
                rec.close()
