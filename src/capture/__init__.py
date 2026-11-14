"""Sumber kamera dengan satu interface untuk live dan replay (aturan kerja 3).

Setiap frame membawa timestamp *sumber* (`ts_ms`), bukan jam dinding saat diproses:
- webcam / URL / GStreamer: jam monotonic saat frame diambil, digeser ke epoch.
- file video: posisi frame di video (CAP_PROP_POS_MSEC / indeks/fps) + epoch awal yang bisa diatur.
Semua indikator memakai `ts_ms` ini, sehingga replay menghasilkan keputusan yang sama dengan live.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Iterator, Optional

import cv2
import numpy as np


@dataclass
class Frame:
    image: np.ndarray          # BGR uint8
    ts_ms: int                 # timestamp sumber (epoch ms)
    index: int                 # nomor frame dari sumber
    video_t_s: float = 0.0     # detik sejak awal sumber (untuk mencocokkan truth.csv)
    meta: dict = field(default_factory=dict)


class FrameSource:
    """Interface sumber frame. Pakai sebagai iterator atau `read()`."""

    fps_hint: float = 30.0

    def read(self) -> Optional[Frame]:
        raise NotImplementedError

    def release(self) -> None:
        pass

    def __iter__(self) -> Iterator[Frame]:
        while True:
            f = self.read()
            if f is None:
                return
            yield f

    def __enter__(self):
        return self

    def __exit__(self, *exc):
        self.release()


_API = {
    "auto": cv2.CAP_ANY,
    "dshow": cv2.CAP_DSHOW,
    "msmf": cv2.CAP_MSMF,
    "v4l2": cv2.CAP_V4L2,
    "gstreamer": cv2.CAP_GSTREAMER,
    "ffmpeg": cv2.CAP_FFMPEG,
}


class _LiveSource(FrameSource):
    """Basis untuk sumber live: timestamp = waktu ambil frame."""

    def __init__(self, cap: cv2.VideoCapture, name: str):
        if not cap.isOpened():
            raise RuntimeError(f"Tidak bisa membuka sumber kamera: {name}")
        self.cap = cap
        self.name = name
        self._i = 0
        self._t0_mono = time.monotonic()
        self._t0_epoch_ms = int(time.time() * 1000)
        fps = cap.get(cv2.CAP_PROP_FPS)
        self.fps_hint = fps if fps and fps > 1 else 30.0

    def read(self) -> Optional[Frame]:
        ok, img = self.cap.read()
        if not ok or img is None:
            return None
        dt = time.monotonic() - self._t0_mono
        f = Frame(img, self._t0_epoch_ms + int(dt * 1000), self._i, dt)
        self._i += 1
        return f

    def release(self) -> None:
        self.cap.release()


class WebcamSource(_LiveSource):
    def __init__(self, index: int = 0, width: int | None = None, height: int | None = None,
                 fps: float | None = None, api: str = "auto"):
        cap = cv2.VideoCapture(int(index), _API.get(api, cv2.CAP_ANY))
        if width:
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        if height:
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        if fps:
            cap.set(cv2.CAP_PROP_FPS, fps)
        super().__init__(cap, f"webcam:{index}")


class UrlSource(_LiveSource):
    """Stream RTSP/HTTP, misalnya aplikasi IP camera di HP."""

    def __init__(self, url: str):
        super().__init__(cv2.VideoCapture(url, cv2.CAP_FFMPEG), url)


class GStreamerSource(_LiveSource):
    """Fase 2 (Jetson): pipeline nvarguscamerasrc / v4l2src yang diakhiri appsink."""

    def __init__(self, pipeline: str):
        super().__init__(cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER), pipeline)


def sidecar_ts_path(video: str | Path) -> Path:
    """File pendamping timestamp per frame yang ditulis tools/recorder.py: <video>.ts.csv"""
    return Path(str(video) + ".ts.csv")


def load_sidecar_ts(video: str | Path) -> tuple[int | None, list[float] | None]:
    """(epoch_ms frame pertama, list detik relatif per frame) atau (None, None)."""
    p = sidecar_ts_path(video)
    if not p.exists():
        return None, None
    ts = []
    with open(p, encoding="utf-8") as f:
        next(f, None)   # header: frame,ts_ms
        for line in f:
            parts = line.strip().split(",")
            if len(parts) >= 2:
                ts.append(int(parts[1]))
    if not ts:
        return None, None
    return ts[0], [(t - ts[0]) / 1000.0 for t in ts]


class VideoFileSource(FrameSource):
    """File video. `realtime=True` memutar sesuai timestamp asli (dipakai replay harness).

    Jika ada file pendamping `<video>.ts.csv` (dari recorder), timestamp asli per frame dipakai,
    sehingga FPS webcam yang naik-turun saat merekam tetap direplay dengan waktu yang benar.
    """

    def __init__(self, path: str | Path, realtime: bool = False, start_epoch_ms: int | None = None):
        self.path = str(path)
        self.cap = cv2.VideoCapture(self.path)
        if not self.cap.isOpened():
            raise RuntimeError(f"Tidak bisa membuka video: {self.path}")
        fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.fps_hint = fps if fps and fps > 1 else 30.0
        self.realtime = realtime
        side_epoch, self._side_ts = load_sidecar_ts(self.path)
        if start_epoch_ms is not None:
            self.start_epoch_ms = int(start_epoch_ms)
        elif side_epoch is not None:
            self.start_epoch_ms = side_epoch
        else:
            self.start_epoch_ms = int(time.time() * 1000)
        self._i = 0
        self._wall0: float | None = None
        self.frame_count = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    def read(self) -> Optional[Frame]:
        ok, img = self.cap.read()
        if not ok or img is None:
            return None
        if self._side_ts is not None and self._i < len(self._side_ts):
            t_s = self._side_ts[self._i]
        else:
            pos_ms = self.cap.get(cv2.CAP_PROP_POS_MSEC)
            # Beberapa backend mengembalikan 0 untuk POS_MSEC; fallback ke indeks/fps.
            t_s = pos_ms / 1000.0 if pos_ms and pos_ms > 0 else self._i / self.fps_hint
        if self.realtime:
            if self._wall0 is None:
                self._wall0 = time.monotonic() - t_s
            wait = self._wall0 + t_s - time.monotonic()
            if wait > 0:
                time.sleep(wait)
        f = Frame(img, self.start_epoch_ms + int(round(t_s * 1000)), self._i, t_s)
        self._i += 1
        return f

    def release(self) -> None:
        self.cap.release()


class ArraySource(FrameSource):
    """Sumber dari list gambar + fps tetap. Dipakai di unit test."""

    def __init__(self, images: list[np.ndarray], fps: float = 30.0, start_epoch_ms: int = 0):
        self.images = images
        self.fps_hint = fps
        self.start_epoch_ms = start_epoch_ms
        self._i = 0

    def read(self) -> Optional[Frame]:
        if self._i >= len(self.images):
            return None
        t_s = self._i / self.fps_hint
        f = Frame(self.images[self._i], self.start_epoch_ms + int(round(t_s * 1000)), self._i, t_s)
        self._i += 1
        return f


def make_source(cfg: dict) -> FrameSource:
    """Buat sumber dari blok config `source:` (runtime.yaml)."""
    t = cfg.get("type", "webcam")
    if t == "webcam":
        return WebcamSource(cfg.get("index", 0), cfg.get("width"), cfg.get("height"),
                            cfg.get("fps"), cfg.get("api", "auto"))
    if t == "file":
        return VideoFileSource(cfg["path"], realtime=cfg.get("realtime", True),
                               start_epoch_ms=cfg.get("start_epoch_ms"))
    if t == "url":
        return UrlSource(cfg["url"])
    if t == "gstreamer":
        return GStreamerSource(cfg["pipeline"])
    raise ValueError(f"Tipe sumber tidak dikenal: {t}")


def parse_source_arg(arg: str) -> dict:
    """CLI `--source`: 'webcam', 'webcam:1', path video, atau URL rtsp/http."""
    if arg == "webcam":
        return {"type": "webcam", "index": 0}
    if arg.startswith("webcam:"):
        return {"type": "webcam", "index": int(arg.split(":", 1)[1])}
    if arg.isdigit():
        return {"type": "webcam", "index": int(arg)}
    if arg.startswith(("rtsp://", "http://", "https://", "udp://")):
        return {"type": "url", "url": arg}
    if "!" in arg and "appsink" in arg:
        return {"type": "gstreamer", "pipeline": arg}
    return {"type": "file", "path": arg}
