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

    def clock_correction_ms(self) -> int:
        """Jam dinding sekarang - jam sumber (`ts_ms`). ~0 kecuali jam sistem disetel setelah sumber dibuka
        (mis. NTP sinkron belakangan di Jetson tanpa RTC). `ts_ms` frame sengaja tetap monotonic (indikator butuh
        selisih waktu yang mulus); koreksi ini hanya untuk timestamp event yang dikirim (K5)."""
        return epoch_correction_ms(self._t0_mono, self._t0_epoch_ms)

    def release(self) -> None:
        self.cap.release()


def epoch_correction_ms(t0_mono: float, t0_epoch_ms: int) -> int:
    now_src = t0_epoch_ms + (time.monotonic() - t0_mono) * 1000
    return int(round(time.time() * 1000 - now_src))


class WebcamSource(_LiveSource):
    def __init__(self, index: int = 0, width: int | None = None, height: int | None = None,
                 fps: float | None = None, api: str = "auto", fourcc: str | None = None,
                 autofocus: bool | None = None, focus: float | None = None):
        # Format/ukuran lewat parameter konstruktor: di DirectShow, set(FOURCC) sesudah open sering diabaikan
        # (C930e tetap YUY2 -> 1080p cuma ~5 fps lewat USB). MJPG membuka 1080p/720p penuh.
        params: list[int] = []
        if fourcc:
            params += [cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*fourcc)]
        if width:
            params += [cv2.CAP_PROP_FRAME_WIDTH, int(width)]
        if height:
            params += [cv2.CAP_PROP_FRAME_HEIGHT, int(height)]
        if fps:
            params += [cv2.CAP_PROP_FPS, int(fps)]
        api_id = _API.get(api, cv2.CAP_ANY)
        cap = cv2.VideoCapture(int(index), api_id, params) if params else cv2.VideoCapture(int(index), api_id)
        # Kamera depan: fokus dikunci supaya kalibrasi intrinsik (estimasi jarak) tidak bergeser.
        if autofocus is not None:
            cap.set(cv2.CAP_PROP_AUTOFOCUS, 1 if autofocus else 0)
        if focus is not None:
            cap.set(cv2.CAP_PROP_FOCUS, float(focus))
        super().__init__(cap, f"webcam:{index}")


class UrlSource(_LiveSource):
    """Stream RTSP/HTTP, misalnya aplikasi IP camera di HP."""

    def __init__(self, url: str):
        super().__init__(cv2.VideoCapture(url, cv2.CAP_FFMPEG), url)


class GStreamerSource(_LiveSource):
    """Fase 2 (Jetson): pipeline nvarguscamerasrc / v4l2src yang diakhiri appsink."""

    def __init__(self, pipeline: str, gray_range: str = "full"):
        super().__init__(cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER), pipeline)
        # nvvidconv format=GRAY8 = luma Y studio range (16..235). Diregangkan ke 0..255 supaya nilai abu-abu
        # sama dengan cvtColor(BGR->GRAY) jalur lama (ambang D-09, input CLAHE). Uji 2026-11-23 di Jetson:
        # BGR p1/p99 = 31/253, GRAY8 mentah 43/234 -> setelah regang 31/254.
        self._lut = None
        if gray_range == "limited":
            x = np.arange(256, dtype=np.float32)
            self._lut = np.clip((x - 16.0) * 255.0 / 219.0, 0, 255).round().astype(np.uint8)

    def read(self) -> Optional[Frame]:
        f = super().read()
        if f is not None and self._lut is not None and f.image.ndim == 2:
            f.image = cv2.LUT(f.image, self._lut)
        return f


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


def device_index(device: str) -> int:
    """`/dev/v4l/by-id/...` (glob boleh) -> nomor /dev/videoN saat ini. Nomor video bisa berubah setelah
    reboot/colok ulang (Jetson 2026-11-23: C930e pindah video1 -> video0), path by-id tidak."""
    import glob
    import os
    import re

    hits = sorted(glob.glob(device))
    if not hits:
        raise RuntimeError(f"Kamera tidak ditemukan: {device}")
    m = re.search(r"video(\d+)$", os.path.realpath(hits[0]))
    if not m:
        raise RuntimeError(f"Bukan perangkat video: {hits[0]}")
    return int(m.group(1))


def make_source(cfg: dict) -> FrameSource:
    """Buat sumber dari blok config `source:` (runtime.yaml)."""
    t = cfg.get("type", "webcam")
    if t == "webcam" and cfg.get("device"):            # path stabil menang atas index
        cfg = dict(cfg, index=device_index(cfg["device"]))
    if t == "webcam" and (cfg.get("h264_mux") or {}).get("enabled", False):
        from src.capture.uvch264 import UvcH264WebcamSource   # H.264 dari kamera (C930e), lihat modulnya
        return UvcH264WebcamSource(cfg.get("index", 0), cfg.get("width"), cfg.get("height"),
                                   cfg.get("fps"), cfg.get("api", "auto"), cfg.get("fourcc", "MJPG"),
                                   cfg.get("autofocus"), cfg.get("focus"), cfg.get("h264_mux"))
    if t == "webcam":
        return WebcamSource(cfg.get("index", 0), cfg.get("width"), cfg.get("height"),
                            cfg.get("fps"), cfg.get("api", "auto"), cfg.get("fourcc"),
                            cfg.get("autofocus"), cfg.get("focus"))
    if t == "file":
        return VideoFileSource(cfg["path"], realtime=cfg.get("realtime", True),
                               start_epoch_ms=cfg.get("start_epoch_ms"))
    if t == "url":
        return UrlSource(cfg["url"])
    if t == "gstreamer":
        return GStreamerSource(cfg["pipeline"], cfg.get("gray_range", "full"))
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
