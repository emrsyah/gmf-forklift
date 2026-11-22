"""Rekaman kamera kabin terus-menerus untuk pengumpulan data (frame mentah, tanpa overlay).

Segmen video (default MJPG .avi: murah di CPU Jetson, tanpa encoder hardware) + file pendamping
`<video>.ts.csv` (frame,ts_ms) dengan format yang sama seperti tools/recorder.py, jadi segmen bisa
langsung diputar ulang lewat replay harness dengan timestamp asli.

Ditulis di thread sendiri dengan antrean terbatas: kalau disk lambat, frame rekaman dibuang,
deteksi tidak ikut melambat. Berhenti sendiri jika ruang disk < min_free_gb.

Privasi (CLAUDE.md aturan 11): hanya untuk peraga yang setuju direkam; disimpan lokal, jangan diunggah.
"""
from __future__ import annotations

import queue
import shutil
import threading
import time
from pathlib import Path

import cv2
import numpy as np


class ContinuousRecorder:
    def __init__(self, cfg: dict, device_id: str = "dev"):
        self.dir = Path(cfg.get("dir", "recordings"))
        self.segment_s = float(cfg.get("segment_s", 300))
        self.fps = float(cfg.get("fps", 10))
        self.max_side = int(cfg.get("max_side", 960))
        self.codec = str(cfg.get("codec", "MJPG"))
        self.ext = str(cfg.get("ext", "avi"))
        self.quality = int(cfg.get("jpeg_quality", 85))
        self.min_free_gb = float(cfg.get("min_free_gb", 10))
        self.prefix = f"{cfg.get('prefix', 'cabin')}_{device_id}"
        self._size = None
        self._q: queue.Queue = queue.Queue(maxsize=int(cfg.get("max_queue", 60)))
        self._next_ts = -1e18
        self.dropped = 0
        self.frames = 0
        self.stopped_reason: str | None = None
        self._writer = None
        self._ts_file = None
        self._seg_start_ms = None
        self._seg_frames = 0
        self._thread = threading.Thread(target=self._run, name="continuous-rec", daemon=True)
        self._thread.start()

    # ---------------------------------------------------------------- pipeline thread
    def push(self, img: np.ndarray | None, ts_ms: int) -> None:
        """Dipanggil tiap frame; hanya ~fps frame per detik yang diambil."""
        if img is None or self.stopped_reason is not None or ts_ms < self._next_ts:
            return
        period = 1000.0 / self.fps
        # jadwal tetap (tidak bergeser oleh jitter frame); kalau tertinggal jauh, mulai lagi dari frame ini
        self._next_ts = (ts_ms if ts_ms - self._next_ts > period else self._next_ts) + period
        try:
            self._q.put_nowait((img, int(ts_ms)))
        except queue.Full:
            self.dropped += 1

    def close(self, timeout_s: float = 5.0) -> None:
        self._q.put(None)
        self._thread.join(timeout_s)

    # ---------------------------------------------------------------- writer thread
    def free_gb(self) -> float:
        self.dir.mkdir(parents=True, exist_ok=True)
        return shutil.disk_usage(self.dir).free / 1e9

    def _open(self, shape, ts_ms: int) -> bool:
        free = self.free_gb()
        if free < self.min_free_gb:
            self.stopped_reason = f"ruang disk {free:.1f} GB < {self.min_free_gb} GB"
            print(f"[rekam] BERHENTI: {self.stopped_reason}", flush=True)
            return False
        day = time.strftime("%Y-%m-%d", time.localtime(ts_ms / 1000))
        stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime(ts_ms / 1000))
        out = self.dir / day
        out.mkdir(parents=True, exist_ok=True)
        path = out / f"{self.prefix}_{stamp}.{self.ext}"
        h, w = shape[:2]
        wr = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*self.codec), self.fps, (w, h))
        if not wr.isOpened():
            self.stopped_reason = f"VideoWriter {self.codec} gagal membuka {path}"
            print(f"[rekam] BERHENTI: {self.stopped_reason}", flush=True)
            return False
        if self.codec == "MJPG":
            wr.set(cv2.VIDEOWRITER_PROP_QUALITY, self.quality)
        self._writer = wr
        self._ts_file = open(str(path) + ".ts.csv", "w", encoding="utf-8")
        self._ts_file.write("frame,ts_ms\n")
        self._seg_start_ms = ts_ms
        self._seg_frames = 0
        print(f"[rekam] segmen baru {path} ({free:.0f} GB bebas)", flush=True)
        return True

    def _close_segment(self) -> None:
        if self._writer is not None:
            self._writer.release()
            self._writer = None
        if self._ts_file is not None:
            self._ts_file.close()
            self._ts_file = None

    def _run(self) -> None:
        while True:
            item = self._q.get()
            if item is None:
                break
            img, ts_ms = item
            if self.max_side and max(img.shape[:2]) > self.max_side:
                s = self.max_side / max(img.shape[:2])
                img = cv2.resize(img, (round(img.shape[1] * s), round(img.shape[0] * s)), interpolation=cv2.INTER_AREA)
            if self._writer is not None and (ts_ms - self._seg_start_ms >= self.segment_s * 1000
                                             or (img.shape[1], img.shape[0]) != self._size):
                self._close_segment()
            if self._writer is None:
                self._size = (img.shape[1], img.shape[0])
                if not self._open(img.shape, ts_ms):
                    break
            try:
                self._writer.write(img)
                self._ts_file.write(f"{self._seg_frames},{ts_ms}\n")
                self._seg_frames += 1
                self.frames += 1
            except (OSError, cv2.error) as e:
                self.stopped_reason = f"tulis gagal: {e}"
                print(f"[rekam] BERHENTI: {self.stopped_reason}", flush=True)
                break
        self._close_segment()
