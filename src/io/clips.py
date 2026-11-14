"""Snapshot + klip MP4 per event (K7, L8).

Ring buffer ~pre_s detik (JPEG di RAM, fps rekaman). Saat event `start`:
- snapshot JPG langsung (dengan teks waktu + kode),
- klip = pre_s sebelum + post_s sesudah, di-encode di thread terpisah:
  H.264 yuv420p + `-movflags +faststart` (ffmpeg) agar bisa diputar langsung di browser.
Fallback ke encoder OpenCV (mp4v) jika ffmpeg tidak ada — catat bahwa mp4v belum tentu bisa diputar browser.
"""
from __future__ import annotations

import shutil
import subprocess
import threading
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from src.config import resolve_path


def stamp(img: np.ndarray, ts_ms: int, label: str) -> np.ndarray:
    out = img.copy()
    t = datetime.fromtimestamp(ts_ms / 1000).strftime("%Y-%m-%d %H:%M:%S.") + f"{ts_ms % 1000:03d}"
    txt = f"{t}  {label}"
    cv2.rectangle(out, (0, 0), (out.shape[1], 28), (0, 0, 0), -1)
    cv2.putText(out, txt, (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 1, cv2.LINE_AA)
    return out


class _PendingClip:
    def __init__(self, event: dict, frames: list, until_ms: int, clip_path: Path):
        self.event = event
        self.frames = frames          # list[(ts_ms, jpeg_bytes)]
        self.until_ms = until_ms
        self.clip_path = clip_path


class ClipRecorder:
    def __init__(self, cfg: dict, root: str | Path = "events"):
        self.c = cfg
        self.root = resolve_path(root)
        self.pre_ms = float(cfg.get("pre_s", 10)) * 1000
        self.post_ms = float(cfg.get("post_s", 5)) * 1000
        self.fps = float(cfg.get("fps", 15))
        self.max_w = int(cfg.get("width", 1280))
        self.quality = int(cfg.get("jpeg_quality", 80))
        self.ffmpeg = shutil.which(cfg.get("ffmpeg", "ffmpeg"))
        self.buf: deque = deque()
        self.pending: list[_PendingClip] = []
        self.threads: list[threading.Thread] = []
        self._last_keep_ms: int | None = None
        self.label = ""
        self.done_paths: list[Path] = []

    def _encode_jpg(self, img: np.ndarray) -> bytes:
        if img.shape[1] > self.max_w:
            k = self.max_w / img.shape[1]
            img = cv2.resize(img, (self.max_w, int(img.shape[0] * k)), interpolation=cv2.INTER_AREA)
        ok, b = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, self.quality])
        return b.tobytes() if ok else b""

    def push(self, img: np.ndarray, ts_ms: int, label: str = "") -> None:
        """Panggil setiap frame (gambar dengan overlay jika ingin). Disubsample ke fps rekaman."""
        step = 1000.0 / self.fps
        if self._last_keep_ms is not None and ts_ms - self._last_keep_ms < step * 0.95:
            self._check_pending(ts_ms)
            return
        self._last_keep_ms = ts_ms
        jpg = self._encode_jpg(stamp(img, ts_ms, label or self.label))
        self.buf.append((ts_ms, jpg))
        while self.buf and self.buf[0][0] < ts_ms - self.pre_ms:
            self.buf.popleft()
        for p in self.pending:
            p.frames.append((ts_ms, jpg))
        self._check_pending(ts_ms)

    def _check_pending(self, ts_ms: int):
        ready = [p for p in self.pending if ts_ms >= p.until_ms]
        for p in ready:
            self.pending.remove(p)
            th = threading.Thread(target=self._write_clip, args=(p,), daemon=True)
            th.start()
            self.threads.append(th)

    def paths_for(self, event: dict) -> tuple[Path, Path]:
        d = datetime.fromtimestamp(event["ts_event_ms"] / 1000).strftime("%Y-%m-%d")
        folder = self.root / d
        folder.mkdir(parents=True, exist_ok=True)
        return folder / f"{event['event_id']}.jpg", folder / f"{event['event_id']}.mp4"

    def on_event_start(self, event: dict, img: np.ndarray) -> dict:
        """Simpan snapshot sekarang, jadwalkan klip. Mengembalikan path relatif untuk event JSON."""
        snap, clip = self.paths_for(event)
        label = f"{event['code']} {event['level']}"
        cv2.imwrite(str(snap), stamp(img, event["ts_event_ms"], label))
        frames = list(self.buf)
        self.pending.append(_PendingClip(event, frames, int(event["ts_event_ms"] + self.post_ms), clip))
        rel = lambda p: p.relative_to(self.root.parent).as_posix() if self.root.parent in p.parents else str(p)
        return {"snapshot_path": rel(snap), "clip_path": rel(clip)}

    def _write_clip(self, p: _PendingClip):
        if not p.frames:
            return
        try:
            if self.ffmpeg:
                cmd = [self.ffmpeg, "-y", "-loglevel", "error", "-f", "image2pipe", "-c:v", "mjpeg",
                       "-framerate", str(self.fps), "-i", "-",
                       "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2,scale=out_range=tv,format=yuv420p",
                       "-c:v", "libx264", "-preset", "ultrafast", "-pix_fmt", "yuv420p",
                       "-color_range", "tv", "-movflags", "+faststart", str(p.clip_path)]
                proc = subprocess.Popen(cmd, stdin=subprocess.PIPE)
                for _, jpg in p.frames:
                    proc.stdin.write(jpg)
                proc.stdin.close()
                proc.wait(timeout=120)
            else:
                first = cv2.imdecode(np.frombuffer(p.frames[0][1], np.uint8), cv2.IMREAD_COLOR)
                h, w = first.shape[:2]
                vw = cv2.VideoWriter(str(p.clip_path), cv2.VideoWriter_fourcc(*"mp4v"), self.fps, (w, h))
                for _, jpg in p.frames:
                    vw.write(cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR))
                vw.release()
            self.done_paths.append(p.clip_path)
        except Exception as e:  # jangan pernah menjatuhkan pipeline karena klip
            print(f"[clips] gagal menulis {p.clip_path}: {e}")

    def flush(self, timeout: float = 120.0):
        """Tulis semua klip yang masih menunggu (akhir sesi) dan tunggu thread selesai."""
        for p in list(self.pending):
            self.pending.remove(p)
            th = threading.Thread(target=self._write_clip, args=(p,), daemon=True)
            th.start()
            self.threads.append(th)
        for th in self.threads:
            th.join(timeout)
        self.threads = []
