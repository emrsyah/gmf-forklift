"""Snapshot + klip MP4 per event (K7, L8).

Ring buffer ~pre_s detik (JPEG di RAM, fps rekaman). Saat event `start`:
- snapshot JPG langsung (dengan teks waktu + kode),
- klip = pre_s sebelum event sampai kondisi selesai + post_end_s (clip_until=end, maks max_s), atau
  pre_s sebelum + post_s sesudah (clip_until=fixed), di-encode di thread terpisah:
  H.264 yuv420p + `-movflags +faststart` (ffmpeg) agar bisa diputar langsung di browser.
Fallback ke encoder OpenCV (mp4v) jika ffmpeg tidak ada — catat bahwa mp4v belum tentu bisa diputar browser.
"""
from __future__ import annotations

import os
import queue
import shutil
import subprocess
import threading
from collections import deque
from datetime import datetime
from pathlib import Path

import cv2
import numpy as np

from src.config import resolve_path

_START, _END = "__event_start__", "__event_end__"   # penanda di antrean encode (defer_event_start)


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
        # "end" (brief MINERVA): klip = pre_s sebelum event sampai kondisi selesai (+ post_end_s), maks max_s.
        # "fixed" (lama): pre_s sebelum + post_s sesudah event start.
        self.until = cfg.get("clip_until", "fixed")
        self.post_end_ms = float(cfg.get("post_end_s", 2)) * 1000
        self.max_ms = float(cfg.get("max_s", 60)) * 1000
        # kode yang snapshot-nya diambil dari buffer SEBELUM kondisi mulai (D-09: frame terakhir sebelum lensa
        # tertutup, bukan gambar tangan/hitam)
        self.pre_snap_codes = set(cfg.get("snapshot_before_condition_codes", ["D09"]))
        self.pre_snap_ms = float(cfg.get("snapshot_before_condition_s", 0.5)) * 1000
        self.fps = float(cfg.get("fps", 15))
        self.max_w = int(cfg.get("width", 1280))
        self.quality = int(cfg.get("jpeg_quality", 80))
        self.ffmpeg = shutil.which(cfg.get("ffmpeg", "ffmpeg"))
        # batasi bitrate (klip diunduh lewat uplink 5G yang sama dengan alarm) dan thread encoder (CPU CV)
        self.x264_args = [str(a) for a in cfg.get("x264_args", ["-crf", "26", "-maxrate", "1500k",
                                                                "-bufsize", "3000k", "-threads", "2"])]
        self.encoder_nice = int(cfg.get("encoder_nice", 0))
        if self.ffmpeg is None:
            print("[clips] PERINGATAN: ffmpeg tidak ada -> klip mp4v (belum tentu bisa diputar browser, K7)",
                  flush=True)
        self.buf: deque = deque()
        self.pending: list[_PendingClip] = []
        self.threads: list[threading.Thread] = []
        self._last_keep_ms: int | None = None
        self.label = ""
        self.done_paths: list[Path] = []
        # dipanggil (dari thread encoder) setelah klip selesai: dict info media -> mis. sinks.media
        self.on_done = None
        # Stempel + resize + JPEG (~6 ms/frame di Jetson) di thread sendiri, bukan di loop deteksi.
        # Antrean dikuras (join) saat event start/end dan flush, jadi isi klip sama dengan mode sinkron.
        self._lock = threading.RLock()
        self.defer_start = bool(cfg.get("defer_event_start", False))
        self._q: queue.Queue | None = None
        if cfg.get("async_encode", True):
            self._q = queue.Queue(maxsize=int(cfg.get("encode_queue", 30)))
            threading.Thread(target=self._encode_loop, name="clip-jpeg", daemon=True).start()

    def _encode_jpg(self, img: np.ndarray) -> bytes:
        if img.shape[1] > self.max_w:
            k = self.max_w / img.shape[1]
            img = cv2.resize(img, (self.max_w, int(img.shape[0] * k)), interpolation=cv2.INTER_AREA)
        ok, b = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, self.quality])
        return b.tobytes() if ok else b""

    def wants(self, ts_ms: int) -> bool:
        """True jika frame pada `ts_ms` akan disimpan (pemanggil bisa melewati gambar overlay kalau False)."""
        return self._last_keep_ms is None or ts_ms - self._last_keep_ms >= 1000.0 / self.fps * 0.95

    def push(self, img: np.ndarray | None, ts_ms: int, label: str = "") -> None:
        """Panggil setiap frame (gambar dengan overlay jika ingin). Disubsample ke fps rekaman."""
        if not self.wants(ts_ms) or img is None:
            if self._q is None:      # mode async: dicek thread encoder setelah frame antrean masuk klip
                self._check_pending(ts_ms)
            return
        self._last_keep_ms = ts_ms
        if self._q is None:
            self._add(ts_ms, img, label or self.label)
            return
        # antrean penuh -> tunggu (tidak dibuang): replay lebih cepat dari real-time tetap dapat klip utuh,
        # dan tidak pernah lebih lambat dari encode sinkron
        self._q.put((ts_ms, img, label or self.label))

    def _encode_loop(self):
        while True:
            a, b, c = self._q.get()
            try:
                if a == _START:                # penanda event (mode defer_event_start), urut dengan frame
                    self._start_now(b, c)
                elif a == _END:
                    self._end_now(b)
                else:
                    self._add(a, b, c)
            except Exception as e:  # jangan pernah mematikan thread klip
                print(f"[clips] encode gagal: {e}", flush=True)
            finally:
                self._q.task_done()

    def _drain(self):
        if self._q is not None:
            self._q.join()

    def _add(self, ts_ms: int, img: np.ndarray, label: str):
        jpg = self._encode_jpg(stamp(img, ts_ms, label))
        with self._lock:
            self.buf.append((ts_ms, jpg))
            while self.buf and self.buf[0][0] < ts_ms - self.pre_ms:
                self.buf.popleft()
            for p in self.pending:
                p.frames.append((ts_ms, jpg))
            self._check_pending(ts_ms)

    def _check_pending(self, ts_ms: int):
        with self._lock:
            ready = [p for p in self.pending if ts_ms >= p.until_ms]
            if ready:
                self.threads = [t for t in self.threads if t.is_alive()]
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

    def _pre_condition_jpg(self, event: dict) -> bytes | None:
        """Frame buffer terakhir sebelum kondisi mulai (dikurangi margin), atau None."""
        t0 = event.get("ts_condition_start_ms") or event["ts_event_ms"]
        cut = t0 - self.pre_snap_ms
        best = None
        for ts, jpg in self.buf:
            if ts <= cut:
                best = jpg
        return best

    def on_event_start(self, event: dict, img: np.ndarray) -> dict:
        """Simpan snapshot sekarang, jadwalkan klip. Mengembalikan path relatif untuk event JSON.
        defer_event_start (butuh async_encode): tidak menunggu antrean JPEG dikuras dan tidak menulis snapshot di
        loop deteksi -> event langsung terkirim (K5). Penanda masuk antrean yang sama dengan frame, jadi isi
        klip/snapshot sama dengan mode sinkron; path sudah pasti (paths_for). snapshot_before_condition = None
        (belum diketahui saat event dikirim)."""
        if self.defer_start and self._q is not None:
            snap, clip = self.paths_for(event)
            self._q.put((_START, dict(event), img.copy()))
            return {"snapshot_path": self._rel(snap), "clip_path": self._rel(clip),
                    "snapshot_before_condition": None}
        self._drain()
        return self._start_now(event, img)

    def _start_now(self, event: dict, img: np.ndarray) -> dict:
        snap, clip = self.paths_for(event)
        label = f"{event['code']} {event['level']}"
        with self._lock:
            pre = self._pre_condition_jpg(event) if event.get("code") in self.pre_snap_codes else None
            frames = [f for f in self.buf if f[0] >= event["ts_event_ms"] - self.pre_ms]
            t_first = frames[0][0] if frames else event["ts_event_ms"]
            if self.until == "end":
                until = t_first + self.max_ms           # batas atas; dipendekkan oleh on_event_end
            else:
                until = event["ts_event_ms"] + self.post_ms
            pc = _PendingClip(event, frames, int(until), clip)
            pc.snap_path = snap
            self.pending.append(pc)
        if pre is not None:
            snap.write_bytes(pre)
        elif not cv2.imwrite(str(snap), stamp(img, event["ts_event_ms"], label)):
            print(f"[clips] snapshot gagal ditulis: {snap}")
        return {"snapshot_path": self._rel(snap), "clip_path": self._rel(clip),
                "snapshot_before_condition": pre is not None}

    def _rel(self, p: Path) -> str:
        return p.relative_to(self.root.parent).as_posix() if self.root.parent in p.parents else str(p)

    def on_event_end(self, event: dict) -> None:
        """Mode clip_until=end: klip berhenti post_end_s setelah kondisi selesai (tetap <= max_s)."""
        if self.until != "end":
            return
        if self.defer_start and self._q is not None:   # urut sesudah penanda start yang mungkin masih antre
            self._q.put((_END, dict(event), None))
            return
        self._end_now(event)

    def _end_now(self, event: dict) -> None:
        with self._lock:
            for p in self.pending:
                if p.event.get("event_id") == event.get("event_id"):
                    p.until_ms = min(p.until_ms, int(event["ts_event_ms"] + self.post_end_ms))

    def clip_fps(self, frames: list) -> float:
        """fps NYATA klip. Jika pipeline lebih lambat dari fps rekaman (mis. 9 fps di CPU), frame lebih sedikit;
        encode dengan fps nominal akan mempercepat video dan memendekkan durasinya (bukti K7 jadi salah)."""
        if len(frames) < 2:
            return self.fps
        span_s = (frames[-1][0] - frames[0][0]) / 1000.0
        if span_s <= 0:
            return self.fps
        return max(1.0, min(self.fps, (len(frames) - 1) / span_s))

    def _write_clip(self, p: _PendingClip):
        if not p.frames:
            return
        fps = self.clip_fps(p.frames)
        # tulis ke file sementara lalu rename: dashboard/HTTP tidak pernah melihat MP4 setengah jadi
        tmp = p.clip_path.with_name(p.clip_path.stem + ".part.mp4")
        ok = False
        proc = None
        try:
            if self.ffmpeg:
                cmd = [self.ffmpeg, "-y", "-loglevel", "error", "-f", "image2pipe", "-c:v", "mjpeg",
                       "-framerate", f"{fps:.3f}", "-i", "-",
                       "-vf", "pad=ceil(iw/2)*2:ceil(ih/2)*2,scale=out_range=tv,format=yuv420p",
                       "-c:v", "libx264", "-preset", "ultrafast", *self.x264_args, "-pix_fmt", "yuv420p",
                       "-color_range", "tv", "-movflags", "+faststart", str(tmp)]
                from src.io.proc import popen_nice
                # nice: encoder klip kalah prioritas dari deteksi (tanpa ini FPS turun tajam setelah alarm)
                proc = popen_nice(cmd, self.encoder_nice, stdin=subprocess.PIPE)
                for _, jpg in p.frames:
                    proc.stdin.write(jpg)
                proc.stdin.close()
                ok = proc.wait(timeout=120) == 0
            else:
                first = cv2.imdecode(np.frombuffer(p.frames[0][1], np.uint8), cv2.IMREAD_COLOR)
                h, w = first.shape[:2]
                vw = cv2.VideoWriter(str(tmp), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
                for _, jpg in p.frames:
                    vw.write(cv2.imdecode(np.frombuffer(jpg, np.uint8), cv2.IMREAD_COLOR))
                vw.release()
                ok = True
            ok = ok and tmp.exists() and tmp.stat().st_size > 0
            if ok:
                os.replace(tmp, p.clip_path)
                self.done_paths.append(p.clip_path)
            else:
                print(f"[clips] encoder gagal untuk {p.clip_path}")
        except Exception as e:  # jangan pernah menjatuhkan pipeline karena klip
            print(f"[clips] gagal menulis {p.clip_path}: {e}")
            ok = False
            if proc is not None and proc.poll() is None:
                proc.kill()
        finally:
            if not ok and tmp.exists():
                try:
                    tmp.unlink()
                except OSError:
                    pass
        if self.on_done is not None:
            try:
                ev = p.event
                self.on_done({
                    "event_id": ev.get("event_id"), "device_id": ev.get("device_id"),
                    "code": ev.get("code"), "level": ev.get("level"), "ts_event_ms": ev["ts_event_ms"],
                    "snapshot_path": self._rel(getattr(p, "snap_path", p.clip_path.with_suffix(".jpg"))),
                    "clip_path": self._rel(p.clip_path) if ok else None,
                    "clip_ok": ok,
                    "clip_start_ms": p.frames[0][0], "clip_end_ms": p.frames[-1][0],
                    "clip_duration_s": round((p.frames[-1][0] - p.frames[0][0]) / 1000, 2),
                })
            except Exception as e:
                print(f"[clips] callback media gagal: {e}")

    def flush(self, timeout: float = 120.0):
        """Tulis semua klip yang masih menunggu (akhir sesi) dan tunggu thread selesai."""
        self._drain()
        with self._lock:
            for p in list(self.pending):
                self.pending.remove(p)
                th = threading.Thread(target=self._write_clip, args=(p,), daemon=True)
                th.start()
                self.threads.append(th)
            threads = list(self.threads)
        for th in threads:
            th.join(timeout)
        self.threads = []
