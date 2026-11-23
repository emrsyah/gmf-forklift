"""Streaming H.264 kamera (tanpa encode ulang) ke server RTSP, mis. MediaMTX di Jetson.

MediaMTX lalu menyajikan stream yang sama ke laptop lewat RTSP / WebRTC (browser) / HLS, dan bisa
meneruskannya ke server lain (SRT/RTSP) untuk command center. Bagian kita cuma mendorong aliran:
ffmpeg `-c copy` ke `rtsp://…/front` (TCP; aman lewat NAT/5G).

- Mulai (dan mulai ulang setelah putus) di keyframe IDR, diawali SPS/PPS yang disimpan dari stream.
- Server belum jalan / putus -> coba lagi tiap `retry_s`; frame selama itu dibuang (live, bukan arsip).
- Antrean penuh -> frame dibuang dan menunggu IDR berikutnya (tanpa gambar rusak di penonton).
- RTSP tidak membawa metadata rotasi: penonton memutar sendiri (rotasi pasang ada di front_preview.rotate).
"""
from __future__ import annotations

import queue
import shutil
import subprocess
import threading
import time
from typing import Callable, Optional

from src.capture.uvch264 import nal_units


class _FfmpegRtsp:
    def __init__(self, ffmpeg: str, url: str, transport: str = "tcp", extra_args: list | None = None):
        # probesize kecil + analyzeduration 0: SPS sudah cukup; default ffmpeg menahan ~5 s input dulu.
        # JANGAN -fflags nobuffer: diuji (ffmpeg 9) separuh frame hilang di penerima.
        cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", "-use_wallclock_as_timestamps", "1",
               "-probesize", "32768", "-analyzeduration", "0",
               "-f", "h264", "-i", "pipe:0", "-c", "copy", *(extra_args or []),
               "-f", "rtsp", "-rtsp_transport", transport, url]
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                                     stderr=subprocess.PIPE)

    def alive(self) -> bool:
        return self.proc.poll() is None

    def write(self, data: bytes) -> None:
        self.proc.stdin.write(data)
        self.proc.stdin.flush()

    def error_text(self) -> str:
        try:
            return (self.proc.stderr.read() or b"").decode(errors="replace").strip()[-300:]
        except (OSError, ValueError):
            return ""

    def close(self, timeout_s: float = 5.0) -> None:
        try:
            self.proc.stdin.close()
        except OSError:
            pass
        try:
            self.proc.wait(timeout_s)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait()


class H264RtspPublisher:
    def __init__(self, cfg: dict, connect: Optional[Callable[[], object]] = None,
                 on_restart: Optional[Callable[[], None]] = None):
        self.url = str(cfg.get("url", "rtsp://127.0.0.1:8554/front"))
        self.retry_s = float(cfg.get("retry_s", 5))
        self.on_restart = on_restart        # minta IDR ke kamera supaya penonton cepat dapat gambar
        if connect is None:
            ff = shutil.which(cfg.get("ffmpeg", "ffmpeg"))
            if ff is None:
                raise RuntimeError("ffmpeg tidak ditemukan; streaming H.264 butuh ffmpeg")
            transport = str(cfg.get("transport", "tcp"))
            args = cfg.get("ffmpeg_args") or []
            connect = lambda: _FfmpegRtsp(ff, self.url, transport, args)  # noqa: E731
        self._connect = connect
        self._q: queue.Queue = queue.Queue(maxsize=int(cfg.get("max_queue", 30)))
        self._resync = True
        self._idr_asked = False
        self._sps: bytes | None = None
        self._pps: bytes | None = None
        self._conn = None
        self._next_try = 0.0
        self.sent = 0
        self.dropped = 0
        self.connects = 0
        self.connected = False
        self._thread = threading.Thread(target=self._run, name="h264-rtsp", daemon=True)
        self._thread.start()

    def push(self, au: bytes | None, ts_ms: int, idr: bool | None = None) -> None:
        if not au:
            return
        try:
            self._q.put_nowait((au, idr))
        except queue.Full:
            self.dropped += 1
            self._resync = True

    def close(self, timeout_s: float = 5.0) -> None:
        self._q.put(None)
        self._thread.join(timeout_s)

    def _drop_conn(self, why: str) -> None:
        if self._conn is not None:
            err = getattr(self._conn, "error_text", lambda: "")()
            try:
                self._conn.close()
            except Exception:  # noqa: BLE001
                pass
            self._conn = None
            print(f"[stream] {self.url} terputus ({why}{': ' + err if err else ''}); coba lagi tiap "
                  f"{self.retry_s:.0f} s", flush=True)
        self.connected = False
        self._resync = True
        self._next_try = time.monotonic() + self.retry_s

    def _run(self) -> None:
        while True:
            item = self._q.get()
            if item is None:
                break
            au, idr_hint = item
            units = nal_units(au)
            types = {t for t, _, _ in units}
            for t, s, e in units:
                if t == 7:
                    self._sps = au[s:e]
                elif t == 8:
                    self._pps = au[s:e]
            idr = idr_hint if idr_hint is not None else 5 in types
            if self._conn is not None and not getattr(self._conn, "alive", lambda: True)():
                self._drop_conn("ffmpeg berhenti")
            if self._resync:
                if not idr or self._sps is None or self._pps is None:
                    self.dropped += 1
                    if not self._idr_asked and self.on_restart is not None:
                        self._idr_asked = True      # sekali per putus: jangan banjiri kamera
                        try:
                            self.on_restart()
                        except Exception:  # noqa: BLE001
                            pass
                    continue
            if self._conn is None:
                if time.monotonic() < self._next_try:
                    self.dropped += 1
                    continue
                try:
                    self._conn = self._connect()
                except OSError as e:
                    self._conn = None
                    self._next_try = time.monotonic() + self.retry_s
                    print(f"[stream] tidak bisa memulai ffmpeg: {e}", flush=True)
                    continue
                self.connects += 1
                print(f"[stream] mendorong H.264 ke {self.url}", flush=True)
            prefix = b""
            if self._resync and 7 not in types:
                prefix = self._sps + self._pps
            try:
                self._conn.write(prefix + au)
            except (OSError, ValueError) as e:
                self._drop_conn(str(e))
                continue
            self._resync = False
            self._idr_asked = False
            self.connected = True
            self.sent += 1
        if self._conn is not None:
            self._conn.close()
            self._conn = None
