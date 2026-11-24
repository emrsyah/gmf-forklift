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
        # srt://: MPEG-TS lewat SRT (UDP + kirim ulang) -> tahan paket hilang di uplink 5G (RTSP/TCP macet)
        out = (["-f", "mpegts", url] if url.startswith("srt://")
               else ["-f", "rtsp", "-rtsp_transport", transport, url])
        cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", "-use_wallclock_as_timestamps", "1",
               "-probesize", "32768", "-analyzeduration", "0",
               "-f", "h264", "-i", "pipe:0", "-c", "copy", *(extra_args or []), *out]
        import sys
        from src.io.overlay_stream import _die_with_parent
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                                     stderr=subprocess.PIPE,
                                     preexec_fn=_die_with_parent if sys.platform.startswith("linux") else None)
        self._err = b""
        # stderr dibaca terus (proses hidup lama): pipa penuh bisa membuat ffmpeg macet
        threading.Thread(target=self._drain, daemon=True).start()

    def _drain(self) -> None:
        try:
            for line in self.proc.stderr:
                self._err = (self._err + line)[-600:]
        except (OSError, ValueError):
            pass

    def alive(self) -> bool:
        return self.proc.poll() is None

    def write(self, data: bytes) -> None:
        self.proc.stdin.write(data)
        self.proc.stdin.flush()

    def error_text(self) -> str:
        return self._err.decode(errors="replace").strip()[-300:]

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
        # on_demand: hanya mendorong selama ada "sewa" (want) yang belum habis, mis. dari agen MINERVA saat
        # ada penonton di dashboard. Hemat uplink 5G untuk klip kejadian. Sewa habis -> ffmpeg ditutup.
        self.on_demand = bool(cfg.get("on_demand", False))
        self._want_until = 0.0
        # gagal-terbuka: tanpa kabar sewa (on maupun off) selama lease_orphan_s, mis. agen mati atau agen
        # lama yang belum tahu /stream -> dorong terus seperti tanpa on_demand
        self.lease_orphan_s = float(cfg.get("lease_orphan_s", 60))
        self._last_lease = time.monotonic()
        self.idle = 0
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

    def want(self, on: bool, ttl_s: float = 15.0) -> None:
        """Sewa aliran on_demand selama ttl_s (diperbarui berkala oleh pemanggil); on=False menghentikan segera."""
        now = time.monotonic()
        self._last_lease = now
        self._want_until = now + max(0.0, float(ttl_s)) if on else 0.0
        if not on:
            self._idr_asked = False             # saat diminta lagi: minta IDR baru, jangan tunggu IDR alami

    def wanted(self) -> bool:
        now = time.monotonic()
        return not self.on_demand or now < self._want_until or now - self._last_lease > self.lease_orphan_s

    def push(self, au: bytes | None, ts_ms: int, idr: bool | None = None) -> None:
        if not au:
            return
        try:
            self._q.put_nowait((au, idr))
        except queue.Full:
            self.dropped += 1
            self._resync = True

    def close(self, timeout_s: float = 5.0) -> None:
        try:
            self._q.put(None, timeout=timeout_s)    # penulis macet + antrean penuh: jangan menggantung
        except queue.Full:
            pass
        self._thread.join(timeout_s)
        conn = self._conn
        if self._thread.is_alive() and conn is not None:
            conn.close(0)                           # lepaskan penulis yang tertahan di pipa

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
            if not self.wanted():
                if self._conn is not None:
                    try:
                        self._conn.close()
                    except Exception:  # noqa: BLE001
                        pass
                    self._conn = None
                    print(f"[stream] {self.url} berhenti: tidak ada penonton", flush=True)
                self.connected = False
                self._resync = True                 # mulai lagi di IDR saat diminta
                self.idle += 1
                continue
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
