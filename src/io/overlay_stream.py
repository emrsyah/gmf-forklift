"""Video beranotasi (overlay kabin, kotak deteksi depan) di-encode H.264 dan didorong ke RTSP (MediaMTX).

Kenapa: dashboard MINERVA menampilkan JPEG per frame lewat HTTP (agen edge) -> patah-patah (~7 fps, tiap
lonjakan RTT 5G = gambar macet). Dengan stream ini dashboard memutar WebRTC dari MediaMTX server MINERVA
(`http://<server>:8889/<path>`), mulus dan tetap memperlihatkan hasil AI.

- Frame yang sama dengan tampilan web (`WebView.publish`) -> `VideoFanout` meneruskan ke keduanya.
- Satu ffmpeg per kanal: rawvideo BGR dari pipa -> libx264 ultrafast/zerolatency -> RTSP TCP.
  (Orin Nano tidak punya encoder H.264 hardware; uji 2026-11-24: 640x360@15 ~9% satu core.)
- Ukuran output tetap per koneksi (dari frame pertama, sisi terpanjang `max_side`, genap).
- Slot terbaru saja: encoder lambat / jaringan macet tidak pernah menahan loop deteksi.
- Server mati -> coba lagi tiap `retry_s`; frame selama itu tidak digambar (`wants` False).
"""
from __future__ import annotations

import shutil
import subprocess
import threading
import time
from typing import Callable, Optional

import cv2
import numpy as np


def even_size(w: int, h: int, max_side: int | None) -> tuple[int, int]:
    """Ukuran output: diperkecil agar sisi terpanjang <= max_side, dibulatkan genap (syarat yuv420p)."""
    s = 1.0
    if max_side and max(w, h) > max_side:
        s = max_side / max(w, h)
    return max(2, int(w * s + 1e-6) // 2 * 2), max(2, int(h * s + 1e-6) // 2 * 2)


def ffmpeg_cmd(ffmpeg: str, url: str, w: int, h: int, cfg: dict) -> list[str]:
    fps = float(cfg.get("fps", 15))
    gop = max(1, int(round(fps * float(cfg.get("gop_s", 1)))))
    return [ffmpeg, "-hide_banner", "-loglevel", "error",
            "-use_wallclock_as_timestamps", "1",
            "-f", "rawvideo", "-pix_fmt", "bgr24", "-s", f"{w}x{h}", "-i", "pipe:0",
            "-c:v", "libx264", "-preset", str(cfg.get("preset", "ultrafast")), "-tune", "zerolatency",
            # baseline: semua browser bisa memutar via WebRTC tanpa transcode
            "-profile:v", str(cfg.get("profile", "baseline")), "-pix_fmt", "yuv420p",
            "-g", str(gop), "-keyint_min", str(gop), "-sc_threshold", "0",
            "-b:v", str(cfg.get("bitrate", "1000k")), "-maxrate", str(cfg.get("bitrate", "1000k")),
            "-bufsize", str(cfg.get("bufsize", "500k")),
            *[str(x) for x in (cfg.get("ffmpeg_args") or [])],
            "-f", "rtsp", "-rtsp_transport", str(cfg.get("transport", "tcp")), url]


class _FfmpegRaw:
    def __init__(self, cmd: list[str]):
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                                     stderr=subprocess.PIPE)
        self._err = b""
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

    def kill(self) -> None:
        try:
            self.proc.kill()
        except OSError:
            pass

    def close(self, timeout_s: float = 3.0) -> None:
        try:
            self.proc.stdin.close()
        except OSError:
            pass
        try:
            self.proc.wait(timeout_s)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait()


class _Channel:
    def __init__(self, name: str, url: str, cfg: dict, connect: Callable[[str, int, int], object]):
        self.name = name
        self.url = url
        self.period_s = 1.0 / max(float(cfg.get("fps", 15)), 0.1)
        self.max_side = cfg.get("max_side", 640)
        self.retry_s = float(cfg.get("retry_s", 5))
        # Server di-restart di balik NAT 5G: RST tidak sampai, ffmpeg tidak tahu, pipa penuh -> write() macet
        # selamanya. Tulisan yang tertahan > stall_s -> ffmpeg dibunuh, tulisan gagal, sambung ulang.
        self.stall_s = float(cfg.get("stall_s", 5))
        self._write_t0: float | None = None
        self.stalls = 0
        self._connect = connect
        self._conn = None
        self._size: tuple[int, int] | None = None
        self._slot: np.ndarray | None = None
        self._cv = threading.Condition()
        self._stop = False
        self.last_pub = 0.0
        self.next_try = 0.0
        self.sent = 0
        self.connects = 0
        self.connected = False
        self._thread = threading.Thread(target=self._run, name=f"overlay-{name}", daemon=True)
        self._thread.start()

    def wants(self) -> bool:
        now = time.monotonic()
        t0 = self._write_t0
        if t0 is not None and now - t0 > self.stall_s:
            self._kill_stalled()
            return False
        return now >= self.next_try and now - self.last_pub >= self.period_s and t0 is None

    def _kill_stalled(self) -> None:
        conn = self._conn
        if conn is None or self._write_t0 is None:
            return
        self._write_t0 = None              # sekali per macet; thread penulis menangani putusnya
        self.stalls += 1
        self.next_try = time.monotonic() + self.retry_s
        print(f"[overlay] {self.url} macet > {self.stall_s:.0f} s; ffmpeg dihentikan", flush=True)
        kill = getattr(conn, "kill", None)
        if kill is not None:
            kill()

    def publish(self, img: np.ndarray) -> None:
        self.last_pub = time.monotonic()
        with self._cv:
            self._slot = img                # frame lama yang belum terkirim ditimpa
            self._cv.notify()

    def _drop(self, why: str) -> None:
        if self._conn is not None:
            err = getattr(self._conn, "error_text", lambda: "")()
            try:
                self._conn.close()
            except Exception:  # noqa: BLE001
                pass
            self._conn = None
            print(f"[overlay] {self.url} terputus ({why}{': ' + err if err else ''}); coba lagi tiap "
                  f"{self.retry_s:.0f} s", flush=True)
        self.connected = False
        self.next_try = time.monotonic() + self.retry_s

    def _run(self) -> None:
        while True:
            with self._cv:
                while self._slot is None and not self._stop:
                    self._cv.wait(0.5)
                if self._stop:
                    break
                img, self._slot = self._slot, None
            if self._conn is not None and not getattr(self._conn, "alive", lambda: True)():
                self._drop("ffmpeg berhenti")
            if self._conn is None:
                if time.monotonic() < self.next_try:
                    continue
                h, w = img.shape[:2]
                self._size = even_size(w, h, self.max_side)
                try:
                    self._conn = self._connect(self.url, *self._size)
                except OSError as e:
                    self.next_try = time.monotonic() + self.retry_s
                    print(f"[overlay] tidak bisa memulai ffmpeg: {e}", flush=True)
                    continue
                self.connects += 1
                print(f"[overlay] {self.name} -> {self.url} ({self._size[0]}x{self._size[1]})", flush=True)
            w, h = self._size
            if img.shape[1] != w or img.shape[0] != h:
                img = cv2.resize(img, (w, h), interpolation=cv2.INTER_AREA)
            data = np.ascontiguousarray(img).tobytes()
            self._write_t0 = time.monotonic()
            try:
                self._conn.write(data)
            except (OSError, ValueError) as e:
                self._write_t0 = None
                self._drop(str(e))
                continue
            self._write_t0 = None
            self.connected = True
            self.sent += 1
        if self._conn is not None:
            self._conn.close()
            self._conn = None

    def close(self, timeout_s: float = 3.0) -> None:
        with self._cv:
            self._stop = True
            self._cv.notify()
        self._thread.join(timeout_s)
        conn = self._conn
        if self._thread.is_alive() and conn is not None:
            conn.close(0)


class OverlayStreams:
    """`wants(channel)` / `publish(img, channel)` seperti WebView, tapi ke RTSP H.264 per kanal."""

    def __init__(self, cfg: dict, connect: Optional[Callable[[str, int, int], object]] = None):
        base = str(cfg.get("base_url", "rtsp://127.0.0.1:8554")).rstrip("/")
        paths = cfg.get("channels") or {}
        if connect is None:
            ff = shutil.which(cfg.get("ffmpeg", "ffmpeg"))
            if ff is None:
                raise RuntimeError("ffmpeg tidak ditemukan; overlay stream butuh ffmpeg")
            connect = lambda url, w, h: _FfmpegRaw(ffmpeg_cmd(ff, url, w, h, cfg))  # noqa: E731
        self.channels = {ch: _Channel(ch, f"{base}/{str(p).strip('/')}", cfg, connect)
                         for ch, p in paths.items() if p}

    def wants(self, channel: str = "cabin") -> bool:
        c = self.channels.get(channel)
        return c is not None and c.wants()

    def publish(self, img: np.ndarray | None, channel: str = "cabin") -> None:
        c = self.channels.get(channel)
        if img is not None and c is not None and c.wants():
            c.publish(img)

    def close(self) -> None:
        for c in self.channels.values():
            c.close()


class VideoFanout:
    """Satu titik publish untuk banyak keluaran video (WebView, OverlayStreams). None diabaikan."""

    def __init__(self, *outs):
        self.outs = [o for o in outs if o is not None]

    def __bool__(self) -> bool:
        return bool(self.outs)

    def wants(self, channel: str = "cabin") -> bool:
        return any(o.wants(channel) for o in self.outs)

    def publish(self, img: np.ndarray | None, channel: str = "cabin") -> None:
        for o in self.outs:
            o.publish(img, channel)

    def add_channel(self, name: str, label: str) -> None:
        for o in self.outs:
            if hasattr(o, "add_channel"):
                o.add_channel(name, label)
