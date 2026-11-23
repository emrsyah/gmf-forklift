"""Rekaman H.264 apa adanya dari kamera (tanpa encode ulang) ke segmen MP4 terfragmentasi.

Dipakai bersama `src.capture.uvch264`: setiap frame kamera membawa satu access unit H.264. Dibanding
MJPG .avi, ukurannya ~4-5x lebih kecil dan CPU Jetson nyaris tidak terpakai (ffmpeg hanya `-c copy`).

- Setiap segmen dimulai di keyframe IDR dan diawali SPS/PPS (disimpan dari stream, karena kamera tidak
  mengirim SPS di setiap IDR). Segmen diganti di IDR pertama setelah `segment_s`.
- H.264 mentah tidak punya timestamp. Sumber kebenaran waktu = `<video>.ts.csv` (frame,ts_ms), satu
  baris per access unit, format sama dengan rekaman lain sehingga replay harness memakai waktu asli.
  ffmpeg memberi timestamp jam dinding saat data masuk (`-use_wallclock_as_timestamps`) agar file
  diputar dengan kecepatan wajar di pemutar biasa walau FPS kamera naik-turun (gelap ~12-15 fps).
- Rotasi pemasangan kamera disimpan sebagai metadata `rotate` (piksel tidak diputar).
- MP4 terfragmentasi: file tetap bisa dibuka walau proses mati di tengah segmen.
- Antrean penuh -> frame dibuang dan rekaman menunggu IDR berikutnya (tanpa frame rusak).

Privasi (CLAUDE.md aturan 11): hanya untuk peraga yang setuju direkam; disimpan lokal, jangan diunggah.
"""
from __future__ import annotations

import queue
import shutil
import subprocess
import threading
import time
from pathlib import Path
from typing import Callable, Optional, Protocol

from src.capture.uvch264 import nal_types


class _Sink(Protocol):
    def write(self, data: bytes) -> None: ...
    def close(self) -> None: ...


_DISPLAY_ROTATION: dict[str, bool] = {}


def _has_display_rotation(ffmpeg: str) -> bool:
    """ffmpeg baru (laptop, 7+) mengabaikan tag `rotate`; harus `-display_rotation` (input, berlawanan
    jarum jam). ffmpeg 4.4 (Jetson) hanya mengenal tag `rotate` (searah jarum jam)."""
    if ffmpeg not in _DISPLAY_ROTATION:
        try:
            out = subprocess.run([ffmpeg, "-hide_banner", "-h", "long"], capture_output=True, text=True,
                                 timeout=10).stdout
        except (OSError, subprocess.TimeoutExpired):
            out = ""
        _DISPLAY_ROTATION[ffmpeg] = "-display_rotation" in out
    return _DISPLAY_ROTATION[ffmpeg]


def ffmpeg_cmd(ffmpeg: str, path: Path, rotate: int = 0, extra_args: list | None = None) -> list[str]:
    """`rotate` searah jarum jam (sama seperti front_preview.rotate).

    Diuji 2026-11-23: di ffmpeg 4.4 (Jetson) `-metadata rotate=N` dan di ffmpeg baru `-display_rotation N`
    sama-sama berarti N derajat BERLAWANAN jarum jam saat diputar (ffmpeg autorotate dan OpenCV setuju),
    jadi keduanya diberi 360 - rotate."""
    rot = int(rotate) % 360
    ccw = (360 - rot) % 360
    # probesize kecil + analyzeduration 0: SPS sudah cukup; default ffmpeg menahan ~5 s input sebelum menulis
    cmd = [ffmpeg, "-hide_banner", "-loglevel", "error", "-y", "-use_wallclock_as_timestamps", "1",
           "-probesize", "32768", "-analyzeduration", "0"]
    new_style = rot and _has_display_rotation(ffmpeg)
    if new_style:
        cmd += ["-display_rotation:v:0", str(ccw)]
    cmd += ["-f", "h264", "-i", "pipe:0", "-c", "copy", "-movflags", "+frag_keyframe+empty_moov+default_base_moof"]
    if rot and not new_style:
        cmd += ["-metadata:s:v:0", f"rotate={ccw}"]
    return cmd + list(extra_args or []) + ["-f", "mp4", str(path)]


class FfmpegMp4Sink:
    """Pipa Annex B -> ffmpeg -c copy -> MP4 terfragmentasi."""

    def __init__(self, ffmpeg: str, path: Path, rotate: int = 0, extra_args: list | None = None):
        cmd = ffmpeg_cmd(ffmpeg, path, rotate, extra_args)
        self.proc = subprocess.Popen(cmd, stdin=subprocess.PIPE, stdout=subprocess.DEVNULL,
                                     stderr=subprocess.PIPE)

    def write(self, data: bytes) -> None:
        self.proc.stdin.write(data)
        self.proc.stdin.flush()

    def close(self, timeout_s: float = 10.0) -> None:
        try:
            self.proc.stdin.close()
        except OSError:
            pass
        try:
            self.proc.wait(timeout_s)
        except subprocess.TimeoutExpired:
            self.proc.kill()
            self.proc.wait()
        err = self.proc.stderr.read().decode(errors="replace").strip() if self.proc.stderr else ""
        if self.proc.returncode not in (0, None) or err:
            print(f"[rekam-h264] ffmpeg selesai kode {self.proc.returncode}: {err[-300:]}", flush=True)


class H264Recorder:
    records = True      # front_preview: selama perekam ini hidup, perekam MJPG cadangan tidak dipakai

    def __init__(self, cfg: dict, device_id: str = "dev", rotate: int = 0,
                 sink_factory: Optional[Callable[[Path], _Sink]] = None,
                 on_segment: Optional[Callable[[], None]] = None):
        self.dir = Path(cfg.get("dir", "recordings"))
        self.segment_s = float(cfg.get("segment_s", 300))
        self.ext = str(cfg.get("h264_ext", "mp4"))
        self.min_free_gb = float(cfg.get("min_free_gb", 10))
        self.prefix = f"{cfg.get('prefix', 'front')}_{device_id}"
        self.rotate = int(rotate)
        self.on_segment = on_segment            # mis. minta IDR ke kamera supaya ganti segmen cepat
        if sink_factory is None:
            ff = shutil.which(cfg.get("ffmpeg", "ffmpeg"))
            if ff is None:
                raise RuntimeError("ffmpeg tidak ditemukan; rekaman H.264 butuh ffmpeg")
            args = cfg.get("h264_ffmpeg_args") or []
            sink_factory = lambda p: FfmpegMp4Sink(ff, p, self.rotate, args)  # noqa: E731
        self._sink_factory = sink_factory
        self._q: queue.Queue = queue.Queue(maxsize=int(cfg.get("max_queue", 60)))
        self._resync = True                     # mulai di IDR pertama
        self.dropped = 0
        self.frames = 0
        self.segments: list[Path] = []
        self.stopped_reason: str | None = None
        self._sink: _Sink | None = None
        self._ts_file = None
        self._seg_start_ms = 0
        self._seg_frames = 0
        self._sps: bytes | None = None
        self._pps: bytes | None = None
        self._rotate_asked = False
        self._thread = threading.Thread(target=self._run, name="h264-rec", daemon=True)
        self._thread.start()

    # ---------------------------------------------------------------- thread kamera
    def push(self, au: bytes | None, ts_ms: int, idr: bool | None = None) -> None:
        """Dipanggil untuk SETIAP frame (H.264 tidak boleh dicuplik seperti MJPG)."""
        if not au or self.stopped_reason is not None:
            return
        if self._resync:
            if not (idr if idr is not None else 5 in nal_types(au)):
                self.dropped += 1
                return
            self._resync = False
        try:
            self._q.put_nowait((au, int(ts_ms)))
        except queue.Full:
            self.dropped += 1
            self._resync = True                 # rantai referensi putus: tunggu IDR berikutnya

    def close(self, timeout_s: float = 10.0) -> None:
        try:
            self._q.put(None, timeout=timeout_s)    # penulis macet + antrean penuh: jangan menggantung
        except queue.Full:
            pass
        self._thread.join(timeout_s)

    # ---------------------------------------------------------------- thread penulis
    def free_gb(self) -> float:
        self.dir.mkdir(parents=True, exist_ok=True)
        return shutil.disk_usage(self.dir).free / 1e9

    def _remember_params(self, au: bytes) -> None:
        from src.capture.uvch264 import nal_units
        for t, s, e in nal_units(au):
            if t == 7:
                self._sps = au[s:e]
            elif t == 8:
                self._pps = au[s:e]

    def _open(self, ts_ms: int) -> bool:
        free = self.free_gb()
        if free < self.min_free_gb:
            self.stopped_reason = f"ruang disk {free:.1f} GB < {self.min_free_gb} GB"
            print(f"[rekam-h264] BERHENTI: {self.stopped_reason}", flush=True)
            return False
        day = time.strftime("%Y-%m-%d", time.localtime(ts_ms / 1000))
        stamp = time.strftime("%Y%m%d_%H%M%S", time.localtime(ts_ms / 1000))
        out = self.dir / day
        out.mkdir(parents=True, exist_ok=True)
        path = out / f"{self.prefix}_{stamp}.{self.ext}"
        try:
            self._sink = self._sink_factory(path)
        except OSError as e:
            self.stopped_reason = f"tidak bisa memulai penulis: {e}"
            print(f"[rekam-h264] BERHENTI: {self.stopped_reason}", flush=True)
            return False
        self._ts_file = open(str(path) + ".ts.csv", "w", encoding="utf-8", buffering=1)
        self._ts_file.write("frame,ts_ms\n")
        self._seg_start_ms = ts_ms
        self._seg_frames = 0
        self._rotate_asked = False
        self.segments.append(path)
        print(f"[rekam-h264] segmen baru {path} ({free:.0f} GB bebas)", flush=True)
        return True

    def _close_segment(self) -> None:
        if self._sink is not None:
            self._sink.close()
            self._sink = None
        if self._ts_file is not None:
            self._ts_file.close()
            self._ts_file = None

    def _run(self) -> None:
        while True:
            item = self._q.get()
            if item is None:
                break
            au, ts_ms = item
            types = nal_types(au)
            self._remember_params(au)
            idr = 5 in types
            due = self._sink is not None and ts_ms - self._seg_start_ms >= self.segment_s * 1000
            if due and not self._rotate_asked and self.on_segment is not None:
                self._rotate_asked = True
                try:
                    self.on_segment()
                except Exception:  # noqa: BLE001 - permintaan IDR hanya optimasi
                    pass
            if due and idr:
                self._close_segment()
            if self._sink is None:
                if not idr or self._sps is None or self._pps is None:
                    continue                    # tunggu keyframe yang bisa didekode (SPS/PPS sudah dikenal)
                if not self._open(ts_ms):
                    break
            prefix = b""
            if idr and 7 not in types and self._seg_frames == 0 and self._sps and self._pps:
                prefix = self._sps + self._pps  # segmen baru harus bisa didekode sendiri
            try:
                self._sink.write(prefix + au)
                self._ts_file.write(f"{self._seg_frames},{ts_ms}\n")
                self._seg_frames += 1
                self.frames += 1
            except (OSError, ValueError) as e:
                self.stopped_reason = f"tulis gagal: {e}"
                print(f"[rekam-h264] BERHENTI: {self.stopped_reason}", flush=True)
                break
        self._close_segment()
