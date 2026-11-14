"""Kalibrasi per pengemudi (Tech Spec bag. 3): menatap depan ~5 s -> ear_baseline + pose nol.

Sebelum kalibrasi selesai dipakai baseline populasi agar sistem tetap jalan.
Kalibrasi ditolak (dan diulang) jika variansi terlalu tinggi.
"""
from __future__ import annotations

import numpy as np


class Calibration:
    def __init__(self, cfg: dict, auto_start: bool = True):
        self.c = cfg
        self.duration_ms = float(cfg.get("duration_s", 5.0)) * 1000
        self.min_frames = int(cfg.get("min_frames", 30))
        self.pop_ear = float(cfg.get("population_ear_baseline", 0.28))
        self.ear_baseline = self.pop_ear
        self.yaw0 = self.pitch0 = self.roll0 = 0.0
        self.done = False
        self.active = auto_start
        self.status = "menunggu wajah" if auto_start else "baseline populasi"
        self.attempts = 0
        self._t0: int | None = None
        self._buf: list[dict] = []

    def set_population(self):
        self.active = False
        self.done = False
        self.ear_baseline = self.pop_ear
        self.yaw0 = self.pitch0 = self.roll0 = 0.0
        self.status = "baseline populasi"

    def observe(self, ts_ms: int, raw: dict) -> None:
        if not self.active:
            return
        if self._t0 is None:
            self._t0 = ts_ms
            self.status = "kalibrasi: tatap lurus ke depan"
        self._buf.append(raw)
        if ts_ms - self._t0 < self.duration_ms:
            return
        self.attempts += 1
        self._finish()

    def _maybe_give_up(self):
        """Setelah `max_attempts` gagal, berhenti mencoba dan pakai baseline populasi
        (supaya kalibrasi tidak diam-diam berhasil di tengah skenario). Hotkey 'c' memulai ulang."""
        if self.attempts >= int(self.c.get("max_attempts", 3)):
            self.set_population()
            self.status = f"baseline populasi (kalibrasi gagal {self.attempts}x; tekan c untuk ulang)"

    def start(self):
        """Mulai (ulang) kalibrasi, misalnya lewat hotkey 'c'."""
        self.active = True
        self.attempts = 0
        self._t0 = None
        self._buf = []
        self.status = "menunggu wajah"

    def _finish(self):
        buf = self._buf
        self._t0 = None
        self._buf = []
        if len(buf) < self.min_frames:
            self.status = f"kalibrasi gagal: frame kurang ({len(buf)}), ulangi"
            self._maybe_give_up()
            return
        ear = np.array([(b["ear_l"] + b["ear_r"]) / 2 for b in buf])
        yaw = np.array([b["yaw"] for b in buf])
        pitch = np.array([b["pitch"] for b in buf])
        roll = np.array([b["roll"] for b in buf])
        if ear.std() > self.c.get("max_ear_std", 0.04):
            self.status = f"kalibrasi ditolak: mata tidak stabil (std {ear.std():.3f}), ulangi"
            self._maybe_give_up()
            return
        max_pose = self.c.get("max_pose_std_deg", 5.0)
        if max(yaw.std(), pitch.std(), roll.std()) > max_pose:
            self.status = "kalibrasi ditolak: kepala bergerak, ulangi"
            self._maybe_give_up()
            return
        y0, p0, r0 = float(np.median(yaw)), float(np.median(pitch)), float(np.median(roll))
        # Pose "lurus" harus masuk akal; tolak jika pengemudi sedang menoleh/miring/menunduk
        # (mis. kalibrasi ulang tidak sengaja saat skenario 19 kepala miring bertahan).
        if (abs(r0) > self.c.get("max_abs_roll_deg", 15) or abs(y0) > self.c.get("max_abs_yaw_deg", 25)
                or abs(p0) > self.c.get("max_abs_pitch_deg", 35)):
            self.status = f"kalibrasi ditolak: pose tidak lurus (yaw {y0:+.0f} pitch {p0:+.0f} roll {r0:+.0f})"
            self._maybe_give_up()
            return
        self.ear_baseline = float(np.median(ear))
        self.yaw0, self.pitch0, self.roll0 = y0, p0, r0
        self.done = True
        self.active = False
        self.status = f"terkalibrasi (EAR {self.ear_baseline:.3f})"
