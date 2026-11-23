"""Kalibrasi per pengemudi (Tech Spec bag. 3): menatap depan ~5 s -> ear_baseline + pose nol.

Sebelum kalibrasi selesai dipakai baseline populasi agar sistem tetap jalan.
Kalibrasi ditolak (dan diulang) jika variansi terlalu tinggi.
"""
from __future__ import annotations

from collections import deque

import numpy as np


class Calibration:
    def __init__(self, cfg: dict, auto_start: bool = True):
        self.c = cfg
        self.duration_ms = float(cfg.get("duration_s", 5.0)) * 1000
        self.min_frames = int(cfg.get("min_frames", 30))
        self.pop_ear = float(cfg.get("population_ear_baseline", 0.28))
        self.ear_baseline = self.pop_ear
        # sudut pasang kamera (mis. di pilar depan, wajah terlihat menyamping ~20-30 deg saat menatap depan):
        # pose "lurus" yang diharapkan = sudut ini, dan baseline populasi memakai sudut ini sebagai nol.
        self.mount = (float(cfg.get("mount_yaw_deg", 0.0)), float(cfg.get("mount_pitch_deg", 0.0)),
                      float(cfg.get("mount_roll_deg", 0.0)))
        self.yaw0, self.pitch0, self.roll0 = self.mount
        self.done = False
        self.active = auto_start
        self.status = "menunggu wajah" if auto_start else "baseline populasi"
        self.attempts = 0
        self._t0: int | None = None
        self._buf: list[dict] = []
        # baseline adaptif selama kalibrasi belum berhasil: persentil tinggi EAR wajah frontal (pejam = ekor bawah)
        ad = cfg.get("adaptive_baseline") or {}
        self.adaptive = bool(ad.get("enabled", False))
        self._ad = ad
        self._ad_buf: deque = deque()          # (ts_ms, ear)
        self.adaptive_used = False

    def set_population(self):
        self.active = False
        self.done = False
        if not self.adaptive_used:            # baseline adaptif yang sudah terbentuk tetap dipakai
            self.ear_baseline = self.pop_ear
        self.yaw0, self.pitch0, self.roll0 = self.mount
        self.status = "baseline populasi"

    def _adapt(self, ts_ms: int, raw: dict) -> None:
        """Tanpa kalibrasi yang berhasil, baseline populasi bisa jauh dari mata pengemudi (mata sipit / NIR:
        sim_P1b terbuka = openness 0.55 -> mata setengah terbuka dianggap terpejam 11 s). Perkirakan EAR mata
        terbuka dari persentil tinggi EAR saat wajah frontal; pejam (D-01/D-02) hanya mengisi ekor bawah."""
        a = self._ad
        my, mp, _ = self.mount
        if (abs(raw["yaw"] - my) > a.get("max_abs_yaw_deg", 20)
                or abs(raw["pitch"] - mp) > a.get("max_abs_pitch_deg", 20)):
            return
        self._ad_buf.append((ts_ms, (raw["ear_l"] + raw["ear_r"]) / 2))
        win = float(a.get("window_s", 60)) * 1000
        while self._ad_buf and self._ad_buf[0][0] < ts_ms - win:
            self._ad_buf.popleft()
        if self._ad_buf[-1][0] - self._ad_buf[0][0] < float(a.get("min_s", 5)) * 1000:
            return
        ear = float(np.percentile([e for _, e in self._ad_buf], a.get("percentile", 85)))
        floor = self.pop_ear * float(self.c.get("min_ear_ratio_population", 0.5))
        self.ear_baseline = max(ear, floor)
        self.adaptive_used = True

    def observe(self, ts_ms: int, raw: dict) -> None:
        if self.adaptive and not self.done:
            self._adapt(ts_ms, raw)
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
        # Mata terpejam yang diam juga "stabil": tanpa cek ini baseline = EAR mata tertutup (mis. percobaan ulang
        # jatuh tepat di skenario 5) -> openness ~4x, D-01/D-02 buta sampai kalibrasi berikutnya.
        blink = np.array([b.get("blink", 0.0) for b in buf])
        min_ratio = float(self.c.get("min_ear_ratio_population", 0.6))
        if blink.mean() > self.c.get("max_blink_mean", 0.5) or np.median(ear) < self.pop_ear * min_ratio:
            self.status = (f"kalibrasi ditolak: mata tampak tertutup (EAR {np.median(ear):.3f}, "
                           f"blink {blink.mean():.2f}), ulangi")
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
        my, mp, mr = self.mount
        if (abs(r0 - mr) > self.c.get("max_abs_roll_deg", 15) or abs(y0 - my) > self.c.get("max_abs_yaw_deg", 25)
                or abs(p0 - mp) > self.c.get("max_abs_pitch_deg", 35)):
            self.status = f"kalibrasi ditolak: pose tidak lurus (yaw {y0:+.0f} pitch {p0:+.0f} roll {r0:+.0f})"
            self._maybe_give_up()
            return
        self.ear_baseline = float(np.median(ear))
        self.yaw0, self.pitch0, self.roll0 = y0, p0, r0
        self.done = True
        self.active = False
        self.status = f"terkalibrasi (EAR {self.ear_baseline:.3f})"
