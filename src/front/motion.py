"""Kecepatan radial per track dari riwayat jarak (Tech Spec bag. 6, Model 5).

- Timestamp = `Frame.ts_ms` (timestamp SUMBER), bukan jam dinding -> replay = live.
- Kecepatan = kemiringan regresi linear jarak terhadap waktu pada `speed_window_s` terakhir.
  Waktu dihitung relatif ke sampel terbaru (detik) agar fit tetap stabil walau ts_ms skala epoch.
- speed_mps < 0 = jarak mengecil. approaching = -speed_mps > approach_speed_mps.
- Jika metode jarak berganti (foot <-> height/clipped) riwayat direset, karena loncatan jarak
  antar-metode akan terbaca sebagai gerak cepat.
- Track yang tidak terlihat > track_stale_s dibuang.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass, field

import numpy as np


@dataclass
class MotionState:
    speed_mps: float | None
    approaching: bool


@dataclass
class _Track:
    samples: deque = field(default_factory=deque)   # (ts_ms, distance_m)
    method: str | None = None
    last_seen_ms: int = 0


class MotionTracker:
    def __init__(self, approach_speed_mps: float, speed_window_s: float, min_samples: int,
                 min_span_s: float, history_s: float, track_stale_s: float):
        self.approach_speed_mps = float(approach_speed_mps)
        self.window_ms = float(speed_window_s) * 1000.0
        self.min_samples = int(min_samples)
        self.min_span_ms = float(min_span_s) * 1000.0
        self.history_ms = max(float(history_s) * 1000.0, self.window_ms)
        self.stale_ms = float(track_stale_s) * 1000.0
        self.tracks: dict[int, _Track] = {}

    @classmethod
    def from_config(cls, thr_front: dict, cam_cfg: dict) -> "MotionTracker":
        m = cam_cfg.get("motion", {})
        return cls(thr_front["approach_speed_mps"], thr_front["speed_window_s"],
                   m["min_samples"], m["min_span_s"], m["history_s"], m["track_stale_s"])

    def update(self, track_id: int, ts_ms: int, distance_m: float | None,
               method: str | None = None) -> MotionState:
        tr = self.tracks.setdefault(track_id, _Track())
        tr.last_seen_ms = max(tr.last_seen_ms, int(ts_ms))
        if distance_m is not None:
            if method is not None and tr.method is not None and method != tr.method:
                tr.samples.clear()
            tr.method = method if method is not None else tr.method
            if tr.samples and ts_ms <= tr.samples[-1][0]:
                # timestamp duplikat / mundur: ganti sampel terakhir, jangan tambah
                tr.samples[-1] = (tr.samples[-1][0], float(distance_m))
            else:
                tr.samples.append((int(ts_ms), float(distance_m)))
            while tr.samples and ts_ms - tr.samples[0][0] > self.history_ms:
                tr.samples.popleft()
        return self.state(track_id)

    def speed(self, track_id: int) -> float | None:
        tr = self.tracks.get(track_id)
        if tr is None or not tr.samples:
            return None
        t_last = tr.samples[-1][0]
        win = [(t, d) for t, d in tr.samples if t_last - t <= self.window_ms]
        if len(win) < self.min_samples:
            return None
        t = np.array([(ti - t_last) / 1000.0 for ti, _ in win], dtype=np.float64)
        d = np.array([di for _, di in win], dtype=np.float64)
        if (t[-1] - t[0]) * 1000.0 < self.min_span_ms:
            return None
        tc = t - t.mean()
        denom = float(np.dot(tc, tc))
        if denom <= 0:
            return None
        return float(np.dot(tc, d - d.mean()) / denom)

    def state(self, track_id: int) -> MotionState:
        v = self.speed(track_id)
        return MotionState(v, v is not None and -v > self.approach_speed_mps)

    def prune(self, now_ms: int) -> list[int]:
        """Buang track yang tidak terlihat > track_stale_s. Mengembalikan id yang dibuang."""
        dead = [k for k, tr in self.tracks.items() if now_ms - tr.last_seen_ms > self.stale_ms]
        for k in dead:
            del self.tracks[k]
        return dead

    def reset(self) -> None:
        self.tracks.clear()
