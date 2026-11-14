"""Generator deret sinyal sintetis untuk menguji indikator dan keputusan tanpa kamera."""
from __future__ import annotations

from dataclasses import replace

from src.cabin.indicators import IndicatorInput, build_indicators
from src.cabin.signals import FaceSignals
from src.config import Config
from src.decision import DecisionEngine

FPS = 30.0


def sig(**kw) -> FaceSignals:
    base = dict(ts_ms=0, face_present=True, openness=1.0, blink=0.0, eye_closed=False,
                eye_valid=True, yaw=0.0, pitch=0.0, roll=0.0, calibrated=True)
    base.update(kw)
    return FaceSignals(**base)


class Timeline:
    """Bangun deret IndicatorInput per frame: tl.add(1.5, eye_closed=True).add(2, ...)."""

    def __init__(self, fps: float = FPS, t0_ms: int = 1_000_000):
        self.fps = fps
        self.frames: list[IndicatorInput] = []
        self.t_ms = float(t0_ms)
        self.t0_ms = t0_ms

    def add(self, dur_s: float, face: dict | None = None, **inp) -> "Timeline":
        face = face or {}
        n = int(round(dur_s * self.fps))
        for _ in range(n):
            ts = int(round(self.t_ms))
            s = sig(ts_ms=ts, **face)
            self.frames.append(IndicatorInput(ts_ms=ts, sig=s, **inp))
            self.t_ms += 1000.0 / self.fps
        return self

    def closed(self, dur_s: float, **face) -> "Timeline":
        return self.add(dur_s, dict(eye_closed=True, openness=0.1, blink=0.9, **face))

    def opened(self, dur_s: float, **face) -> "Timeline":
        return self.add(dur_s, dict(**face))

    def ramp(self, dur_s: float, key: str, a: float, b: float, **face) -> "Timeline":
        n = int(round(dur_s * self.fps))
        for i in range(n):
            v = a + (b - a) * (i + 1) / n
            self.add(1 / self.fps, {key: v, **face})
        return self


def thresholds(overrides: dict | None = None) -> dict:
    return Config.load(overrides={"thresholds": overrides or {}}).thresholds


def events_cfg(overrides: dict | None = None) -> dict:
    return Config.load(overrides={"events": overrides or {}}).events


def run_indicator(ind, tl: Timeline) -> list[bool]:
    return [ind.update(x) for x in tl.frames]


def run_system(tl: Timeline, th: dict | None = None, ev: dict | None = None):
    """Jalankan semua indikator + decision engine. Kembalikan (events, first_active_time_per_code)."""
    inds = build_indicators(th or thresholds())
    dec = DecisionEngine(ev or events_cfg())
    events = []
    first_active: dict[str, float] = {}
    for x in tl.frames:
        active = {}
        for code, ind in inds.items():
            if ind.update(x):
                active[code] = {"confidence": ind.confidence, "condition_start_ms": ind.condition_start_ms}
                first_active.setdefault(code, (x.ts_ms - tl.t0_ms) / 1000)
        for a in dec.step(x.ts_ms, active, {"t_video_s": (x.ts_ms - tl.t0_ms) / 1000}):
            if a.kind == "event":
                events.append(a.payload)
    if tl.frames:
        for a in dec.flush(tl.frames[-1].ts_ms):
            events.append(a.payload)
    return events, first_active


def starts(events):
    return [e for e in events if e["phase"] == "start"]
