"""Modul keputusan: prioritas kode, cooldown A-02, buzzer berulang A-03, hysteresis, event start/end.

Aturan (Tech Spec bag. 4, configs/events.yaml):
1. Kumpulkan indikator aktif, pilih SATU kode dengan prioritas tertinggi.
2. Jika belum ada event terbuka untuk kode itu (dan lewat cooldown untuk A-02) -> event `start`.
3. A-03: buzzer berulang selama kondisi aktif. Setelah kondisi hilang `end_hysteresis_s` -> event `end`.
4. Satu event terbuka pada satu waktu. Kode berprioritas LEBIH TINGGI memotong event terbuka
   (end lalu start). Kode berprioritas lebih rendah tidak membuka event selama event terbuka
   masih dalam masa hysteresis -> mencegah kode salah sesaat (risiko K4).
5. Tidak ada eskalasi otomatis antar kode.
"""
from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta


@dataclass
class OpenEvent:
    event_id: str
    code: str
    level: str
    start_ms: int
    condition_start_ms: int
    last_active_ms: int
    confidence: float
    last_buzz_ms: int


@dataclass
class Action:
    kind: str              # "event" | "buzz"
    payload: dict = field(default_factory=dict)


class DecisionEngine:
    def __init__(self, events_cfg: dict, device_id: str = "laptop-dev", id_prefix: str = "laptop",
                 model_versions: dict | None = None, tz_offset_h: float = 7.0):
        self.c = events_cfg
        self.priority: list[str] = list(events_cfg["priority"])
        self.levels: dict[str, str] = dict(events_cfg["levels"])
        self.device_id = device_id
        self.id_prefix = id_prefix
        self.model_versions = model_versions or {}
        self.tz = timezone(timedelta(hours=tz_offset_h))
        self.open: OpenEvent | None = None
        self.last_start_ms: dict[str, int] = {}
        self._counter = 0
        self.history: list[dict] = []

    # ------------------------------------------------------------------ util
    def rank(self, code: str) -> int:
        return self.priority.index(code) if code in self.priority else len(self.priority)

    def _new_id(self, ts_ms: int) -> str:
        self._counter += 1
        # tanggal + jam (agar unik antar-run) + nomor urut dalam run
        d = datetime.fromtimestamp(ts_ms / 1000, self.tz).strftime("%Y%m%d-%H%M%S")
        return f"{self.id_prefix}-{d}-{self._counter:04d}"

    def _event(self, phase: str, ev: OpenEvent, ts_ms: int, extra: dict | None = None) -> dict:
        e = {
            "event_id": ev.event_id,
            "device_id": self.device_id,
            "phase": phase,
            "code": ev.code,
            "level": ev.level,
            "ts_event_ms": int(ts_ms),
            "ts_condition_start_ms": int(ev.condition_start_ms),
            "confidence": round(float(ev.confidence), 3),
            "zone_id": None,
            "snapshot_path": None,
            "clip_path": None,
            "model_versions": dict(self.model_versions),
        }
        if extra:
            e.update(extra)
        self.history.append(e)
        return e

    def _cooldown_ok(self, code: str, ts_ms: int) -> bool:
        if self.levels.get(code) != "A-02":
            return True
        last = self.last_start_ms.get(code)
        return last is None or ts_ms - last >= self.c.get("A02_cooldown_s", 20) * 1000

    # ------------------------------------------------------------------ main
    def step(self, ts_ms: int, active: dict[str, dict], extra: dict | None = None) -> list[Action]:
        """`active`: {code: {"confidence": float, "condition_start_ms": int}} untuk indikator aktif."""
        acts: list[Action] = []
        top = min(active, key=self.rank) if active else None
        hyst_ms = self.c.get("end_hysteresis_s", 1.5) * 1000

        ev = self.open
        if ev is not None:
            if ev.code in active:
                ev.last_active_ms = ts_ms
                ev.confidence = max(ev.confidence, active[ev.code].get("confidence", 0.0))
            preempt = top is not None and top != ev.code and self.rank(top) < self.rank(ev.code)
            expired = ts_ms - ev.last_active_ms >= hyst_ms
            if preempt or expired:
                acts.append(Action("event", self._event("end", ev, ts_ms, extra)))
                self.open = None
                ev = None

        if self.open is None and top is not None and self._cooldown_ok(top, ts_ms):
            info = active[top]
            level = self.levels.get(top, "A-02")
            ev = OpenEvent(self._new_id(ts_ms), top, level, ts_ms,
                           int(info.get("condition_start_ms") or ts_ms), ts_ms,
                           float(info.get("confidence", 0.0)), ts_ms)
            self.open = ev
            self.last_start_ms[top] = ts_ms
            acts.append(Action("event", self._event("start", ev, ts_ms, extra)))
            acts.append(Action("buzz", {"level": level, "code": top, "first": True}))
        elif self.open is not None and self.open.level == "A-03" and self.open.code in active:
            # buzzer berulang selama kondisi A-03 aktif
            if ts_ms - self.open.last_buzz_ms >= self.c.get("A03_repeat_buzzer_s", 1.0) * 1000:
                self.open.last_buzz_ms = ts_ms
                acts.append(Action("buzz", {"level": "A-03", "code": self.open.code, "first": False}))
        return acts

    def flush(self, ts_ms: int, extra: dict | None = None) -> list[Action]:
        """Tutup event terbuka (akhir video / shutdown)."""
        if self.open is None:
            return []
        e = self._event("end", self.open, ts_ms, extra)
        self.open = None
        return [Action("event", e)]
