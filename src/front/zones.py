"""Indikator KANDIDAT kamera depan (USULAN, belum disepakati HSE GMF — pertanyaan terbuka #4).

    F01: person di zona peringatan (<= warning_zone_m) DAN mendekat   -> level dari events.yaml (A-02)
    F02: person di zona bahaya (<= danger_zone_m)                     -> A-03
    F03: objek pesawat/GSE (danger_object_classes) di zona bahaya     -> A-03

Modul ini hanya menghasilkan kandidat. Modul keputusan yang memutuskan apakah dikirim
(events.yaml: front_events_enabled: false secara default).
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Iterable, Protocol


class _TrackLike(Protocol):
    track_id: int | None
    cls: str
    conf: float
    distance_m: float | None
    approaching: bool


@dataclass
class FrontCandidate:
    code: str              # "F01" | "F02" | "F03" (kunci events.yaml)
    level: str | None      # dari events.yaml levels
    track_id: int | None
    cls: str
    distance_m: float
    confidence: float


class ZoneEvaluator:
    def __init__(self, warning_zone_m: float, danger_zone_m: float, person_class: str,
                 danger_object_classes: Iterable[str], levels: dict[str, str] | None,
                 priority: list[str]):
        self.warning_zone_m = float(warning_zone_m)
        self.danger_zone_m = float(danger_zone_m)
        self.person_class = person_class
        self.danger_object_classes = set(danger_object_classes)
        self.levels = dict(levels or {})
        self.priority = list(priority)

    @classmethod
    def from_config(cls, thr_front: dict, cam_cfg: dict, events: dict, person_class: str) -> "ZoneEvaluator":
        return cls(thr_front["warning_zone_m"], thr_front["danger_zone_m"], person_class,
                   cam_cfg.get("classes", {}).get("danger_object_classes", []),
                   events.get("levels", {}), cam_cfg.get("zones", {}).get("priority", []))

    def _rank(self, code: str) -> int:
        return self.priority.index(code) if code in self.priority else len(self.priority)

    def evaluate(self, tracks: Iterable[_TrackLike]) -> list[FrontCandidate]:
        """Satu kandidat per track (kode tertinggi untuk track itu), diurutkan prioritas lalu jarak."""
        out: list[FrontCandidate] = []
        for t in tracks:
            d = t.distance_m
            if d is None:
                continue
            codes = []
            if t.cls == self.person_class:
                if d <= self.danger_zone_m:
                    codes.append("F02")
                if d <= self.warning_zone_m and t.approaching:
                    codes.append("F01")
            elif t.cls in self.danger_object_classes and d <= self.danger_zone_m:
                codes.append("F03")
            if not codes:
                continue
            code = min(codes, key=self._rank)
            out.append(FrontCandidate(code, self.levels.get(code), t.track_id, t.cls, d, float(t.conf)))
        out.sort(key=lambda c: (self._rank(c.code), c.distance_m))
        return out

    def active_codes(self, candidates: list[FrontCandidate]) -> list[str]:
        seen: list[str] = []
        for c in candidates:
            if c.code not in seen:
                seen.append(c.code)
        return seen
