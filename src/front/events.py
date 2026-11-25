"""Event kamera depan F-01..F-03 (USULAN, belum disepakati HSE GMF -- pertanyaan terbuka #4).

Kandidat dari ZoneEvaluator (FrontResult.active_codes) harus bertahan `event_min_active_s` sebelum dianggap
aktif (kotak YOLO berkedip satu frame tidak memicu alert). Lifecycle start/end, cooldown A-02 dan hysteresis
memakai DecisionEngine yang sama dengan kabin, tetapi instance TERPISAH: event depan tidak memotong event
kabin (D-xx) dan sebaliknya; prioritas antar kode F dari front_camera.yaml `zones.priority`.

Snapshot = gambar kamera depan beranotasi (kotak + jarak), ditulis di folder event yang sama dengan kabin,
sehingga agen edge mengunggahnya seperti snapshot D-xx.
"""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import cv2

from src.decision import Action, DecisionEngine


class FrontEventGate:
    def __init__(self, events_cfg: dict, priority: list[str], min_active_s: float, device_id: str,
                 id_prefix: str, model_versions: dict | None = None):
        self.min_active_ms = float(min_active_s) * 1000
        cfg = dict(events_cfg, priority=list(priority))
        self.decision = DecisionEngine(cfg, device_id, id_prefix, model_versions)
        self._since: dict[str, int] = {}            # kode kandidat -> ts pertama terlihat (beruntun)

    @classmethod
    def from_config(cls, cfg, cam_cfg: dict) -> "FrontEventGate":
        rt = cfg.runtime
        return cls(cfg.events, cam_cfg.get("zones", {}).get("priority", ["F02", "F03", "F01"]),
                   cfg.thresholds.get("front", {}).get("event_min_active_s", 0.5),
                   rt.get("device_id", "laptop-dev"), f"{rt.get('event_id_prefix', 'laptop')}-front",
                   {"front": rt.get("model_versions", {}).get("front", "front")})

    def step(self, res) -> list[Action]:
        """res: FrontResult. Mengembalikan aksi keputusan (event start/end, buzz) untuk frame ini."""
        ts = int(res.ts_ms)
        codes = set(res.active_codes)
        for c in list(self._since):
            if c not in codes:
                del self._since[c]
        active = {}
        for c in codes:
            t0 = self._since.setdefault(c, ts)
            if ts - t0 >= self.min_active_ms:
                cand = [k for k in res.candidates if k.code == c]
                conf = max((k.confidence for k in cand), default=0.0)
                dist = min((k.distance_m for k in cand), default=None)
                active[c] = {"confidence": conf, "condition_start_ms": t0, "distance_m": dist}
        extra = None
        if active:
            top = min(active, key=self.decision.rank)
            if active[top]["distance_m"] is not None:
                extra = {"distance_m": round(float(active[top]["distance_m"]), 2), "camera": "front"}
        return self.decision.step(ts, active, extra or {"camera": "front"})


def write_snapshot(events_dir: str | Path, event: dict, img) -> str | None:
    """Simpan snapshot depan seperti ClipRecorder.paths_for; kembalikan path relatif seperti event kabin."""
    root = Path(events_dir)
    day = datetime.fromtimestamp(event["ts_event_ms"] / 1000).strftime("%Y-%m-%d")
    folder = root / day
    folder.mkdir(parents=True, exist_ok=True)
    p = folder / f"{event['event_id']}.jpg"
    out = img.copy()
    t = datetime.fromtimestamp(event["ts_event_ms"] / 1000).strftime("%Y-%m-%d %H:%M:%S")
    cv2.rectangle(out, (0, 0), (out.shape[1], 28), (0, 0, 0), -1)
    cv2.putText(out, f"{t}  {event['code']} {event['level']}  depan", (8, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.6,
                (255, 255, 255), 1, cv2.LINE_AA)
    if not cv2.imwrite(str(p), out):
        return None
    return p.relative_to(root.parent).as_posix() if root.parent in p.parents else str(p)
