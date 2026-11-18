"""Heartbeat perangkat: status berkala untuk command center (perangkat hidup, kamera, model, jam).

Jam: K5 diukur dari `ts_event_ms` yang dibuat di edge, jadi command center perlu tahu apakah jam perangkat
tersinkron NTP. Di Linux/Jetson dibaca dari `timedatectl` (tanpa sudo); di OS lain `clock_synced` = null.
"""
from __future__ import annotations

import shutil
import subprocess
import time


def ntp_synced() -> bool | None:
    exe = shutil.which("timedatectl")
    if not exe:
        return None
    try:
        out = subprocess.run([exe, "show", "-p", "NTPSynchronized", "--value"],
                             capture_output=True, text=True, timeout=2).stdout.strip().lower()
        return {"yes": True, "no": False}.get(out)
    except Exception:
        return None


class Heartbeat:
    def __init__(self, cfg: dict | None, device_id: str, model_versions: dict | None = None,
                 clock_fn=ntp_synced):
        cfg = cfg or {}
        self.enabled = bool(cfg.get("enabled", True))
        self.interval_ms = float(cfg.get("interval_s", 5)) * 1000
        self.clock_check_ms = float(cfg.get("clock_check_s", 60)) * 1000
        self.device_id = device_id
        self.model_versions = dict(model_versions or {})
        self._clock_fn = clock_fn
        self._clock: bool | None = None
        self._clock_at: float | None = None
        self._last_ms: float | None = None
        self.t_start_ms = int(time.time() * 1000)

    def due(self, now_ms: int) -> bool:
        return self.enabled and (self._last_ms is None or now_ms - self._last_ms >= self.interval_ms)

    def _clock_synced(self, now_ms: int) -> bool | None:
        if self._clock_at is None or now_ms - self._clock_at >= self.clock_check_ms:
            self._clock = self._clock_fn()
            self._clock_at = now_ms
        return self._clock

    def build(self, now_ms: int, *, fps: float, camera_ok: bool, face_present: bool,
              calibration: str, yolo: str, open_events: list[str] | None = None) -> dict:
        self._last_ms = now_ms
        return {
            "device_id": self.device_id,
            "ts_ms": int(now_ms),
            "uptime_s": round((now_ms - self.t_start_ms) / 1000, 1),
            "fps": round(float(fps), 1),
            "camera_ok": bool(camera_ok),
            "face_present": bool(face_present),
            "calibration": calibration,
            "yolo": yolo,
            "clock_synced": self._clock_synced(now_ms),
            "open_events": list(open_events or []),
            "model_versions": self.model_versions,
        }
