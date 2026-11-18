"""Logika gateway tanpa jaringan: seq persisten, buffer replay, log, telemetri.

Kontrak pesan = `minerva-forklift/src/integrations/hardware.ts` (repo impactto). Pesan produsen boleh
tanpa `seq`/`forkliftId`; gateway yang mengisi. Kode boleh `D02` atau `D-02` (dinormalisasi ke `D-02`).
"""
from __future__ import annotations

import json
import os
import re
import threading
import time
from collections import deque
from datetime import datetime
from pathlib import Path

from src.config import resolve_path

_CODE_RE = re.compile(r"^([A-Z])-?(\d{2})$")
# pesan produsen yang diteruskan apa adanya ke dashboard (ditambah forkliftId)
PASSTHROUGH = {"gnss", "ups", "frame"}


def normalize_code(code: str) -> str:
    c = str(code).strip().upper()
    m = _CODE_RE.match(c)
    return f"{m.group(1)}-{m.group(2)}" if m else c


class Gateway:
    def __init__(self, cfg: dict, forklift_id: str, now_ms=None):
        self.c = cfg
        self.forklift_id = forklift_id
        self.now_ms = now_ms or (lambda: int(time.time() * 1000))
        self.dir = resolve_path(cfg.get("state_dir", "gateway_state"))
        self.dir.mkdir(parents=True, exist_ok=True)
        self.seq_path = self.dir / "seq.txt"
        self.events_path = self.dir / "events.jsonl"
        self.buffer: deque = deque(maxlen=int(cfg.get("replay_buffer", 500)))
        self.status: dict[str, dict] = {}         # status terakhir per produsen (mis. "cabin")
        self.status_fresh_ms = float(cfg.get("status_fresh_s", 6)) * 1000
        self._lock = threading.Lock()
        self.seq = self._load()

    # ---------------------------------------------------------------- persistensi
    def _load(self) -> int:
        seq = 0
        if self.seq_path.exists():
            try:
                seq = int(self.seq_path.read_text(encoding="utf-8").strip() or 0)
            except ValueError:
                seq = 0
        if self.events_path.exists():
            with open(self.events_path, encoding="utf-8") as f:
                for line in f:
                    try:
                        ev = json.loads(line)
                    except json.JSONDecodeError:
                        continue                   # baris terakhir bisa terpotong saat listrik putus
                    self.buffer.append(ev)
                    seq = max(seq, int(ev.get("seq", 0)))
        return seq

    def _save_seq(self):
        tmp = self.seq_path.with_suffix(".tmp")
        with open(tmp, "w", encoding="utf-8") as f:
            f.write(str(self.seq))
            f.flush()
            os.fsync(f.fileno())
        os.replace(tmp, self.seq_path)

    def _append_event(self, ev: dict):
        with open(self.events_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(ev, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())
        self._trim_events_file()

    def _trim_events_file(self):
        # file tidak tumbuh tanpa batas: simpan 2x ukuran buffer
        keep = self.buffer.maxlen or 500
        if self.seq % keep == 0 and self.events_path.exists():
            lines = self.events_path.read_text(encoding="utf-8").splitlines()[-2 * keep:]
            self.events_path.write_text("\n".join(lines) + "\n", encoding="utf-8")

    def log(self, direction: str, msg: dict):
        """Log sesi JSONL: semua event, shown/handled, replay, media. Cadangan Lembar A/B (jetson-scope 3.9)."""
        d = datetime.now().strftime("%Y-%m-%d")
        p = self.dir / "log" / f"{d}.jsonl"
        p.parent.mkdir(parents=True, exist_ok=True)
        with open(p, "a", encoding="utf-8") as f:
            f.write(json.dumps({"t_gateway_ms": self.now_ms(), "dir": direction, "msg": msg},
                               ensure_ascii=False) + "\n")

    # ---------------------------------------------------------------- produsen -> dashboard
    def ingest(self, msg: dict) -> list[str]:
        """Pesan dari produsen lokal. Mengembalikan teks JSON yang harus disiarkan ke dashboard."""
        if not isinstance(msg, dict):
            return []
        t = msg.get("type")
        if t == "event":
            if not msg.get("code"):
                self.log("reject", msg)
                return []
            key = (msg.get("event_id"), msg.get("state", "started"))
            with self._lock:
                # kirim ulang dari produsen (ack hilang) -> jangan jadi kejadian kedua di dashboard
                if key[0] is not None and any((e.get("event_id"), e.get("state")) == key for e in self.buffer):
                    self.log("duplicate", msg)
                    return []
                self.seq += 1
                ev = {k: v for k, v in msg.items() if v is not None and k != "ack_id"}
                ev.update(type="event", forkliftId=msg.get("forkliftId") or self.forklift_id, seq=self.seq,
                          code=normalize_code(msg["code"]), state=msg.get("state", "started"))
                self._save_seq()                   # seq disimpan SEBELUM dikirim
                self._append_event(ev)
                self.buffer.append(ev)
            self.log("out", ev)
            return [json.dumps(ev, ensure_ascii=False)]
        if t in PASSTHROUGH:
            out = dict(msg, forkliftId=msg.get("forkliftId") or self.forklift_id)
            return [json.dumps(out, ensure_ascii=False)]
        if t == "status":                          # tanda hidup produsen -> digabung ke telemetri
            self.status[str(msg.get("source", "cabin"))] = dict(msg, _rx_ms=self.now_ms())
            return []
        if t == "media":                           # info klip/snapshot: dicatat (dashboard belum punya player)
            self.log("media", msg)
            return []
        self.log("reject", msg)
        return []

    # ---------------------------------------------------------------- dashboard -> gateway
    def on_dashboard(self, msg: dict) -> list[str]:
        """Pesan dari dashboard. Mengembalikan balasan untuk klien itu saja (replay)."""
        if not isinstance(msg, dict):
            return []
        t = msg.get("type")
        if t in ("shown", "handled"):
            self.log(t, msg)
            return []
        if t == "replay":
            try:
                from_seq = int(msg.get("fromSeq", 0))
            except (TypeError, ValueError):
                from_seq = 0
            with self._lock:
                out = [json.dumps(ev, ensure_ascii=False) for ev in self.buffer if ev.get("seq", 0) >= from_seq]
            self.log("replay", {"fromSeq": from_seq, "count": len(out)})
            return out
        self.log("unknown_in", msg)
        return []

    # ---------------------------------------------------------------- telemetri
    def _fresh(self, source: str) -> dict | None:
        s = self.status.get(source)
        if s and self.now_ms() - s["_rx_ms"] <= self.status_fresh_ms:
            return s
        return None

    def telemetry(self, system: dict | None = None) -> str:
        cab = self._fresh("cabin")
        fwd = self._fresh("forward")
        m = {"type": "telemetry", "forkliftId": self.forklift_id, "ts_ms": self.now_ms()}
        m.update(system or {})
        m["fpsCabin"] = round(float(cab.get("fps", 0)), 1) if cab else 0
        m["fpsForward"] = round(float(fwd.get("fps", 0)), 1) if fwd else 0
        # usulan tambahan jetson-scope (belum ditampilkan dashboard)
        m["camOk"] = bool(cab.get("camera_ok")) if cab else False
        m["cabinAlive"] = cab is not None
        m["ntpSynced"] = cab.get("clock_synced") if cab else None
        m["seq"] = self.seq
        return json.dumps(m, ensure_ascii=False)


def system_stats() -> dict:
    """CPU/RAM/suhu/GPU. Jetson: suhu dari thermal_zone, beban GPU dari /sys/.../gpu.0/load (0..1000)."""
    out = {"cpuPct": 0.0, "memUsedMb": 0, "tempC": 0.0, "gpuPct": 0.0}
    try:
        import psutil
        out["cpuPct"] = psutil.cpu_percent(interval=None)
        out["memUsedMb"] = int(psutil.virtual_memory().used / 2**20)
    except Exception:
        pass
    temps = []
    for z in Path("/sys/class/thermal").glob("thermal_zone*/temp"):
        try:
            temps.append(int(z.read_text().strip()) / 1000)
        except Exception:
            pass
    if temps:
        out["tempC"] = round(max(temps), 1)
    for g in ("/sys/devices/platform/gpu.0/load", "/sys/devices/gpu.0/load",
              "/sys/devices/platform/17000000.gpu/load"):
        try:
            out["gpuPct"] = round(int(Path(g).read_text().strip()) / 10, 1)
            break
        except Exception:
            pass
    return out
