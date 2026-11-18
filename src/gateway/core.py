"""Logika gateway tanpa jaringan: seq persisten, buffer replay, log, telemetri, URL bukti.

Kontrak pesan = `minerva-forklift/src/integrations/hardware.ts` (repo impactto). Pesan produsen boleh
tanpa `seq`/`forkliftId`; gateway yang mengisi. Kode boleh `D02` atau `D-02` (dinormalisasi ke `D-02`).

Bukti (K7): produsen mengirim path lokal (`snapshotPath`, `clipPath`, relatif ke root repo). Gateway
menyimpannya, lalu saat mengirim ke tiap dashboard mengubahnya menjadi URL HTTP media server gateway
(`snapshotUrl`, `clipUrl`) dengan host yang dipakai dashboard itu untuk tersambung.
Klip baru ada setelah kondisi selesai: saat itu produsen mengirim `media`, gateway memperbarui event di
buffer dan menyiarkan `{type:'media', seq, clipUrl, ...}`.
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
from urllib.parse import quote

from src.config import resolve_path

_CODE_RE = re.compile(r"^([A-Z])-?(\d{2})$")
# pesan produsen yang diteruskan apa adanya ke dashboard (ditambah forkliftId)
PASSTHROUGH = {"gnss", "ups", "frame"}
# field internal yang tidak dikirim ke dashboard (diganti URL)
_PRIVATE = {"snapshotPath", "clipPath", "ack_id", "run_id"}


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
        m = cfg.get("media", {}) or {}
        self.media_enabled = bool(m.get("enabled", True))
        self.media_root = resolve_path(m.get("root", "events"))
        self.media_port = int(m.get("port", 8789))
        self.media_public = m.get("public_base_url")   # null = http://<host dashboard>:<port>
        self._lock = threading.Lock()
        self.storage_error: str | None = None
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
                    if "_media_for" in ev:         # pembaruan klip untuk event yang sudah ada
                        self._apply_media(ev["_media_for"], ev)
                        continue
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

    def _append(self, rec: dict):
        # baris terakhir bisa terpotong (listrik putus) -> mulai di baris baru, supaya record ini tidak ikut rusak
        torn = False
        if self.events_path.exists() and self.events_path.stat().st_size > 0:
            with open(self.events_path, "rb") as f:
                f.seek(-1, os.SEEK_END)
                torn = f.read(1) != b"\n"
        with open(self.events_path, "a", encoding="utf-8") as f:
            f.write(("\n" if torn else "") + json.dumps(rec, ensure_ascii=False) + "\n")
            f.flush()
            os.fsync(f.fileno())

    def _trim_events_file(self):
        # file tidak tumbuh tanpa batas: simpan 2x ukuran buffer; tulis atomik (tmp + replace)
        keep = self.buffer.maxlen or 500
        if self.seq % keep == 0 and self.events_path.exists():
            lines = self.events_path.read_text(encoding="utf-8").splitlines()[-2 * keep:]
            tmp = self.events_path.with_suffix(".tmp")
            with open(tmp, "w", encoding="utf-8") as f:
                f.write("\n".join(lines) + "\n")
                f.flush()
                os.fsync(f.fileno())
            os.replace(tmp, self.events_path)

    def log(self, direction: str, msg: dict):
        """Log sesi JSONL: semua event, shown/handled, replay, media. Cadangan Lembar A/B (jetson-scope 3.9)."""
        d = datetime.now().strftime("%Y-%m-%d")
        p = self.dir / "log" / f"{d}.jsonl"
        try:
            p.parent.mkdir(parents=True, exist_ok=True)
            with open(p, "a", encoding="utf-8") as f:
                f.write(json.dumps({"t_gateway_ms": self.now_ms(), "dir": direction, "msg": msg},
                                   ensure_ascii=False) + "\n")
        except OSError as e:                       # log tidak boleh menghentikan pengiriman alarm
            self.storage_error = str(e)

    # ---------------------------------------------------------------- URL bukti
    def media_rel(self, path: str | None) -> str | None:
        """Path lokal produsen -> path relatif di bawah media root (None jika di luar root / tidak ada)."""
        if not path or not self.media_enabled:
            return None
        try:
            p = resolve_path(path).resolve()
            rel = p.relative_to(self.media_root.resolve())
        except (ValueError, OSError):
            return None
        return rel.as_posix()

    def media_base(self, host: str | None) -> str | None:
        if not self.media_enabled:
            return None
        if self.media_public:
            return str(self.media_public).rstrip("/")
        if not host:
            return None
        hostname = host.rsplit(":", 1)[0] if host.count(":") == 1 or host.startswith("[") else host
        return f"http://{hostname}:{self.media_port}"

    def render(self, msg: dict, host: str | None = None) -> str:
        """Pesan internal -> teks JSON untuk satu dashboard (path lokal diganti URL media)."""
        out = {k: v for k, v in msg.items() if k not in _PRIVATE and not k.startswith("_")}
        base = self.media_base(host)
        if base:
            for src, dst in (("snapshotPath", "snapshotUrl"), ("clipPath", "clipUrl")):
                rel = self.media_rel(msg.get(src))
                if rel and (src != "clipPath" or msg.get("clipReady", True)):
                    out[dst] = f"{base}/media/{quote(rel)}"
        out.pop("clipReady", None)
        return json.dumps(out, ensure_ascii=False)

    # ---------------------------------------------------------------- produsen -> dashboard
    def ingest(self, msg: dict) -> list[dict]:
        """Pesan dari produsen lokal. Mengembalikan pesan yang harus disiarkan ke dashboard (render per klien)."""
        if not isinstance(msg, dict):
            return []
        t = msg.get("type")
        if t == "event":
            if not msg.get("code"):
                self.log("reject", msg)
                return []
            key = (msg.get("run_id"), msg.get("event_id"), msg.get("state", "started"))
            with self._lock:
                # kirim ulang dari produsen (ack hilang) -> jangan jadi kejadian kedua di dashboard.
                # run_id: event_id hanya unik dalam satu proses produsen.
                if key[1] is not None and any((e.get("run_id"), e.get("event_id"), e.get("state")) == key
                                              for e in self.buffer):
                    self.log("duplicate", msg)
                    return []
                self.seq += 1
                ev = {k: v for k, v in msg.items() if v is not None and k != "ack_id"}
                ev.update(type="event", forkliftId=msg.get("forkliftId") or self.forklift_id, seq=self.seq,
                          code=normalize_code(msg["code"]), state=msg.get("state", "started"),
                          _t_gw=self.now_ms())          # waktu terima gateway (replay maxAgeS); tidak dikirim
                if ev.get("clipPath"):
                    ev["clipReady"] = False         # klip ditulis setelah kondisi selesai
                try:
                    self._save_seq()               # seq disimpan SEBELUM dikirim
                    self._append(ev)
                    self._trim_events_file()
                    self.storage_error = None
                except OSError as e:
                    # disk penuh / SD read-only: alarm TETAP dikirim (K4); replay setelah restart terganggu
                    self.storage_error = str(e)
                    print(f"[gateway] gagal menyimpan seq/event: {e}", flush=True)
                self.buffer.append(ev)
            self.log("out", ev)
            return [ev]
        if t in PASSTHROUGH:
            return [dict(msg, forkliftId=msg.get("forkliftId") or self.forklift_id)]
        if t == "status":                          # tanda hidup produsen -> digabung ke telemetri
            self.status[str(msg.get("source", "cabin"))] = dict(msg, _rx_ms=self.now_ms())
            return []
        if t == "media":                           # klip selesai ditulis
            self.log("media", msg)
            return self._on_media(msg)
        self.log("reject", msg)
        return []

    def _apply_media(self, seq: int, upd: dict) -> dict | None:
        for ev in self.buffer:
            if ev.get("seq") == seq:
                if upd.get("clipPath"):
                    ev["clipPath"] = upd["clipPath"]
                    ev["clipReady"] = True
                if upd.get("clipDurationS") is not None:
                    ev["clipDurationS"] = upd["clipDurationS"]
                return ev
        return None

    def _on_media(self, msg: dict) -> list[dict]:
        eid = msg.get("event_id")
        if not eid or not msg.get("clip_ok"):
            return []
        with self._lock:
            run = msg.get("run_id")
            start = next((e for e in reversed(self.buffer) if e.get("event_id") == eid
                          and e.get("run_id") == run and e.get("state") == "started"), None)
            if start is None:
                return []
            upd = {"_media_for": start["seq"], "clipPath": msg.get("clip_path"),
                   "clipDurationS": msg.get("clip_duration_s")}
            self._apply_media(start["seq"], upd)
            try:
                self._append(upd)
            except OSError as e:
                self.storage_error = str(e)
        return [self._media_msg(start)]

    @staticmethod
    def _media_msg(start: dict) -> dict:
        return {"type": "media", "forkliftId": start["forkliftId"], "seq": start["seq"], "code": start["code"],
                "event_id": start.get("event_id"), "snapshotPath": start.get("snapshotPath"),
                "clipPath": start.get("clipPath"), "clipReady": start.get("clipReady", False),
                "clipDurationS": start.get("clipDurationS")}

    # ---------------------------------------------------------------- dashboard -> gateway
    def on_dashboard(self, msg: dict) -> list[dict]:
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
            # maxAgeS: dashboard yang baru dibuka/di-reload cukup minta kejadian beberapa menit terakhir,
            # bukan 500 alarm lama (yang akan muncul lagi sebagai "perlu tindakan")
            max_age = msg.get("maxAgeS")
            cutoff = self.now_ms() - float(max_age) * 1000 if isinstance(max_age, (int, float)) else None
            with self._lock:
                out = [dict(ev) for ev in self.buffer if ev.get("seq", 0) >= from_seq
                       and (cutoff is None or ev.get("_t_gw", 0) >= cutoff)]
                # klip yang selesai saat dashboard terputus: event-nya (seq lama) tidak ikut replay,
                # jadi kirim ulang info media untuk event berklip terakhir (idempoten di dashboard)
                n_media = int(self.c.get("replay_media_last", 50))
                media = [self._media_msg(ev) for ev in list(self.buffer)[-n_media:]
                         if ev.get("state") == "started" and ev.get("clipReady") and ev.get("seq", 0) < from_seq]
            self.log("replay", {"fromSeq": from_seq, "maxAgeS": max_age, "count": len(out), "media": len(media)})
            return out + media
        self.log("unknown_in", msg)
        return []

    # ---------------------------------------------------------------- telemetri
    def _fresh(self, source: str) -> dict | None:
        s = self.status.get(source)
        if s and self.now_ms() - s["_rx_ms"] <= self.status_fresh_ms:
            return s
        return None

    def telemetry(self, system: dict | None = None) -> dict:
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
        m["storageOk"] = self.storage_error is None
        return m


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
