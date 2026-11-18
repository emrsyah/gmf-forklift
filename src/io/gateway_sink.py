"""Sink proses CV -> gateway lokal (src/gateway), dengan kontrak pesan dashboard MINERVA.

send() hanya memasukkan ke antrean; pengiriman di thread sendiri (loop CV tidak pernah menunggu jaringan).
Event yang belum terkirim tetap di antrean dan dikirim ulang setelah gateway tersambung lagi. Gateway
membalas `ack` per event; event baru dibuang dari antrean setelah di-ack (tidak hilang jika gateway restart).
Status (heartbeat) tidak diantrekan: yang lama tidak berguna.
"""
from __future__ import annotations

import json
import threading
import time
import uuid
from collections import deque

from .sinks import EventSink


def to_contract(ev: dict, forklift_id: str) -> dict:
    """Event internal (D02, phase start/end, ts_*_ms) -> pesan `event` dashboard (D-02, started/ended)."""
    code = str(ev["code"])
    if len(code) == 3 and code[1:].isdigit():
        code = f"{code[0]}-{code[1:]}"
    m = {
        "type": "event",
        "forkliftId": forklift_id,
        "code": code,
        "level": ev.get("level"),
        "state": "started" if ev.get("phase") == "start" else "ended",
        # t_deteksi = frame yang memicu alarm (jetson-scope 3.3), bukan waktu kirim
        "t_deteksi": ev.get("ts_event_ms"),
        "t_buzzer": ev.get("ts_buzzer_ms"),
        "confidence": ev.get("confidence"),
        "detail": ev.get("event_id"),
        # tambahan (diabaikan dashboard sekarang, berguna untuk log/analisis)
        "event_id": ev.get("event_id"),
        "t_kondisi_mulai": ev.get("ts_condition_start_ms"),
        "snapshotPath": ev.get("snapshot_path"),
        "clipPath": ev.get("clip_path"),
    }
    return {k: v for k, v in m.items() if v is not None}


class GatewaySink(EventSink):
    def __init__(self, url: str, forklift_id: str, retry_s: float = 2.0, max_queue: int = 2000,
                 ack_timeout_s: float = 2.0):
        self.url = url
        self.forklift_id = forklift_id
        # id per start proses: event_id hanya unik dalam satu proses (replay video yang sama, atau CV restart
        # di detik yang sama, menghasilkan event_id sama) -> dedup gateway memakai (run_id, event_id, state)
        self.run_id = uuid.uuid4().hex[:12]
        self.retry_s = retry_s
        self.ack_timeout = ack_timeout_s
        self.q: deque = deque(maxlen=max_queue)
        self._status: dict | None = None
        self._frame: dict | None = None             # hanya frame live terbaru; yang lama dibuang
        self.accepts_frames = True
        self._cv = threading.Condition()
        self._stop = False
        self.connected = False
        self._ack = 0
        self.th = threading.Thread(target=self._run, daemon=True, name="gateway-sink")
        self.th.start()

    # ------------------------------------------------------------- API sink
    def send(self, event):
        with self._cv:
            self.q.append(dict(to_contract(event, self.forklift_id), run_id=self.run_id))
            self._cv.notify()

    def heartbeat(self, payload):
        with self._cv:
            self._status = {"type": "status", "source": "cabin", "fps": payload.get("fps"),
                            "camera_ok": payload.get("camera_ok"), "face_present": payload.get("face_present"),
                            "calibration": payload.get("calibration"), "yolo": payload.get("yolo"),
                            "clock_synced": payload.get("clock_synced"), "ts_ms": payload.get("ts_ms")}
            self._cv.notify()

    def media(self, payload):
        with self._cv:
            self.q.append(dict(payload, type="media", run_id=self.run_id))
            self._cv.notify()

    def frame(self, payload):
        if not self.connected:
            return
        with self._cv:
            self._frame = payload
            self._cv.notify()

    def close(self, drain_s: float = 3.0):
        t_end = time.monotonic() + drain_s
        while self.q and self.connected and time.monotonic() < t_end:
            time.sleep(0.05)
        with self._cv:
            self._stop = True
            self._cv.notify()
        self.th.join(timeout=2)

    # ------------------------------------------------------------- thread
    def _run(self):
        from websockets.sync.client import connect

        while not self._stop:
            try:
                with connect(self.url, open_timeout=3, close_timeout=1) as ws:
                    self.connected = True
                    self._loop(ws)
            except Exception:
                pass
            self.connected = False
            if not self._stop:
                time.sleep(self.retry_s)

    def _loop(self, ws):
        while not self._stop:
            with self._cv:
                if not self.q and self._status is None and self._frame is None:
                    self._cv.wait(timeout=1.0)
                status, self._status = self._status, None
                head = self.q[0] if self.q else None
                # event dulu; frame live hanya dikirim saat antrean event kosong
                frame = None
                if head is None:
                    frame, self._frame = self._frame, None
            if status is not None:
                ws.send(json.dumps(status, ensure_ascii=False))
            if frame is not None:
                ws.send(json.dumps(frame))
            if head is None:
                continue
            if head.get("type") != "event":
                ws.send(json.dumps(head, ensure_ascii=False))
                with self._cv:
                    if self.q and self.q[0] is head:
                        self.q.popleft()
                continue
            self._ack += 1
            ws.send(json.dumps(dict(head, ack_id=self._ack), ensure_ascii=False))
            t_end = time.monotonic() + self.ack_timeout
            while True:                            # tunggu ack; pesan lain diabaikan
                left = t_end - time.monotonic()
                if left <= 0:
                    raise TimeoutError("gateway tidak membalas ack")
                reply = json.loads(ws.recv(timeout=left))
                if reply.get("type") == "ack" and reply.get("ack_id") == self._ack:
                    break
            with self._cv:
                if self.q and self.q[0] is head:
                    self.q.popleft()
