"""Tujuan event JSON: file JSONL (selalu) dan MQTT (opsional, QoS 1 + antrean lokal di disk)."""
from __future__ import annotations

import json
import os
import threading
from datetime import datetime
from pathlib import Path

from src.config import resolve_path


class EventSink:
    def send(self, event: dict) -> None:
        raise NotImplementedError

    def heartbeat(self, payload: dict) -> None:
        """Status perangkat berkala (hidup, FPS, kamera, jam). Default: diabaikan."""

    def media(self, payload: dict) -> None:
        """Info file bukti (snapshot/klip) setelah selesai ditulis. Default: diabaikan."""

    def frame(self, payload: dict) -> None:
        """Frame live resolusi rendah (pesan `frame`). Default: diabaikan."""

    def close(self) -> None:
        pass


class MemorySink(EventSink):
    def __init__(self):
        self.events: list[dict] = []
        self.heartbeats: list[dict] = []
        self.medias: list[dict] = []

    def send(self, event):
        self.events.append(dict(event))

    def heartbeat(self, payload):
        self.heartbeats.append(dict(payload))

    def media(self, payload):
        self.medias.append(dict(payload))


class FileSink(EventSink):
    """events/YYYY-MM-DD/events.jsonl"""

    def __init__(self, root: str | Path):
        self.root = resolve_path(root)
        self._lock = threading.Lock()

    def path_for(self, ts_ms: int) -> Path:
        d = datetime.fromtimestamp(ts_ms / 1000).strftime("%Y-%m-%d")
        p = self.root / d
        p.mkdir(parents=True, exist_ok=True)
        return p / "events.jsonl"

    def send(self, event):
        with self._lock, open(self.path_for(event["ts_event_ms"]), "a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")

    def media(self, payload):
        # bukti lokal: events/YYYY-MM-DD/media.jsonl (heartbeat tidak ditulis ke file, terlalu sering)
        p = self.path_for(payload["ts_event_ms"]).with_name("media.jsonl")
        with self._lock, open(p, "a", encoding="utf-8") as f:
            f.write(json.dumps(payload, ensure_ascii=False) + "\n")


class MqttSink(EventSink):
    """paho-mqtt QoS 1. Jika broker putus, event ditulis ke spool dan dikirim ulang saat tersambung (K1).

    Status perangkat: Last Will (retained) `offline` di status_topic diterbitkan broker jika koneksi putus
    tanpa pamit; saat tersambung dikirim `online` (retained); saat close dikirim `offline` dengan rapi.
    """

    def __init__(self, host: str, port: int, topic: str, qos: int = 1, spool_dir: str | Path = "events/_spool",
                 status_topic: str | None = None, heartbeat_topic: str | None = None,
                 media_topic: str | None = None, device_id: str = "", keepalive_s: int = 30,
                 username: str | None = None, password: str | None = None):
        import paho.mqtt.client as mqtt

        self.topic = topic
        self.qos = qos
        self.status_topic = status_topic
        self.heartbeat_topic = heartbeat_topic
        self.media_topic = media_topic
        self.device_id = device_id
        self.spool = resolve_path(spool_dir)
        self.spool.mkdir(parents=True, exist_ok=True)
        try:
            self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        except AttributeError:   # paho < 2.0
            self.client = mqtt.Client()
        if username:
            self.client.username_pw_set(username, password)
        if status_topic:
            self.client.will_set(status_topic, self._status("offline", graceful=False), qos=1, retain=True)
        self.client.on_connect = self._on_connect
        self.client.on_disconnect = self._on_disconnect
        self.connected = False
        try:
            self.client.connect_async(host, int(port), keepalive=int(keepalive_s))
            self.client.loop_start()
        except Exception:
            pass

    def _status(self, state: str, graceful: bool = True) -> str:
        import time
        return json.dumps({"device_id": self.device_id, "state": state, "graceful": graceful,
                           "ts_ms": int(time.time() * 1000)})

    def _on_connect(self, client, userdata, flags, rc, *args):
        self.connected = (rc == 0) if isinstance(rc, int) else not getattr(rc, "is_failure", False)
        if self.connected:
            if self.status_topic:
                self.client.publish(self.status_topic, self._status("online"), qos=1, retain=True)
            self._flush_spool()

    def _on_disconnect(self, *args):
        self.connected = False

    def heartbeat(self, payload):
        if self.heartbeat_topic and self.connected:
            try:   # QoS 0, retained: dashboard yang baru buka langsung melihat status terakhir
                self.client.publish(self.heartbeat_topic, json.dumps(payload, ensure_ascii=False), qos=0,
                                    retain=True)
            except Exception:
                pass

    def media(self, payload):
        if not self.media_topic:
            return
        body = json.dumps(payload, ensure_ascii=False)
        if self.connected:
            try:
                if self.client.publish(self.media_topic, body, qos=self.qos).rc == 0:
                    return
            except Exception:
                pass
        name = f"{payload['ts_event_ms']}_{payload['event_id']}_media.json"
        (self.spool / name).write_text(body, encoding="utf-8")

    def _flush_spool(self):
        for p in sorted(self.spool.glob("*.json")):
            try:
                payload = p.read_text(encoding="utf-8")
                topic = self.media_topic if p.stem.endswith("_media") and self.media_topic else self.topic
                info = self.client.publish(topic, payload, qos=self.qos)
                info.wait_for_publish(timeout=5)
                if info.is_published():
                    p.unlink()
            except Exception:
                break

    def send(self, event):
        payload = json.dumps(event, ensure_ascii=False)
        if self.connected:
            try:
                info = self.client.publish(self.topic, payload, qos=self.qos)
                if info.rc == 0:
                    return
            except Exception:
                pass
        name = f"{event['ts_event_ms']}_{event['event_id']}_{event['phase']}.json"
        (self.spool / name).write_text(payload, encoding="utf-8")

    def close(self):
        try:
            if self.status_topic and self.connected:
                info = self.client.publish(self.status_topic, self._status("offline"), qos=1, retain=True)
                info.wait_for_publish(timeout=2)
            self.client.disconnect()
            self.client.loop_stop()
        except Exception:
            pass


class MultiSink(EventSink):
    def __init__(self, sinks: list[EventSink]):
        self.sinks = sinks

    def send(self, event):
        for s in self.sinks:
            s.send(event)

    def heartbeat(self, payload):
        for s in self.sinks:
            s.heartbeat(payload)

    def media(self, payload):
        for s in self.sinks:
            s.media(payload)

    def frame(self, payload):
        for s in self.sinks:
            s.frame(payload)

    def wants_frames(self) -> bool:
        return any(getattr(s, "accepts_frames", False) for s in self.sinks)

    def close(self):
        for s in self.sinks:
            s.close()


def make_sinks(cfg: dict, device_id: str) -> MultiSink:
    sinks: list[EventSink] = []
    if cfg.get("file", {}).get("enabled", True):
        sinks.append(FileSink(cfg["file"].get("dir", "events")))
    m = cfg.get("mqtt", {})
    if m.get("enabled"):
        fmt = lambda key, default: (m.get(key, default) or None) and m.get(key, default).format(device_id=device_id)
        sinks.append(MqttSink(m.get("host", "localhost"), m.get("port", 1883),
                              fmt("topic", "forklift/{device_id}/events"),
                              m.get("qos", 1), m.get("spool_dir", "events/_spool"),
                              status_topic=fmt("status_topic", None),
                              heartbeat_topic=fmt("heartbeat_topic", None),
                              media_topic=fmt("media_topic", None),
                              device_id=device_id, keepalive_s=m.get("keepalive_s", 30),
                              # kata sandi TIDAK ditulis di YAML: dibaca dari variabel lingkungan
                              username=m.get("username"),
                              password=os.environ.get(m.get("password_env") or "") or None))
    g = cfg.get("gateway", {})
    if g.get("enabled"):
        from .gateway_sink import GatewaySink
        sinks.append(GatewaySink(g.get("url", "ws://127.0.0.1:8788"), g.get("forklift_id", "FL-01"),
                                 retry_s=g.get("retry_s", 2.0), max_queue=g.get("max_queue", 2000)))
    return MultiSink(sinks)
