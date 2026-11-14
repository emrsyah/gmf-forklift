"""Tujuan event JSON: file JSONL (selalu) dan MQTT (opsional, QoS 1 + antrean lokal di disk)."""
from __future__ import annotations

import json
import threading
from datetime import datetime
from pathlib import Path

from src.config import resolve_path


class EventSink:
    def send(self, event: dict) -> None:
        raise NotImplementedError

    def close(self) -> None:
        pass


class MemorySink(EventSink):
    def __init__(self):
        self.events: list[dict] = []

    def send(self, event):
        self.events.append(dict(event))


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


class MqttSink(EventSink):
    """paho-mqtt QoS 1. Jika broker putus, event ditulis ke spool dan dikirim ulang saat tersambung (K1)."""

    def __init__(self, host: str, port: int, topic: str, qos: int = 1, spool_dir: str | Path = "events/_spool"):
        import paho.mqtt.client as mqtt

        self.topic = topic
        self.qos = qos
        self.spool = resolve_path(spool_dir)
        self.spool.mkdir(parents=True, exist_ok=True)
        try:
            self.client = mqtt.Client(mqtt.CallbackAPIVersion.VERSION2)
        except AttributeError:   # paho < 2.0
            self.client = mqtt.Client()
        self.client.on_connect = self._on_connect
        self.connected = False
        try:
            self.client.connect_async(host, int(port), keepalive=30)
            self.client.loop_start()
        except Exception:
            pass

    def _on_connect(self, client, userdata, flags, rc, *args):
        self.connected = (rc == 0) if isinstance(rc, int) else not getattr(rc, "is_failure", False)
        if self.connected:
            self._flush_spool()

    def _flush_spool(self):
        for p in sorted(self.spool.glob("*.json")):
            try:
                payload = p.read_text(encoding="utf-8")
                info = self.client.publish(self.topic, payload, qos=self.qos)
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
            self.client.loop_stop()
            self.client.disconnect()
        except Exception:
            pass


class MultiSink(EventSink):
    def __init__(self, sinks: list[EventSink]):
        self.sinks = sinks

    def send(self, event):
        for s in self.sinks:
            s.send(event)

    def close(self):
        for s in self.sinks:
            s.close()


def make_sinks(cfg: dict, device_id: str) -> MultiSink:
    sinks: list[EventSink] = []
    if cfg.get("file", {}).get("enabled", True):
        sinks.append(FileSink(cfg["file"].get("dir", "events")))
    m = cfg.get("mqtt", {})
    if m.get("enabled"):
        sinks.append(MqttSink(m.get("host", "localhost"), m.get("port", 1883),
                              m.get("topic", "forklift/{device_id}/events").format(device_id=device_id),
                              m.get("qos", 1), m.get("spool_dir", "events/_spool")))
    return MultiSink(sinks)
