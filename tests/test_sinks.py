"""Sink event: file JSONL dan MQTT dengan antrean lokal saat broker tidak tersambung."""
import json

from src.io.sinks import FileSink, MqttSink

EV = {"event_id": "laptop-x-0001", "phase": "start", "code": "D02", "level": "A-03",
      "ts_event_ms": 1794130000123}


def test_file_sink_jsonl(tmp_path):
    s = FileSink(tmp_path)
    s.send(EV)
    s.send({**EV, "phase": "end"})
    files = list(tmp_path.rglob("events.jsonl"))
    lines = files[0].read_text(encoding="utf-8").splitlines()
    assert [json.loads(l)["phase"] for l in lines] == ["start", "end"]


class _Info:
    def __init__(self, ok, delay=0.0):
        self.ok, self.delay, self.rc = ok, delay, 0

    def wait_for_publish(self, timeout=None):
        import time
        time.sleep(min(self.delay, timeout or 0))

    def is_published(self):
        return self.ok


class _Client:
    """Pengganti paho: catat publish; `ok` = broker membalas PUBACK."""

    def __init__(self, ok=True, delay=0.0):
        self.ok, self.delay, self.sent = ok, delay, []

    def publish(self, topic, payload, qos=0, retain=False):
        self.sent.append((topic, payload))
        return _Info(self.ok, self.delay)


def _spool_sink(tmp_path, n, client):
    s = MqttSink("127.0.0.1", 1, "forklift/test/events", spool_dir=tmp_path / "spool")
    for i in range(n):
        s.send({**EV, "event_id": f"e{i}", "ts_event_ms": EV["ts_event_ms"] + i})
    s.client.loop_stop()
    s.client = client
    return s


def test_mqtt_reconnect_flushes_spool_off_network_thread(tmp_path):
    import time
    s = _spool_sink(tmp_path, 3, _Client(ok=True, delay=0.2))
    t0 = time.perf_counter()
    s._on_connect(s.client, None, {}, 0)           # dipanggil dari thread jaringan paho
    assert time.perf_counter() - t0 < 0.1          # callback tidak menunggu PUBACK
    s._flush_thread.join(5)
    assert not list((tmp_path / "spool").glob("*.json"))
    assert [json.loads(p)["event_id"] for _, p in s.client.sent] == ["e0", "e1", "e2"]   # urut waktu


def test_mqtt_spool_kept_without_puback(tmp_path):
    s = _spool_sink(tmp_path, 2, _Client(ok=False))
    s.flush_timeout_s = 0.01
    s._on_connect(s.client, None, {}, 0)
    s._flush_thread.join(5)
    assert len(list((tmp_path / "spool").glob("*.json"))) == 2   # tidak dihapus, dicoba lagi nanti
    assert len(s.client.sent) == 1                                  # berhenti di file pertama yang gagal


def test_mqtt_spools_when_broker_down(tmp_path):
    s = MqttSink("127.0.0.1", 1, "forklift/test/events", spool_dir=tmp_path / "spool")   # port 1: tidak ada broker
    s.send(EV)
    s.close()
    spooled = list((tmp_path / "spool").glob("*.json"))
    assert len(spooled) == 1
    assert json.loads(spooled[0].read_text(encoding="utf-8"))["event_id"] == EV["event_id"]
