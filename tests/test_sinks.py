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


def test_mqtt_spools_when_broker_down(tmp_path):
    s = MqttSink("127.0.0.1", 1, "forklift/test/events", spool_dir=tmp_path / "spool")   # port 1: tidak ada broker
    s.send(EV)
    s.close()
    spooled = list((tmp_path / "spool").glob("*.json"))
    assert len(spooled) == 1
    assert json.loads(spooled[0].read_text(encoding="utf-8"))["event_id"] == EV["event_id"]
