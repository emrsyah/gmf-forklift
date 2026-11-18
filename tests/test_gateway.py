"""Gateway dashboard MINERVA: seq persisten, replay, dedup, kontrak pesan, dan uji jaringan ujung-ke-ujung."""
import asyncio
import json
import socket
import threading
import time

import pytest

from src.gateway.core import Gateway, normalize_code
from src.io.gateway_sink import GatewaySink, to_contract

T = 1_795_000_000_000


def _gw(tmp_path, **kw):
    cfg = {"state_dir": str(tmp_path / "gw"), "replay_buffer": 5}
    cfg.update(kw)
    return Gateway(cfg, "FL-01", now_ms=lambda: T)


def _ev(i, code="D02", state="started"):
    return {"type": "event", "code": code, "level": "A-03", "state": state, "event_id": f"e{i}", "t_deteksi": T}


def test_normalize_code():
    assert normalize_code("D02") == "D-02" and normalize_code("d-10") == "D-10" and normalize_code("G-AREA") == "G-AREA"


def test_seq_survives_restart(tmp_path):
    gw = _gw(tmp_path)
    for i in range(3):
        gw.ingest(_ev(i))
    gw2 = _gw(tmp_path)                                      # Jetson restart
    out = json.loads(gw2.ingest(_ev(9))[0])
    assert out["seq"] == 4 and out["code"] == "D-02" and out["forkliftId"] == "FL-01"


def test_replay_from_seq_and_buffer_limit(tmp_path):
    gw = _gw(tmp_path)
    for i in range(8):
        gw.ingest(_ev(i))
    out = [json.loads(m)["seq"] for m in gw.on_dashboard({"type": "replay", "forkliftId": "FL-01", "fromSeq": 6})]
    assert out == [6, 7, 8]
    assert len(gw.on_dashboard({"type": "replay", "fromSeq": 0})) == 5      # buffer 5


def test_duplicate_resend_is_not_a_new_event(tmp_path):
    gw = _gw(tmp_path)
    gw.ingest(_ev(1))
    assert gw.ingest(_ev(1)) == []                            # ack hilang -> dikirim ulang
    assert json.loads(gw.ingest(_ev(1, state="ended"))[0])["seq"] == 2


def test_event_without_code_rejected_and_passthrough(tmp_path):
    gw = _gw(tmp_path)
    assert gw.ingest({"type": "event"}) == []
    g = json.loads(gw.ingest({"type": "gnss", "lat": -6.1, "lon": 106.6, "sats": 8})[0])
    assert g["forkliftId"] == "FL-01" and gw.seq == 0         # gnss tidak memakai seq


def test_shown_handled_logged(tmp_path):
    gw = _gw(tmp_path)
    gw.ingest(_ev(1))
    gw.on_dashboard({"type": "shown", "seq": 1, "t_tampil": T + 50})
    gw.on_dashboard({"type": "handled", "seq": 1, "t_ditangani": T + 9000, "by": "petugas"})
    log = next((tmp_path / "gw" / "log").glob("*.jsonl")).read_text(encoding="utf-8")
    assert '"shown"' in log and '"handled"' in log


def test_telemetry_uses_fresh_cabin_status(tmp_path):
    now = [T]
    gw = Gateway({"state_dir": str(tmp_path / "gw"), "status_fresh_s": 6}, "FL-01", now_ms=lambda: now[0])
    gw.ingest({"type": "status", "source": "cabin", "fps": 14.6, "camera_ok": True, "clock_synced": True})
    t = json.loads(gw.telemetry({"cpuPct": 10}))
    assert t["fpsCabin"] == 14.6 and t["camOk"] and t["cabinAlive"] and t["cpuPct"] == 10
    now[0] += 7000                                            # proses CV mati
    t = json.loads(gw.telemetry())
    assert t["fpsCabin"] == 0 and not t["cabinAlive"]


def test_to_contract():
    ev = {"event_id": "laptop-1", "phase": "start", "code": "D09", "level": "A-03", "ts_event_ms": T,
          "ts_condition_start_ms": T - 1000, "ts_buzzer_ms": T + 3, "confidence": 0.9,
          "snapshot_path": "events/x.jpg"}
    m = to_contract(ev, "FL-01")
    assert m["code"] == "D-09" and m["state"] == "started" and m["t_deteksi"] == T and m["t_buzzer"] == T + 3
    assert to_contract(dict(ev, phase="end"), "FL-01")["state"] == "ended"


# ------------------------------------------------------------------ jaringan ujung-ke-ujung
def _free_port():
    s = socket.socket()
    s.bind(("127.0.0.1", 0))
    p = s.getsockname()[1]
    s.close()
    return p


@pytest.fixture
def server(tmp_path):
    from src.gateway.server import GatewayServer
    dport, pport = _free_port(), _free_port()
    cfg = {"state_dir": str(tmp_path / "gw"), "telemetry_s": 0.2,
           "dashboard": {"host": "127.0.0.1", "port": dport}, "producer": {"host": "127.0.0.1", "port": pport}}
    loop = asyncio.new_event_loop()
    stop = asyncio.Event()
    srv = GatewayServer(Gateway(cfg, "FL-01"), cfg)
    th = threading.Thread(target=lambda: loop.run_until_complete(srv.run(stop)), daemon=True)
    th.start()
    time.sleep(0.5)
    yield dport, pport
    loop.call_soon_threadsafe(stop.set)
    th.join(timeout=3)


def _recv_until(ws, pred, timeout=5.0):
    t_end = time.monotonic() + timeout
    while time.monotonic() < t_end:
        m = json.loads(ws.recv(timeout=max(0.05, t_end - time.monotonic())))
        if pred(m):
            return m
    raise TimeoutError


def test_end_to_end_cv_to_dashboard(server):
    from websockets.sync.client import connect
    dport, pport = server
    with connect(f"ws://127.0.0.1:{dport}") as dash:
        sink = GatewaySink(f"ws://127.0.0.1:{pport}", "FL-01", retry_s=0.2)
        t0 = time.time()
        sink.send({"event_id": "x1", "phase": "start", "code": "D02", "level": "A-03",
                   "ts_event_ms": T, "ts_buzzer_ms": T + 2})
        sink.heartbeat({"fps": 15.0, "camera_ok": True})
        ev = _recv_until(dash, lambda m: m["type"] == "event")
        assert ev["code"] == "D-02" and ev["seq"] == 1 and ev["state"] == "started" and "ack_id" not in ev
        assert time.time() - t0 < 1.0                          # lokal: jauh di bawah anggaran K5
        tel = _recv_until(dash, lambda m: m["type"] == "telemetry" and m["fpsCabin"] > 0)
        assert tel["camOk"] is True
        sink.send({"event_id": "x1", "phase": "end", "code": "D02", "level": "A-03", "ts_event_ms": T + 3000})
        assert _recv_until(dash, lambda m: m["type"] == "event")["state"] == "ended"
        sink.close()
    # dashboard tersambung lagi -> replay dari seq 1
    with connect(f"ws://127.0.0.1:{dport}") as dash:
        dash.send(json.dumps({"type": "replay", "forkliftId": "FL-01", "fromSeq": 1}))
        got = [_recv_until(dash, lambda m: m["type"] == "event")["seq"] for _ in range(2)]
        assert got == [1, 2]


def test_events_queued_while_gateway_down(tmp_path):
    from src.gateway.server import GatewayServer
    from websockets.sync.client import connect
    dport, pport = _free_port(), _free_port()
    sink = GatewaySink(f"ws://127.0.0.1:{pport}", "FL-01", retry_s=0.2)
    sink.send({"event_id": "q1", "phase": "start", "code": "D05", "level": "A-02", "ts_event_ms": T})
    time.sleep(0.5)
    assert not sink.connected and len(sink.q) == 1             # gateway belum jalan: event tidak hilang
    cfg = {"state_dir": str(tmp_path / "gw"), "dashboard": {"host": "127.0.0.1", "port": dport},
           "producer": {"host": "127.0.0.1", "port": pport}}
    loop, stop = asyncio.new_event_loop(), asyncio.Event()
    th = threading.Thread(target=lambda: loop.run_until_complete(GatewayServer(Gateway(cfg, "FL-01"), cfg).run(stop)),
                          daemon=True)
    th.start()
    try:
        t_end = time.monotonic() + 5
        while sink.q and time.monotonic() < t_end:
            time.sleep(0.05)
        assert not sink.q
        with connect(f"ws://127.0.0.1:{dport}") as dash:
            dash.send(json.dumps({"type": "replay", "fromSeq": 1}))
            assert _recv_until(dash, lambda m: m["type"] == "event")["code"] == "D-05"
    finally:
        sink.close(drain_s=0)
        loop.call_soon_threadsafe(stop.set)
        th.join(timeout=3)
