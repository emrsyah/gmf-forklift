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
    out = gw2.ingest(_ev(9))[0]
    assert out["seq"] == 4 and out["code"] == "D-02" and out["forkliftId"] == "FL-01"


def test_replay_from_seq_and_buffer_limit(tmp_path):
    gw = _gw(tmp_path)
    for i in range(8):
        gw.ingest(_ev(i))
    out = [m["seq"] for m in gw.on_dashboard({"type": "replay", "forkliftId": "FL-01", "fromSeq": 6})]
    assert out == [6, 7, 8]
    assert len(gw.on_dashboard({"type": "replay", "fromSeq": 0})) == 5      # buffer 5


def test_duplicate_resend_is_not_a_new_event(tmp_path):
    gw = _gw(tmp_path)
    gw.ingest(_ev(1))
    assert gw.ingest(_ev(1)) == []                            # ack hilang -> dikirim ulang
    assert gw.ingest(_ev(1, state="ended"))[0]["seq"] == 2


def test_same_event_id_from_new_process_is_not_a_duplicate(tmp_path):
    # replay video yang sama / CV restart -> event_id sama, run_id beda -> kejadian baru, bukan duplikat
    gw = _gw(tmp_path)
    assert gw.ingest(dict(_ev(1), run_id="a"))[0]["seq"] == 1
    assert gw.ingest(dict(_ev(1), run_id="a")) == []
    assert gw.ingest(dict(_ev(1), run_id="b"))[0]["seq"] == 2
    assert "run_id" not in json.loads(gw.render(gw.buffer[-1], "h:8787"))


def test_torn_last_line_does_not_swallow_next_event(tmp_path):
    gw = _gw(tmp_path)
    gw.ingest(_ev(1))
    with open(gw.events_path, "a", encoding="utf-8") as f:
        f.write('{"type": "event", "seq": 2, "co')             # listrik putus di tengah tulis
    gw.ingest(_ev(3))
    gw2 = _gw(tmp_path)
    assert [e["event_id"] for e in gw2.buffer] == ["e1", "e3"]


def test_replay_max_age_and_media_resend(tmp_path):
    now = [T]
    root = tmp_path / "events"
    gw = Gateway({"state_dir": str(tmp_path / "gw"), "media": {"root": str(root)}}, "FL-01", now_ms=lambda: now[0])
    gw.ingest(dict(_ev(1), clipPath=str(root / "a.mp4")))
    now[0] += 300_000                                          # 5 menit kemudian
    gw.ingest(_ev(2))
    out = gw.on_dashboard({"type": "replay", "fromSeq": 0, "maxAgeS": 120})
    assert [m["seq"] for m in out] == [2]                       # alarm lama tidak dihidupkan lagi
    gw.ingest({"type": "media", "event_id": "e1", "clip_ok": True, "clip_path": str(root / "a.mp4")})
    out = gw.on_dashboard({"type": "replay", "fromSeq": 3})     # dashboard sempat terputus saat klip selesai
    assert any(m["type"] == "media" and m["seq"] == 1 for m in out)


def test_storage_error_still_delivers_event(tmp_path, monkeypatch):
    gw = _gw(tmp_path)

    def boom(*a, **k):
        raise OSError("No space left on device")
    monkeypatch.setattr(gw, "_save_seq", boom)
    out = gw.ingest(_ev(1))
    assert out and out[0]["seq"] == 1 and gw.telemetry()["storageOk"] is False


def test_event_without_code_rejected_and_passthrough(tmp_path):
    gw = _gw(tmp_path)
    assert gw.ingest({"type": "event"}) == []
    g = gw.ingest({"type": "gnss", "lat": -6.1, "lon": 106.6, "sats": 8})[0]
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
    t = gw.telemetry({"cpuPct": 10})
    assert t["fpsCabin"] == 14.6 and t["camOk"] and t["cabinAlive"] and t["cpuPct"] == 10
    now[0] += 7000                                            # proses CV mati
    t = gw.telemetry()
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
    dport, pport, mport = _free_port(), _free_port(), _free_port()
    media_root = tmp_path / "events"
    media_root.mkdir()
    cfg = {"state_dir": str(tmp_path / "gw"), "telemetry_s": 0.2,
           "dashboard": {"host": "127.0.0.1", "port": dport}, "producer": {"host": "127.0.0.1", "port": pport},
           "media": {"enabled": True, "host": "127.0.0.1", "port": mport, "root": str(media_root)}}
    loop = asyncio.new_event_loop()
    stop = asyncio.Event()
    srv = GatewayServer(Gateway(cfg, "FL-01"), cfg)
    th = threading.Thread(target=lambda: loop.run_until_complete(srv.run(stop)), daemon=True)
    th.start()
    time.sleep(0.5)
    yield dport, pport, mport, media_root
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
    dport, pport, *_ = server
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
           "producer": {"host": "127.0.0.1", "port": pport}, "media": {"enabled": False}}
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


# ------------------------------------------------------------------ bukti (snapshot/klip) + live
def _http(url, headers=None):
    import urllib.error
    import urllib.request
    req = urllib.request.Request(url, headers=headers or {})
    try:
        with urllib.request.urlopen(req, timeout=5) as r:
            return r.status, dict(r.headers), r.read()
    except urllib.error.HTTPError as e:
        return e.code, dict(e.headers), b""


def test_media_urls_follow_dashboard_host(tmp_path):
    root = tmp_path / "events"
    (root / "2026-11-18").mkdir(parents=True)
    snap = root / "2026-11-18" / "e1.jpg"
    snap.write_bytes(b"jpg")
    gw = Gateway({"state_dir": str(tmp_path / "gw"), "media": {"root": str(root), "port": 8789}}, "FL-01")
    ev = gw.ingest(dict(_ev(1), snapshotPath=str(snap), clipPath=str(root / "2026-11-18" / "e1.mp4")))[0]
    m = json.loads(gw.render(ev, "192.168.1.20:8787"))
    assert m["snapshotUrl"] == "http://192.168.1.20:8789/media/2026-11-18/e1.jpg"
    assert "clipUrl" not in m and "snapshotPath" not in m          # klip belum selesai ditulis
    upd = gw.ingest({"type": "media", "event_id": "e1", "clip_ok": True,
                     "clip_path": str(root / "2026-11-18" / "e1.mp4"), "clip_duration_s": 12.5})[0]
    mm = json.loads(gw.render(upd, "localhost:8787"))
    assert mm["type"] == "media" and mm["seq"] == 1 and mm["clipUrl"].endswith("/media/2026-11-18/e1.mp4")
    gw2 = Gateway({"state_dir": str(tmp_path / "gw"), "media": {"root": str(root)}}, "FL-01")   # restart
    rep = json.loads(gw2.render(gw2.on_dashboard({"type": "replay", "fromSeq": 1})[0], "h:8787"))
    assert rep["clipUrl"].endswith("e1.mp4") and rep["clipDurationS"] == 12.5


def test_path_outside_media_root_gets_no_url(tmp_path):
    gw = Gateway({"state_dir": str(tmp_path / "gw"), "media": {"root": str(tmp_path / "events")}}, "FL-01")
    ev = gw.ingest(dict(_ev(1), snapshotPath=str(tmp_path / "secret.jpg")))[0]
    assert "snapshotUrl" not in json.loads(gw.render(ev, "h:8787"))


def test_media_http_server_range_and_traversal(server):
    mport, root = server[2], server[3]
    (root / "d").mkdir()
    (root / "d" / "c.mp4").write_bytes(bytes(range(256)) * 4)       # 1024 byte
    (root / "d" / "events.jsonl").write_text("{}", encoding="utf-8")
    base = f"http://127.0.0.1:{mport}/media"
    code, h, body = _http(f"{base}/d/c.mp4")
    assert code == 200 and len(body) == 1024 and h["Content-Type"] == "video/mp4"
    code, h, body = _http(f"{base}/d/c.mp4", {"Range": "bytes=100-199"})
    assert code == 206 and body == (bytes(range(256)) * 4)[100:200] and h["Content-Range"] == "bytes 100-199/1024"
    assert _http(f"{base}/d/c.mp4", {"Range": "bytes=2000-"})[0] == 416
    assert _http(f"{base}/d/events.jsonl")[0] == 404                # hanya .jpg/.mp4
    assert _http(f"{base}/..%2F..%2Fgw%2Fseq.txt")[0] == 404         # path traversal
    assert _http(f"http://127.0.0.1:{mport}/d/c.mp4")[0] == 404       # di luar /media/


def test_live_frames_forwarded_and_dropped_when_busy(server):
    from websockets.sync.client import connect

    import numpy as np
    from src.io.live import LiveFrames
    dport, pport, *_ = server
    live = LiveFrames({"enabled": True, "fps": 4, "max_side": 160})
    assert live.due(0)
    fr = live.encode(np.full((480, 640, 3), 90, np.uint8), 0)
    assert not live.due(100) and live.due(250)
    with connect(f"ws://127.0.0.1:{dport}") as dash:
        sink = GatewaySink(f"ws://127.0.0.1:{pport}", "FL-01", retry_s=0.2)
        t_end = time.monotonic() + 3
        while not sink.connected and time.monotonic() < t_end:
            time.sleep(0.05)
        sink.frame(fr)
        m = _recv_until(dash, lambda x: x["type"] == "frame")
        assert m["camera"] == "cabin" and m["forkliftId"] == "FL-01" and len(m["jpegB64"]) > 100
        sink.close(drain_s=0)
