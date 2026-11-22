"""tools/jetson/netwatch.py: ringkasan status 5G + agen."""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "tools" / "jetson"))
import netwatch as nw  # noqa: E402

STATUS = {"modem": {"act": "5G SA", "registered": True},
          "cell": {"rsrp": -82, "sinr": 25},
          "ping": {"ms": 23.3},
          "data": {"iface": "enx42", "ip": "10.111.253.66"}}


def test_ok_via_modem():
    row = nw.summarize(STATUS, {"last_ok": 99.0, "rtt_ms": 500, "pending_events": 0}, "enx42", 100.0, 15)
    assert row["ok"] and row["via_modem"]
    assert row["radio"]["rsrp"] == -82 and row["agent"]["last_ok_age_s"] == 1.0


def test_stale_agent_is_down_and_wifi_route_flagged():
    row = nw.summarize(STATUS, {"last_ok": 50.0, "last_err": "timed out"}, "wlP1p1s0", 100.0, 15)
    assert not row["ok"] and not row["via_modem"]
    assert "50.0 s" in nw.reason(row)


def test_camera_status(tmp_path):
    for i, name in enumerate(["vi-output, imx219 10-0010", "Logitech Webcam C930e", "Logitech Webcam C930e"]):
        (tmp_path / f"video{i}").mkdir()
        (tmp_path / f"video{i}" / "name").write_text(name + "\n")
    names = nw.video_devices(str(tmp_path))
    assert nw.camera_status(names, {"cabin": "imx219", "front": "c930e"}) == {"cabin": True, "front": True}
    assert nw.camera_status(names[:1], {"cabin": "imx219", "front": "C930e"}) == {"cabin": True, "front": False}
    assert nw.video_devices(str(tmp_path / "missing")) == []
    row = nw.summarize(None, None, None, 100.0, 15, {"front": False})
    assert row["cameras"] == {"front": False}


def test_services_missing():
    row = nw.summarize(None, None, None, 100.0, 15)
    assert not row["ok"] and nw.reason(row).startswith("monitor modem")
    row = nw.summarize({**STATUS, "modem": {"act": "LTE", "registered": False}}, None, "enx42", 100.0, 15)
    assert "tidak terdaftar" in nw.reason(row)
