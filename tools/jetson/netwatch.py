"""Catat kesehatan jalur 5G -> server MINERVA di Jetson, terus-menerus (stdlib saja).

Tiap --interval-s detik satu baris JSONL di <--log-dir>/netwatch-YYYYMMDD.jsonl:
    radio  : teknologi, RSRP/SINR, ping ke internet lewat modem      (modem_monitor.py :8088/api/status)
    agent  : umur kontak terakhir dengan server, RTT, event tertunda  (minerva_edge.py  :8098/)
    route  : interface jalur default (modem 5G atau Wi-Fi)
    ok     : radio terdaftar + agen sempat kontak server < --stale-s detik
    cameras: kamera kabin/depan terpasang atau tidak (nama perangkat di /sys/class/video4linux)
Perubahan ok <-> putus dan kamera terpasang <-> terlepas juga dicetak ke stdout (log run_netwatch.sh).

    python3 tools/jetson/netwatch.py --log-dir ~/gmf-forklift/logs
"""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import time
import urllib.request


def fetch_json(url: str, timeout_s: float) -> dict | None:
    try:
        with urllib.request.urlopen(url, timeout=timeout_s) as r:
            return json.loads(r.read().decode("utf-8", "replace"))
    except Exception:  # noqa: BLE001 - layanan lain mati = dicatat sebagai None
        return None


def default_route_iface() -> str | None:
    """Interface dengan metric jalur default terkecil (yang benar-benar dipakai untuk internet)."""
    try:
        out = subprocess.run(["ip", "route", "show", "default"], capture_output=True, text=True, timeout=3).stdout
    except Exception:  # noqa: BLE001
        return None
    best = None
    for line in out.splitlines():
        parts = line.split()
        if "dev" not in parts:
            continue
        dev = parts[parts.index("dev") + 1]
        metric = int(parts[parts.index("metric") + 1]) if "metric" in parts else 0
        if best is None or metric < best[0]:
            best = (metric, dev)
    return best[1] if best else None


def video_devices(root: str = "/sys/class/video4linux") -> list[str]:
    """Nama semua perangkat video yang terpasang (mis. 'vi-output, imx219 10-0010', 'Logitech Webcam C930e')."""
    names = []
    try:
        for d in sorted(os.listdir(root)):
            try:
                with open(os.path.join(root, d, "name"), encoding="utf-8") as f:
                    names.append(f.read().strip())
            except OSError:
                pass
    except OSError:
        pass
    return names


def camera_status(names: list[str], expected: dict[str, str]) -> dict[str, bool]:
    """{peran: terpasang?}; expected = {peran: potongan nama perangkat}, cocok tanpa beda huruf besar/kecil."""
    low = [n.lower() for n in names]
    return {role: any(key.lower() in n for n in low) for role, key in expected.items()}


def summarize(status: dict | None, agent: dict | None, route: str | None, now: float, stale_s: float,
              cameras: dict[str, bool] | None = None) -> dict:
    modem = (status or {}).get("modem") or {}
    cell = (status or {}).get("cell") or {}
    ping = (status or {}).get("ping") or {}
    data = (status or {}).get("data") or {}
    radio = {
        "up": status is not None,
        "act": modem.get("act"),
        "registered": modem.get("registered"),
        "rsrp": cell.get("rsrp"),
        "sinr": cell.get("sinr"),
        "ping_ms": ping.get("ms"),
        "iface": data.get("iface"),
        "ip": data.get("ip"),
    }
    last_ok = (agent or {}).get("last_ok")
    agent_s = {
        "up": agent is not None,
        "server": (agent or {}).get("server"),
        "last_ok_age_s": round(now - last_ok, 1) if isinstance(last_ok, (int, float)) else None,
        "rtt_ms": (agent or {}).get("rtt_ms"),
        "pending_events": (agent or {}).get("pending_events"),
        "last_err": (agent or {}).get("last_err") or None,
        "frame_err": (agent or {}).get("frame_err") or None,
    }
    via_modem = route is not None and radio["iface"] is not None and route == radio["iface"]
    ok = bool(radio["registered"]) and agent_s["last_ok_age_s"] is not None and agent_s["last_ok_age_s"] < stale_s
    return {"ts_ms": int(now * 1000), "ok": ok, "route": route, "via_modem": via_modem,
            "radio": radio, "agent": agent_s, "cameras": cameras or {}}


def reason(row: dict) -> str:
    r, a = row["radio"], row["agent"]
    if not r["up"]:
        return "monitor modem (:8088) tidak menjawab"
    if not r["registered"]:
        return f"modem tidak terdaftar ({r['act']})"
    if not a["up"]:
        return "agen (:8098) tidak menjawab"
    return f"agen tidak kontak server {a['last_ok_age_s']} s ({a['last_err']})"


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--log-dir", default=os.path.expanduser("~/gmf-forklift/logs"))
    ap.add_argument("--interval-s", type=float, default=5.0)
    ap.add_argument("--stale-s", type=float, default=15.0, help="agen dianggap putus jika kontak terakhir > ini")
    ap.add_argument("--status-url", default="http://127.0.0.1:8088/api/status")
    ap.add_argument("--agent-url", default="http://127.0.0.1:8098/")
    ap.add_argument("--timeout-s", type=float, default=3.0)
    ap.add_argument("--camera", action="append", default=None, metavar="PERAN=NAMA",
                    help="kamera yang diharapkan, potongan nama perangkat V4L2 "
                         "(default: cabin=imx219 front=C930e); boleh diulang")
    args = ap.parse_args(argv)
    expected = dict(c.split("=", 1) for c in (args.camera or ["cabin=imx219", "front=C930e"]))

    os.makedirs(args.log_dir, exist_ok=True)
    prev_ok = None
    prev_cams: dict[str, bool] = {}
    down_since = None
    while True:
        t0 = time.time()
        cams = camera_status(video_devices(), expected)
        row = summarize(fetch_json(args.status_url, args.timeout_s), fetch_json(args.agent_url, args.timeout_s),
                        default_route_iface(), time.time(), args.stale_s, cams)
        for role, present in cams.items():
            if prev_cams.get(role) != present:
                print(f"{time.strftime('%Y-%m-%d %H:%M:%S')} KAMERA {role} ({expected[role]}): "
                      f"{'terpasang' if present else 'TERLEPAS / tidak terdeteksi'}", flush=True)
        prev_cams = cams
        path = os.path.join(args.log_dir, time.strftime("netwatch-%Y%m%d.jsonl"))
        with open(path, "a", encoding="utf-8") as f:
            f.write(json.dumps(row, separators=(",", ":")) + "\n")
        if row["ok"] != prev_ok:
            stamp = time.strftime("%Y-%m-%d %H:%M:%S")
            if row["ok"]:
                dur = f" setelah putus {t0 - down_since:.0f} s" if down_since else ""
                print(f"{stamp} TERSAMBUNG{dur} | {row['radio']['act']} RSRP {row['radio']['rsrp']} "
                      f"SINR {row['radio']['sinr']} | rtt {row['agent']['rtt_ms']} ms | via {row['route']}", flush=True)
                down_since = None
            else:
                down_since = t0
                print(f"{stamp} PUTUS | {reason(row)}", flush=True)
            prev_ok = row["ok"]
        time.sleep(max(0.0, args.interval_s - (time.time() - t0)))


if __name__ == "__main__":
    raise SystemExit(main())
