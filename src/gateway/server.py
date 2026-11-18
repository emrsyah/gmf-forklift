"""Lapisan jaringan gateway (asyncio + websockets).

  python -m src.gateway.server                      # pakai runtime.yaml bagian `gateway`
  python -m src.gateway.server --overlay configs/jetson.yaml

Dashboard (Pengaturan -> endpoint Jetson FL-01): ws://<ip-jetson>:8787
Produsen lokal (proses CV, GPS/zona, UPS):       ws://127.0.0.1:8788
"""
from __future__ import annotations

import argparse
import asyncio
import json

from websockets.asyncio.server import broadcast, serve

from src.config import Config, deep_merge, load_yaml

from .core import Gateway, system_stats


class GatewayServer:
    def __init__(self, gw: Gateway, cfg: dict):
        self.gw = gw
        self.c = cfg
        self.dash = set()
        self.max_bytes = int(cfg.get("max_message_kb", 512)) * 1024

    def _broadcast(self, texts: list[str]):
        for t in texts:
            broadcast(self.dash, t)

    async def dashboard_handler(self, ws):
        self.dash.add(ws)
        self.gw.log("dashboard_connect", {"peer": str(ws.remote_address), "clients": len(self.dash)})
        try:
            async for raw in ws:
                try:
                    msg = json.loads(raw)
                except (json.JSONDecodeError, TypeError):
                    continue
                for reply in self.gw.on_dashboard(msg):
                    await ws.send(reply)
        finally:
            self.dash.discard(ws)
            self.gw.log("dashboard_disconnect", {"peer": str(ws.remote_address), "clients": len(self.dash)})

    async def producer_handler(self, ws):
        async for raw in ws:
            try:
                msg = json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                continue
            out = self.gw.ingest(msg)
            self._broadcast(out)
            if msg.get("type") == "event" and msg.get("ack_id") is not None:
                await ws.send(json.dumps({"type": "ack", "ack_id": msg["ack_id"]}))

    async def telemetry_loop(self):
        period = float(self.c.get("telemetry_s", 2))
        while True:
            await asyncio.sleep(period)
            self._broadcast([self.gw.telemetry(system_stats())])

    async def run(self, stop: asyncio.Event | None = None):
        d, p = self.c.get("dashboard", {}), self.c.get("producer", {})
        stop = stop or asyncio.Event()
        async with serve(self.dashboard_handler, d.get("host", "0.0.0.0"), int(d.get("port", 8787)),
                         max_size=self.max_bytes), \
                serve(self.producer_handler, p.get("host", "127.0.0.1"), int(p.get("port", 8788)),
                      max_size=self.max_bytes):
            print(f"[gateway] dashboard ws://{d.get('host', '0.0.0.0')}:{d.get('port', 8787)} | "
                  f"produsen ws://{p.get('host', '127.0.0.1')}:{p.get('port', 8788)} | "
                  f"seq terakhir {self.gw.seq} | buffer {len(self.gw.buffer)}", flush=True)
            tel = asyncio.create_task(self.telemetry_loop())
            try:
                await stop.wait()
            finally:
                tel.cancel()


def main(argv=None):
    ap = argparse.ArgumentParser(description="Gateway Jetson <-> dashboard MINERVA")
    ap.add_argument("--runtime", default="configs/runtime.yaml")
    ap.add_argument("--overlay", action="append", default=[])
    a = ap.parse_args(argv)
    overrides: dict = {}
    for path in a.overlay:
        overrides = deep_merge(overrides, load_yaml(path))
    rt = Config.load(a.runtime, overrides=overrides or None).runtime
    cfg = rt.get("gateway", {})
    gw = Gateway(cfg, cfg.get("forklift_id", "FL-01"))
    try:
        asyncio.run(GatewayServer(gw, cfg).run())
    except KeyboardInterrupt:
        pass


if __name__ == "__main__":
    main()
