"""Lapisan jaringan gateway (asyncio + websockets) + server HTTP media (snapshot/klip).

  python -m src.gateway.server                      # pakai runtime.yaml bagian `gateway`
  python -m src.gateway.server --overlay configs/jetson.yaml

Dashboard (Pengaturan -> endpoint Jetson FL-01): ws://<ip-jetson>:8787
Produsen lokal (proses CV, GPS/zona, UPS):       ws://127.0.0.1:8788
Bukti (snapshot/klip, hanya .jpg/.mp4):          http://<ip-jetson>:8789/media/<tanggal>/<event_id>.mp4
"""
from __future__ import annotations

import argparse
import asyncio
import json
import threading

from websockets.asyncio.server import broadcast, serve

from src.config import Config, deep_merge, load_yaml

from .core import Gateway, system_stats
from .media import start_media_server


class GatewayServer:
    def __init__(self, gw: Gateway, cfg: dict):
        self.gw = gw
        self.c = cfg
        self.dash: dict = {}                      # koneksi -> host yang dipakai dashboard (untuk URL media)
        self.max_bytes = int(cfg.get("max_message_kb", 512)) * 1024
        self.frame_backlog = int(cfg.get("frame_drop_backlog_kb", 16)) * 1024
        self.frames_dropped = 0
        self.media_httpd = None

    @staticmethod
    def _backlog(ws) -> int:
        try:
            return ws.transport.get_write_buffer_size()
        except Exception:
            return 0

    def _send_all(self, msgs: list[dict]):
        """Kirim ke semua dashboard. Dirender per host (URL media memakai host yang dipakai klien itu).
        broadcast() tidak punya backpressure: frame live dibuang untuk klien yang buffer kirimnya masih
        penuh (link 5G lambat), supaya event alarm tidak antre di belakang gambar (K5)."""
        if not self.dash or not msgs:
            return
        for m in msgs:
            conns = list(self.dash.items())
            if m.get("type") == "frame":
                ok = [ws for ws, _ in conns if self._backlog(ws) <= self.frame_backlog]
                self.frames_dropped += len(conns) - len(ok)
                if ok:   # frame tidak punya URL media: serialisasi sekali untuk semua klien
                    broadcast(ok, self.gw.render(m, None))
                continue
            groups: dict = {}
            for ws, host in conns:
                groups.setdefault(host, []).append(ws)
            for host, group in groups.items():
                broadcast(group, self.gw.render(m, host))

    async def dashboard_handler(self, ws):
        host = ws.request.headers.get("Host") if ws.request is not None else None
        self.dash[ws] = host
        self.gw.log("dashboard_connect", {"peer": str(ws.remote_address), "host": host, "clients": len(self.dash)})
        try:
            async for raw in ws:
                try:
                    msg = json.loads(raw)
                except (json.JSONDecodeError, TypeError):
                    continue
                if not isinstance(msg, dict):
                    continue
                for reply in await asyncio.to_thread(self.gw.on_dashboard, msg):
                    await ws.send(self.gw.render(reply, host))
        finally:
            self.dash.pop(ws, None)
            self.gw.log("dashboard_disconnect", {"peer": str(ws.remote_address), "clients": len(self.dash)})

    async def producer_handler(self, ws):
        async for raw in ws:
            try:
                msg = json.loads(raw)
            except (json.JSONDecodeError, TypeError):
                continue
            if not isinstance(msg, dict):
                continue
            if msg.get("type") in ("event", "media"):
                # fsync seq + events.jsonl di thread: event loop (frame, telemetri, klien lain) tidak tertahan
                out = await asyncio.to_thread(self.gw.ingest, msg)
            else:
                out = self.gw.ingest(msg)
            self._send_all(out)
            if msg.get("type") == "event" and msg.get("ack_id") is not None:
                await ws.send(json.dumps({"type": "ack", "ack_id": msg["ack_id"]}))

    async def telemetry_loop(self):
        period = float(self.c.get("telemetry_s", 2))
        while True:
            await asyncio.sleep(period)
            stats = await asyncio.to_thread(system_stats)
            tel = self.gw.telemetry(stats)
            tel["framesDropped"] = self.frames_dropped
            self._send_all([tel])

    async def run(self, stop: asyncio.Event | None = None):
        d, p = self.c.get("dashboard", {}), self.c.get("producer", {})
        m = self.c.get("media", {}) or {}
        stop = stop or asyncio.Event()
        if self.gw.media_enabled:
            self.media_httpd = start_media_server(self.gw.media_root, m.get("host", "0.0.0.0"), self.gw.media_port)
        async with serve(self.dashboard_handler, d.get("host", "0.0.0.0"), int(d.get("port", 8787)),
                         max_size=self.max_bytes), \
                serve(self.producer_handler, p.get("host", "127.0.0.1"), int(p.get("port", 8788)),
                      max_size=self.max_bytes):
            print(f"[gateway] dashboard ws://{d.get('host', '0.0.0.0')}:{d.get('port', 8787)} | "
                  f"produsen ws://{p.get('host', '127.0.0.1')}:{p.get('port', 8788)} | "
                  f"media http://{m.get('host', '0.0.0.0')}:{self.gw.media_port} "
                  f"{'(aktif)' if self.media_httpd else '(mati)'} | "
                  f"seq terakhir {self.gw.seq} | buffer {len(self.gw.buffer)}", flush=True)
            tel = asyncio.create_task(self.telemetry_loop())
            try:
                await stop.wait()
            finally:
                tel.cancel()
                if self.media_httpd is not None:
                    threading.Thread(target=self.media_httpd.shutdown, daemon=True).start()


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
