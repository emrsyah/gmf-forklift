"""Alarm per jam dari rekaman kabin TANPA truth.csv (gerbang L7: alarm palsu < 1 per jam pada video normal).

Memutar rekaman (continuous_recording, dengan <video>.ts.csv) lewat CabinPipeline yang sama dengan live, lalu
menghitung event `start` per kode, dengan dan tanpa gerbang pengemudi (driver_presence). Tidak menulis event ke
folder events/ dan tidak mengirim apa pun: aman dijalankan di Jetson di samping detektor live (pakai `nice`).

  nice -n 19 python -m tools.replay_alarms --overlay configs/jetson.yaml recordings/cabin/2026-11-25/a.avi ...

Klip yang berisi peragaan Lembar A sengaja tidak dipisahkan di sini: lihat daftar event (waktu video) untuk
memilah peragaan vs alarm palsu.
"""
from __future__ import annotations

import argparse
import collections
import json
import sys
import time
from pathlib import Path

from src.config import Config, load_yaml


def _set(d: dict, dotted: str, raw: str) -> None:
    keys = dotted.split(".")
    for k in keys[:-1]:
        d = d.setdefault(k, {})
    d[keys[-1]] = json.loads(raw) if raw[:1] in "[{0123456789-tfn" else raw


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("videos", nargs="+")
    ap.add_argument("--overlay", action="append", default=[])
    ap.add_argument("--set", action="append", default=[], help="thresholds key=value, mis. D05_distraction.yaw_deg=35")
    ap.add_argument("--json", default=None, help="simpan hasil per event ke file ini")
    a = ap.parse_args(argv)

    from src.cabin.pipeline import CabinPipeline
    from src.capture import VideoFileSource
    from src.decision import DecisionEngine

    if len(a.overlay) > 1:
        ap.error("satu --overlay saja")
    cfg = Config.load(overrides=load_yaml(a.overlay[0]) if a.overlay else None)
    rt, th = cfg.runtime, cfg.thresholds
    rt["cabin"]["yolo"]["async"] = False            # deterministik (replay)
    for kv in a.set:
        k, v = kv.split("=", 1)
        _set(th, k, v)

    gated, ungated = collections.Counter(), collections.Counter()
    rows, total_s = [], 0.0
    for v in a.videos:
        # pipeline baru per video: segmen bisa tidak berurutan waktu (timestamp mundur membuat event tak pernah tutup)
        pipe = CabinPipeline(rt, th, cfg.events)
        base = DecisionEngine(cfg.events, "replay", "replay-ungated")
        t_wall = time.time()
        first = last = None
        n = 0
        with VideoFileSource(v, realtime=False) as src:
            for fr in src:
                st = pipe.process(fr)
                first = fr.ts_ms if first is None else first
                last = fr.ts_ms
                n += 1
                for act in st.actions:
                    if act.kind == "event" and act.payload["phase"] == "start":
                        gated[act.payload["code"]] += 1
                        rows.append({"video": Path(v).name, "t_s": round(fr.video_t_s, 1), "code": act.payload["code"],
                                     "gated": True})
                for act in base.step(fr.ts_ms, pipe.active_ungated):
                    if act.kind == "event" and act.payload["phase"] == "start":
                        ungated[act.payload["code"]] += 1
                        rows.append({"video": Path(v).name, "t_s": round(fr.video_t_s, 1), "code": act.payload["code"],
                                     "gated": False})
        dur = (last - first) / 1000 if first is not None else 0.0
        total_s += dur
        print(f"[replay] {Path(v).name}: {n} frame, {dur:.0f} s video, {time.time() - t_wall:.0f} s proses", flush=True)

    hours = max(total_s / 3600, 1e-9)
    codes = sorted(set(gated) | set(ungated))
    print(f"\ntotal video {total_s / 60:.1f} menit")
    print(f"{'kode':6s} {'tanpa gerbang':>14s} {'per jam':>8s} {'dengan gerbang':>15s} {'per jam':>8s}")
    for c in codes:
        print(f"{c:6s} {ungated[c]:14d} {ungated[c] / hours:8.1f} {gated[c]:15d} {gated[c] / hours:8.1f}")
    print(f"{'total':6s} {sum(ungated.values()):14d} {sum(ungated.values()) / hours:8.1f} "
          f"{sum(gated.values()):15d} {sum(gated.values()) / hours:8.1f}")
    for r in rows:
        if r["gated"]:
            print(f"  event {r['video'][-10:-4]} t={r['t_s']:6.1f}s {r['code']}")
    if a.json:
        Path(a.json).write_text(json.dumps(rows, indent=1))
    return 0


if __name__ == "__main__":
    sys.exit(main())
