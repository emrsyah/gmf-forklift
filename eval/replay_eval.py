"""Replay harness (L6, Tech Spec bag. 9).

Memutar setiap video di truth.csv lewat pipeline kabin YANG SAMA dengan live (timestamp asli dari video),
mengumpulkan event, lalu mencocokkan dengan kebenaran:

  hits = event start dengan start_s <= t_video <= end_s + match_tail_s
  terdeteksi = ada hit (K3); kode/level benar dinilai dari hit PERTAMA (K4 bagian AI)
  waktu deteksi = t_video hit pertama - start_s

Laporan: tabel per skenario (format Lembar A), skor K3, % kode+level benar, waktu deteksi per kode,
confusion matrix antar kode, alarm palsu per jam pada video normal (baris expected_code = NONE).

Contoh:
  python -m eval.replay_eval --truth eval/scenarios/truth.csv
  python -m eval.replay_eval --truth eval/scenarios/truth.csv --simulate-nir --out eval/reports/run1
  python -m eval.replay_eval --truth ... --set D05_distraction.pitch_only_extra_hold_s=1.0
"""
from __future__ import annotations

import argparse
import csv
import json
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime
from pathlib import Path
from typing import Callable

import yaml

from src.config import REPO_ROOT, Config, load_yaml, set_dotted

NONE_CODES = {"", "NONE", "NORMAL", "-"}


@dataclass
class TruthRow:
    video: str
    start_s: float
    end_s: float
    expected_code: str
    expected_level: str
    peraga: str = ""
    cahaya: str = ""
    scenario: str = ""

    @property
    def is_normal(self) -> bool:
        return self.expected_code.strip().upper() in NONE_CODES


@dataclass
class RowResult:
    row: TruthRow
    detected: bool
    code_ok: bool
    level_ok: bool
    t_detect: float | None
    got_code: str | None
    got_level: str | None
    n_hits: int


@dataclass
class VideoRun:
    video: str
    duration_s: float
    events: list[dict] = field(default_factory=list)
    wall_s: float = 0.0
    frames: int = 0


def norm_code(c: str) -> str:
    return c.strip().upper().replace("-", "")


def load_truth(path: str | Path) -> list[TruthRow]:
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        for r in csv.DictReader(f):
            rows.append(TruthRow(
                video=r["video"].strip(), start_s=float(r["start_s"]), end_s=float(r["end_s"]),
                expected_code=norm_code(r.get("expected_code", "")),
                expected_level=r.get("expected_level", "").strip(),
                peraga=r.get("peraga", "").strip(), cahaya=r.get("cahaya", "").strip(),
                scenario=str(r.get("scenario", "") or "").strip()))
    return rows


# ---------------------------------------------------------------------------- menjalankan pipeline
def run_video(video_path: Path, cfg: Config, mode: str = "fast", use_yolo: bool | None = None) -> VideoRun:
    """Putar satu video lewat CabinPipeline. mode: fast (deterministik) | realtime (buang frame jika telat, = live)."""
    from src.cabin.pipeline import CabinPipeline
    from src.capture import VideoFileSource

    pipe = CabinPipeline(cfg.runtime, cfg.thresholds, cfg.events, use_yolo=use_yolo)
    src = VideoFileSource(video_path, realtime=False)
    run = VideoRun(str(video_path), 0.0)
    t0 = time.monotonic()
    last = None
    lag_budget = 0.0
    with src:
        for frame in src:
            if mode == "realtime":
                # emulasi kamera live: jika pemrosesan tertinggal, frame yang lewat dibuang
                behind = (time.monotonic() - t0) - frame.video_t_s
                if behind > 1.0 / src.fps_hint:
                    continue
            st = pipe.process(frame)
            last = st
            run.frames += 1
            for a in st.actions:
                if a.kind == "event":
                    run.events.append(a.payload)
    if last is not None:
        for a in pipe.finish(last.frame.ts_ms, last.frame.video_t_s):
            run.events.append(a.payload)
        run.duration_s = last.frame.video_t_s
    run.wall_s = time.monotonic() - t0
    return run


# ---------------------------------------------------------------------------- penilaian
def score(rows: list[TruthRow], runs: dict[str, VideoRun], tail_s: float = 2.0) -> dict:
    results: list[RowResult] = []
    claimed: set[str] = set()
    for row in rows:
        if row.is_normal:
            continue
        run = runs.get(row.video)
        evs = [e for e in (run.events if run else []) if e["phase"] == "start"
               and row.start_s <= e["t_video_s"] <= row.end_s + tail_s]
        evs.sort(key=lambda e: e["t_video_s"])
        for e in evs:
            claimed.add(e["event_id"])
        if evs:
            h = evs[0]
            results.append(RowResult(row, True, norm_code(h["code"]) == row.expected_code,
                                     h["level"] == row.expected_level, h["t_video_s"] - row.start_s,
                                     norm_code(h["code"]), h["level"], len(evs)))
        else:
            results.append(RowResult(row, False, False, False, None, None, None, 0))

    # alarm palsu: event start di video normal, atau di luar jendela skenario mana pun
    scen_videos = {r.video for r in rows if not r.is_normal}
    fa_events = []
    normal_hours = 0.0
    for r in rows:
        if not r.is_normal or r.video not in runs:
            continue
        run = runs[r.video]
        if r.end_s > r.start_s:            # segmen normal di dalam video
            lo, hi = r.start_s, r.end_s
        elif r.video not in scen_videos:   # seluruh video normal (end_s = 0)
            lo, hi = 0.0, run.duration_s
        else:
            continue
        normal_hours += (hi - lo) / 3600
        fa_events += [e for e in run.events if e["phase"] == "start" and lo <= e["t_video_s"] <= hi]
        for e in fa_events:
            claimed.add(e["event_id"])
    stray = [e for v, run in runs.items() if v in scen_videos for e in run.events
             if e["phase"] == "start" and e["event_id"] not in claimed]

    n = len(results)
    det = sum(r.detected for r in results)
    ok = sum(r.code_ok and r.level_ok for r in results)
    det_ok = [r for r in results if r.detected]
    by_code = defaultdict(list)
    for r in det_ok:
        by_code[r.row.expected_code].append(r.t_detect)
    confusion = Counter((r.row.expected_code, r.got_code or "MISS") for r in results)
    three_s = [r for r in det_ok if r.row.expected_code in ("D02", "D05", "D06")
               and (r.row.end_s - r.row.start_s) <= 3.5]
    return {
        "results": results,
        "n": n, "detected": det, "code_level_ok": ok,
        "k3": det / n if n else 0.0,
        "code_level_ok_of_detected": (sum(r.code_ok and r.level_ok for r in det_ok) / len(det_ok)) if det_ok else 0.0,
        "detect_time_by_code": {c: (sum(v) / len(v), max(v)) for c, v in by_code.items()},
        "three_s_max_detect": max((r.t_detect for r in three_s), default=None),
        "confusion": confusion,
        "false_alarms": fa_events, "normal_hours": normal_hours,
        "fa_per_hour": (len(fa_events) / normal_hours) if normal_hours > 0 else None,
        "stray_events": stray,
    }


# ---------------------------------------------------------------------------- laporan
def _fmt(v, nd=2):
    return "-" if v is None else (f"{v:.{nd}f}" if isinstance(v, float) else str(v))


def render_markdown(s: dict, meta: dict, gate: dict) -> str:
    L = []
    L.append(f"# Laporan replay — {meta['when']}\n")
    L.append("> Hasil replay pada data di truth.csv. Bukan klaim lulus kecuali datanya rekaman gladi "
             "Lembar A dengan peraga luar tim (CLAUDE.md L7).\n")
    L.append(f"- truth: `{meta['truth']}`  ·  mode: `{meta['mode']}`  ·  simulate_nir: `{meta['simulate_nir']}`  "
             f"·  YOLO kabin: `{meta['yolo']}`")
    if meta.get("overrides"):
        L.append(f"- override: `{meta['overrides']}`")
    if meta.get("ignored_codes"):
        L.append(f"- **event dibuang sebelum penilaian:** kode `{meta['ignored_codes']}` "
                 f"({meta.get('ignored_events', 0)} event) — truth tidak mencatat kondisi tersebut")
    L.append("")
    L.append("## Ringkasan\n")
    L.append("| Metrik | Nilai | Gerbang |")
    L.append("|---|---|---|")
    k3n = f"{s['detected']}/{s['n']}"
    L.append(f"| K3 terdeteksi | {k3n} ({s['k3']*100:.0f}%) | ≥ 80% wajib; gerbang Fase 1 ≥ 18/20 |")
    L.append(f"| Kode + level benar (dari yang terdeteksi) | {s['code_level_ok_of_detected']*100:.0f}% | 100% |")
    L.append(f"| Waktu deteksi maks. skenario 3 s (D-02/05/06) | {_fmt(s['three_s_max_detect'])} s | ≤ 2,0 s |")
    fa = s["fa_per_hour"]
    L.append(f"| Alarm palsu per jam (video normal, {s['normal_hours']:.2f} jam) | "
             f"{'-' if fa is None else f'{fa:.2f}'} | < 1 |")
    L.append(f"| Event di luar jendela skenario | {len(s['stray_events'])} | — |")
    L.append("")
    L.append("### Status gerbang\n")
    for k, v in gate.items():
        L.append(f"- {'✅' if v is True else ('❌' if v is False else '⚪')} {k}")
    L.append("")
    L.append("## Tabel skenario (format Lembar A)\n")
    L.append("| No | Video | Peraga | Cahaya | Diharapkan | Didapat | Terdeteksi | Kode | Level | t deteksi (s) | #hit |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for r in s["results"]:
        row = r.row
        L.append(f"| {row.scenario or '-'} | {Path(row.video).name} | {row.peraga} | {row.cahaya} | "
                 f"{row.expected_code} {row.expected_level} | {r.got_code or '-'} {r.got_level or ''} | "
                 f"{'✅' if r.detected else '❌'} | {'✅' if r.code_ok else '❌'} | {'✅' if r.level_ok else '❌'} | "
                 f"{_fmt(r.t_detect)} | {r.n_hits} |")
    L.append("")
    L.append("## Waktu deteksi per kode\n")
    L.append("| Kode | Rata-rata (s) | Maks (s) |")
    L.append("|---|---|---|")
    for c, (avg, mx) in sorted(s["detect_time_by_code"].items()):
        L.append(f"| {c} | {avg:.2f} | {mx:.2f} |")
    L.append("")
    L.append("## Confusion matrix (diharapkan → event pertama)\n")
    exp = sorted({e for e, _ in s["confusion"]})
    got = sorted({g for _, g in s["confusion"]})
    L.append("| diharapkan \\ didapat | " + " | ".join(got) + " |")
    L.append("|---|" + "---|" * len(got))
    for e in exp:
        L.append(f"| {e} | " + " | ".join(str(s["confusion"].get((e, g), "")) for g in got) + " |")
    L.append("")
    if s["false_alarms"]:
        L.append("## Alarm palsu (video normal)\n")
        cnt = Counter(e["code"] for e in s["false_alarms"])
        L.append(", ".join(f"{c}: {n}" for c, n in cnt.most_common()))
        L.append("")
    if s["stray_events"]:
        L.append("## Event di luar jendela skenario\n")
        for e in s["stray_events"][:50]:
            L.append(f"- t={e['t_video_s']:.2f}s {e['code']} {e['level']}")
        L.append("")
    return "\n".join(L)


def gate_status(s: dict) -> dict:
    return {
        "K3 ≥ 18/20 (gerbang Fase 1; berlaku jika data = 20 skenario peraga luar tim)":
            (s["detected"] >= 18) if s["n"] >= 20 else None,
        "Kode + level benar 100% untuk yang terdeteksi": s["code_level_ok_of_detected"] == 1.0 if s["detected"] else None,
        "Waktu deteksi skenario 3 s ≤ 2,0 s":
            (s["three_s_max_detect"] <= 2.0) if s["three_s_max_detect"] is not None else None,
        "Alarm palsu < 1 per jam": (s["fa_per_hour"] < 1.0) if s["fa_per_hour"] is not None else None,
    }


def write_report(s: dict, meta: dict, out_dir: Path) -> Path:
    out_dir.mkdir(parents=True, exist_ok=True)
    gate = gate_status(s)
    md = render_markdown(s, meta, gate)
    (out_dir / "report.md").write_text(md, encoding="utf-8")
    js = {k: v for k, v in s.items() if k not in ("results", "confusion", "false_alarms", "stray_events")}
    js["rows"] = [{**r.row.__dict__, "detected": r.detected, "code_ok": r.code_ok, "level_ok": r.level_ok,
                   "t_detect": r.t_detect, "got_code": r.got_code, "got_level": r.got_level} for r in s["results"]]
    js["confusion"] = {f"{a}->{b}": n for (a, b), n in s["confusion"].items()}
    js["false_alarm_codes"] = Counter(e["code"] for e in s["false_alarms"])
    js["gate"] = gate
    js["meta"] = meta
    (out_dir / "report.json").write_text(json.dumps(js, indent=2, ensure_ascii=False, default=str), encoding="utf-8")
    try:
        import html
        (out_dir / "report.html").write_text(
            "<!doctype html><meta charset='utf-8'><title>Replay report</title>"
            "<pre style='font:14px/1.4 ui-monospace,monospace;white-space:pre-wrap'>" + html.escape(md) + "</pre>",
            encoding="utf-8")
    except Exception:
        pass
    return out_dir / "report.md"


def drop_codes(runs: dict[str, VideoRun], codes: set[str]) -> int:
    """Buang event kode tertentu sebelum penilaian (mis. D10 pada rekaman simulasi yang sengaja tanpa APD:
    alert D10 di sana BENAR, tetapi truth.csv tidak mencatat kondisi APD). Kembalikan jumlah event dibuang."""
    n = 0
    for run in runs.values():
        keep = [e for e in run.events if norm_code(e["code"]) not in codes]
        n += len(run.events) - len(keep)
        run.events = keep
    return n


def evaluate(truth: Path, cfg: Config, mode: str = "fast", use_yolo: bool | None = None,
             runner: Callable[[Path, Config], VideoRun] | None = None, tail_s: float | None = None,
             ignore_codes: set[str] | None = None) -> dict:
    rows = load_truth(truth)
    if tail_s is None:
        tail_s = load_yaml("configs/scenarios.yaml").get("match_tail_s", 2.0)
    runs: dict[str, VideoRun] = {}
    for v in dict.fromkeys(r.video for r in rows):
        p = Path(v)
        if not p.is_absolute():
            p = (truth.parent / v) if (truth.parent / v).exists() else (REPO_ROOT / v)
        print(f"[replay] {p.name} ...", flush=True)
        runs[v] = runner(p, cfg) if runner else run_video(p, cfg, mode, use_yolo)
        print(f"[replay]   {len(runs[v].events)} event, {runs[v].frames} frame, {runs[v].wall_s:.1f} s", flush=True)
    dropped = drop_codes(runs, ignore_codes) if ignore_codes else 0
    s = score(rows, runs, tail_s)
    s["ignored_events"] = dropped
    return s


def main(argv=None):
    ap = argparse.ArgumentParser(description="Replay harness Lembar A")
    ap.add_argument("--truth", default="eval/scenarios/truth.csv")
    ap.add_argument("--out", default=None, help="folder laporan (default eval/reports/<waktu>)")
    ap.add_argument("--mode", choices=["fast", "realtime"], default="fast")
    ap.add_argument("--simulate-nir", action="store_true")
    ap.add_argument("--no-yolo", action="store_true")
    ap.add_argument("--set", action="append", default=[],
                    help="override threshold, mis. D05_distraction.pitch_only_extra_hold_s=1.0")
    ap.add_argument("--set-runtime", action="append", default=[],
                    help="override runtime.yaml, mis. cabin.face.crop.enabled=true")
    ap.add_argument("--ignore-codes", default="",
                    help="kode yang event-nya dibuang sebelum penilaian, dipisah koma (mis. D10 untuk rekaman "
                         "simulasi tanpa properti APD). Dicatat di laporan.")
    a = ap.parse_args(argv)

    cfg = Config.load()
    if a.simulate_nir:
        cfg.runtime["cabin"]["simulate_nir"] = True
    for kv in a.set:
        k, v = kv.split("=", 1)
        set_dotted(cfg.thresholds, k, yaml.safe_load(v))
    for kv in a.set_runtime:
        k, v = kv.split("=", 1)
        set_dotted(cfg.runtime, k, yaml.safe_load(v))
    truth = Path(a.truth)
    ignore = {norm_code(c) for c in a.ignore_codes.split(",") if c.strip()}
    s = evaluate(truth, cfg, a.mode, use_yolo=False if a.no_yolo else None, ignore_codes=ignore)
    when = datetime.now().strftime("%Y%m%d-%H%M%S")
    out = Path(a.out) if a.out else REPO_ROOT / "eval" / "reports" / when
    meta = {"when": when, "truth": str(truth), "mode": a.mode, "simulate_nir": a.simulate_nir,
            "yolo": not a.no_yolo, "overrides": a.set + a.set_runtime,
            "ignored_codes": sorted(ignore), "ignored_events": s.get("ignored_events", 0)}
    p = write_report(s, meta, out)
    print(p.read_text(encoding="utf-8"))
    print(f"\nLaporan: {p}")


if __name__ == "__main__":
    main()
