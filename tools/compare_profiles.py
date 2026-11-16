"""Bandingkan profil kamera pada SATU video mentah: video diputar lewat simulasi tiap profil + pipeline kabin.

    python -m tools.compare_profiles --video eval/scenarios/sesi_P1_raw.mp4
    python -m tools.compare_profiles --video v.mp4 --profiles imx219_77ir,imx708_noir_wide_102 --phone-hfov 104
    python -m tools.compare_profiles --video v.mp4 --max-s 60 --no-yolo

Video harus berisi frame HP MENTAH (rekam dengan tools/recorder.py tanpa profil, atau dengan --save-raw).
Baris "none" = tanpa simulasi (pembanding). Timestamp asli video dipakai (fast, deterministik).

Kolom:
  wajah%   frame dengan wajah terdeteksi        mata%   frame dengan sinyal mata valid (eye_valid)
  open     median openness saat mata valid       yolo    jumlah deteksi per kelas (rata-rata conf)
  event    event start per kode                  sim/pipe ms  waktu simulasi / pipeline per frame
Ini perbandingan RELATIF antar profil pada rekaman yang sama, bukan skor KPI (pakai eval/replay_eval.py).
"""
from __future__ import annotations

import argparse
import copy
import statistics
import time
from collections import Counter, defaultdict
from pathlib import Path

from src.capture import VideoFileSource
from src.capture.camera_sim import (CameraSim, SimulatedSource, already_simulated, list_profiles,
                                    profiles_path_from_runtime)
from src.config import Config


def run_one(video: Path, cfg: Config, sim: CameraSim | None, use_yolo: bool | None, max_s: float) -> dict:
    from src.cabin.pipeline import CabinPipeline

    rt = copy.deepcopy(cfg.runtime)
    if sim is not None and sim.nir:
        rt["cabin"]["simulate_nir"] = True      # sama dengan src/main.py
    pipe = CabinPipeline(rt, cfg.thresholds, cfg.events, use_yolo=use_yolo)
    src = VideoFileSource(video, realtime=False)
    if sim is not None:
        src = SimulatedSource(src, sim)
    n = face = eye = 0
    opens: list[float] = []
    yolo_n: Counter = Counter()
    yolo_conf: dict[str, list[float]] = defaultdict(list)
    events: Counter = Counter()
    pipe_ms = 0.0
    with src:
        for frame in src:
            if max_s and frame.video_t_s > max_s:
                break
            t0 = time.perf_counter()
            st = pipe.process(frame)
            pipe_ms += (time.perf_counter() - t0) * 1000
            n += 1
            face += bool(st.sig.face_present)
            if st.sig.eye_valid:
                eye += 1
                opens.append(float(st.sig.openness))
            if st.yolo is not None and st.yolo.dets:
                for d in st.yolo.dets:
                    yolo_n[d.cls_name] += 1
                    yolo_conf[d.cls_name].append(float(d.conf))
            for a in st.actions:
                if a.kind == "event" and a.payload.get("phase") == "start":
                    events[a.payload["code"]] += 1
    return {
        "profile": sim.name if sim is not None else "none",
        "out": f"{sim.out_w}x{sim.out_h}" if sim is not None else "asli",
        "hfov": (f"{sim.plan.eff_hfov_deg:.0f}" + ("" if sim.plan.fov_ok else "!")) if sim and sim.plan else "-",
        "frames": n,
        "face_pct": 100.0 * face / n if n else 0.0,
        "eye_pct": 100.0 * eye / n if n else 0.0,
        "open_med": statistics.median(opens) if opens else float("nan"),
        "yolo": " ".join(f"{k}:{yolo_n[k]}({statistics.mean(yolo_conf[k]):.2f})" for k in sorted(yolo_n)) or "-",
        "events": " ".join(f"{k}x{v}" for k, v in sorted(events.items())) or "-",
        "sim_ms": sim.timing_ms if sim is not None else 0.0,
        "pipe_ms": pipe_ms / n if n else 0.0,
    }


def render(rows: list[dict]) -> str:
    head = ["profil", "keluaran", "HFOV", "frame", "wajah%", "mata%", "open", "yolo (conf)", "event",
            "sim ms", "pipe ms"]
    lines = ["| " + " | ".join(head) + " |", "|" + "---|" * len(head)]
    for r in rows:
        lines.append(f"| {r['profile']} | {r['out']} | {r['hfov']} | {r['frames']} | {r['face_pct']:.1f} | "
                     f"{r['eye_pct']:.1f} | {r['open_med']:.2f} | {r['yolo']} | {r['events']} | "
                     f"{r['sim_ms']:.1f} | {r['pipe_ms']:.1f} |")
    lines.append("\nHFOV dengan '!' = profil lebih lebar dari kamera HP (tidak bisa ditiru penuh; pakai lensa ultrawide).")
    return "\n".join(lines)


def main(argv=None):
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--video", required=True, help="video frame HP MENTAH")
    ap.add_argument("--profiles", default=None, help="daftar dipisah koma (default: semua profil role=kabin)")
    ap.add_argument("--phone-hfov", type=float, default=None, help="HFOV kamera HP saat merekam (derajat)")
    ap.add_argument("--max-s", type=float, default=0.0, help="hanya proses sekian detik pertama (0 = semua)")
    ap.add_argument("--no-yolo", action="store_true")
    ap.add_argument("--out", default=None, help="simpan tabel Markdown ke file ini")
    a = ap.parse_args(argv)

    video = Path(a.video)
    prev = already_simulated(video)
    if prev:
        raise SystemExit(f"Video ini sudah berisi frame simulasi profil {prev!r}. Rekam ulang dengan --save-raw "
                         f"atau tanpa profil.")
    cfg = Config.load()
    ppath = profiles_path_from_runtime(cfg.runtime)
    names = [s.strip() for s in a.profiles.split(",")] if a.profiles else list_profiles(ppath, role="kabin")
    hfov = a.phone_hfov if a.phone_hfov is not None else (cfg.runtime.get("camera_sim") or {}).get("phone_hfov_deg")
    use_yolo = False if a.no_yolo else None
    rows = []
    for name in [None] + names:
        sim = CameraSim.from_name(name, ppath, hfov) if name else None
        print(f"[compare] {name or 'none'} ...", flush=True)
        rows.append(run_one(video, cfg, sim, use_yolo, a.max_s))
    md = render(rows)
    print("\n" + md)
    if a.out:
        Path(a.out).write_text(f"# Perbandingan profil kamera: {video.name}\n\n{md}\n", encoding="utf-8")
        print(f"\nTabel: {a.out}")


if __name__ == "__main__":
    main()
