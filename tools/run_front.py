"""Jalankan pipeline kamera depan (L9) pada webcam, file video, atau stream URL (HP).

    python tools/run_front.py --source webcam:1
    python tools/run_front.py --source data/samples/bus_zoom.mp4 --no-show --no-realtime --out out.mp4
    python tools/run_front.py --source http://192.168.1.5:8080/video
    python tools/run_front.py --source http://192.168.1.5:8080/video --camera-profile imx477_m12_75 --phone-hfov 70

Dengan --camera-profile, frame HP disimulasikan menjadi kamera target (configs/camera_profiles.yaml) dan K/distorsi
untuk jarak diturunkan dari profil (bukan dari blok `camera` front_camera.yaml); mount/distance tetap dipakai.

Overlay: kotak + #id, kelas, jarak (m), kecepatan radial (m/s, - = mendekat), panah arah,
garis zona peringatan/bahaya, dan kode KANDIDAT F01..F03 (usulan, belum disepakati HSE GMF).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from collections import defaultdict
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cv2  # noqa: E402

from src.capture import make_source, parse_source_arg  # noqa: E402
from src.capture.camera_sim import SimulatedSource, setup_camera_sim, window_title  # noqa: E402
from src.config import CONFIG_DIR, Config, load_yaml, resolve_path  # noqa: E402
from src.front.pipeline import FrontPipeline  # noqa: E402
from src.preprocess import resize_max_side  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--source", default=None,
                    help="webcam | webcam:N | path video | rtsp/http URL (default: runtime.yaml front.source)")
    ap.add_argument("--runtime", default=str(CONFIG_DIR / "runtime.yaml"))
    ap.add_argument("--thresholds", default=str(CONFIG_DIR / "thresholds.yaml"))
    ap.add_argument("--events", default=str(CONFIG_DIR / "events.yaml"))
    ap.add_argument("--front-camera", default=None, help="default: runtime.yaml front.calibration")
    ap.add_argument("--no-show", action="store_true", help="headless, tanpa jendela")
    ap.add_argument("--out", default=None, help="simpan video overlay (.mp4)")
    ap.add_argument("--jsonl", default=None, help="simpan FrontResult per frame (.jsonl)")
    rt = ap.add_mutually_exclusive_group()
    rt.add_argument("--realtime", dest="realtime", action="store_true", default=None,
                    help="file: putar sesuai timestamp asli")
    rt.add_argument("--no-realtime", dest="realtime", action="store_false",
                    help="file: proses secepat mungkin (benchmark FPS); timestamp tetap dari video")
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--warmup", type=int, default=5, help="frame awal yang tidak dihitung di FPS rata-rata")
    ap.add_argument("--camera-profile", default=None,
                    help="simulasikan kamera target (configs/camera_profiles.yaml); 'none' = matikan. "
                         "Default: runtime.yaml front.camera_profile")
    ap.add_argument("--phone-hfov", type=float, default=None, help="HFOV kamera HP/webcam sumber (derajat)")
    args = ap.parse_args(argv)

    cfg = Config.load(args.runtime, args.thresholds, args.events)
    front_rt = cfg.runtime.get("front", {})
    cam_path = resolve_path(args.front_camera or front_rt.get("calibration", "configs/front_camera.yaml"))
    cam_cfg = load_yaml(cam_path)

    src_cfg = dict(front_rt.get("source", {}))
    if args.source:
        src_cfg.update(parse_source_arg(args.source))
    if args.realtime is not None:
        src_cfg["realtime"] = args.realtime

    sim = setup_camera_sim(cfg.runtime, "front", args.camera_profile, args.phone_hfov, src_cfg)
    pipe = FrontPipeline.from_config(cfg, cam_cfg)
    if sim is not None:
        if cam_cfg.get("camera", {}).get("calibrated", False):
            print(f"[front] Blok `camera` di {cam_path} DIABAIKAN selama simulasi profil "
                  f"(frame sudah di-crop/resize); K diturunkan dari profil {sim.name}.")
    elif not pipe.calibrated:
        print(f"[front] PERINGATAN: {cam_path} belum dikalibrasi (camera.calibrated=false); jarak kasar.")

    writer = None
    jsonl = open(args.jsonl, "w", encoding="utf-8") if args.jsonl else None
    win = window_title(cfg.runtime.get("ui", {}).get("window_front", "Forklift Safety AI - Front"), sim)
    max_side = cfg.runtime.get("ui", {}).get("display_max_side")
    camera_set = None   # CropPlan yang K-nya sedang dipakai pipeline
    ids_by_cls: dict[str, set] = defaultdict(set)
    max_persons = 0
    n = 0
    t_start = None
    t_last_print = time.perf_counter()
    n_last_print = 0
    inst_fps = None

    source = make_source(src_cfg)
    if sim is not None:
        source = SimulatedSource(source, sim)
    with source:
        for frame in source:
            t0 = time.perf_counter()
            if sim is not None and sim.plan is not camera_set:
                # plan geometri baru ada setelah frame pertama (ukuran frame HP); dibuat ulang jika ukuran berubah
                block = sim.camera_block()
                pipe.set_camera(cam_cfg, block)
                print(f"[front] K dari profil {sim.name}: fx={block['K'][0][0]:.1f} fy={block['K'][1][1]:.1f} "
                      f"cx={block['K'][0][2]:.1f} cy={block['K'][1][2]:.1f} dist={block['dist']} "
                      f"@ {block['image_size']}" + ("" if block["calibrated"] else
                                                    " (HFOV HP belum diukur -> jarak kasar)"))
                camera_set = sim.plan
            res = pipe.process(frame)
            n += 1
            if n == args.warmup + 1:
                t_start = t0
            for t in res.tracks:
                if t.track_id is not None:
                    ids_by_cls[t.cls].add(t.track_id)
            max_persons = max(max_persons, sum(t.cls == pipe.zones.person_class for t in res.tracks))

            now = time.perf_counter()
            if now - t_last_print >= 1.0:
                inst_fps = (n - n_last_print) / (now - t_last_print)
                desc = ", ".join(f"#{t.track_id}:{t.cls}:{t.distance_m:.1f}m" if t.distance_m is not None
                                 else f"#{t.track_id}:{t.cls}:?" for t in res.tracks)
                print(f"[front] frame {frame.index} t={frame.video_t_s:6.2f}s {inst_fps:5.1f} FPS | "
                      f"{desc} | cand={res.active_codes}")
                t_last_print, n_last_print = now, n

            if jsonl:
                jsonl.write(json.dumps(res.to_dict()) + "\n")
            if not args.no_show or args.out:
                vis = pipe.draw(frame.image, res, inst_fps)
                if args.out:
                    if writer is None:
                        Path(args.out).parent.mkdir(parents=True, exist_ok=True)
                        h, w = vis.shape[:2]
                        writer = cv2.VideoWriter(args.out, cv2.VideoWriter_fourcc(*"mp4v"),
                                                 source.fps_hint, (w, h))
                    writer.write(vis)
                if not args.no_show:
                    cv2.imshow(win, resize_max_side(vis, max_side) if max_side else vis)
                    if (cv2.waitKey(1) & 0xFF) in (ord("q"), 27):
                        break
            if args.max_frames and n >= args.max_frames:
                break

    if writer is not None:
        writer.release()
    if jsonl:
        jsonl.close()
    if not args.no_show:
        cv2.destroyAllWindows()

    measured = n - args.warmup
    avg_fps = measured / (time.perf_counter() - t_start) if t_start and measured > 0 else float("nan")
    print(f"[front] selesai: {n} frame, FPS rata-rata (tanpa {args.warmup} frame warm-up) = {avg_fps:.1f}")
    for k, v in sorted(ids_by_cls.items()):
        print(f"[front] kelas {k}: {len(v)} track id unik -> {sorted(v)}")
    print(f"[front] maks {pipe.zones.person_class} bersamaan dalam satu frame: {max_persons}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
