"""Entry point runtime kabin.

Contoh:
  python -m src.main --source webcam
  python -m src.main --source webcam --simulate-nir
  python -m src.main --source path/ke/video.mp4 --no-show
  python -m src.main --source webcam --fps-only          # L1: hanya buka kamera + FPS
  python -m src.main --source http://192.168.1.5:8080/video --camera-profile imx219_77ir   # HP + simulasi kamera
  python -m src.main --source http://192.168.1.5:8080/video --camera-profile imx219_77ir+forklift --phone-hfov 68
Hotkey jendela: q = keluar, c = kalibrasi ulang, p = pakai baseline populasi, d = panel debug (angka mentah).
"""
from __future__ import annotations

import argparse
import time

import cv2

from src.capture import make_source, parse_source_arg
from src.capture.camera_sim import SimulatedSource, setup_camera_sim, window_title
from src.config import Config, deep_merge
from src.preprocess import resize_max_side


def parse_args(argv=None):
    ap = argparse.ArgumentParser(description="Forklift Safety AI — kamera kabin")
    ap.add_argument("--source", default=None, help="webcam | webcam:N | file.mp4 | rtsp://... (default: runtime.yaml)")
    ap.add_argument("--runtime", default="configs/runtime.yaml")
    ap.add_argument("--thresholds", default="configs/thresholds.yaml")
    ap.add_argument("--events", default="configs/events.yaml")
    ap.add_argument("--simulate-nir", action="store_true", help="grayscale 3 channel + CLAHE")
    ap.add_argument("--no-show", action="store_true")
    ap.add_argument("--no-yolo", action="store_true", help="matikan YOLO kabin (D-06/D-10)")
    ap.add_argument("--no-beep", action="store_true")
    ap.add_argument("--no-record", action="store_true", help="jangan simpan snapshot/klip")
    ap.add_argument("--fps-only", action="store_true", help="L1: tampilkan kamera + FPS saja")
    ap.add_argument("--max-frames", type=int, default=0)
    ap.add_argument("--fast", action="store_true", help="file video: proses secepat mungkin (bukan realtime)")
    ap.add_argument("--camera-profile", default=None,
                    help="simulasikan kamera target dari configs/camera_profiles.yaml (mis. imx219_77ir, "
                         "imx219_77ir+forklift); 'none' = matikan. Default: runtime.yaml cabin.camera_profile")
    ap.add_argument("--phone-hfov", type=float, default=None, help="HFOV kamera HP/webcam sumber (derajat)")
    return ap.parse_args(argv)


def show_image(win: str, img, max_side: int | None):
    cv2.imshow(win, resize_max_side(img, max_side) if max_side else img)


def run_fps_only(src, show: bool, max_frames: int, win: str = "Forklift Safety AI - FPS",
                 max_side: int | None = None):
    n, t0, fps = 0, time.monotonic(), 0.0
    for f in src:
        n += 1
        el = time.monotonic() - t0
        if el >= 1.0:
            fps, n, t0 = n / el, 0, time.monotonic()
            print(f"FPS {fps:.1f}  frame {f.image.shape[1]}x{f.image.shape[0]}")
        if show:
            img = f.image.copy()
            cv2.putText(img, f"FPS {fps:.1f}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            show_image(win, img, max_side)
            if cv2.waitKey(1) & 0xFF == ord("q"):
                break
        if max_frames and f.index + 1 >= max_frames:
            break


def main(argv=None):
    a = parse_args(argv)
    cfg = Config.load(a.runtime, a.thresholds, a.events)
    rt = cfg.runtime
    if a.source:
        rt["cabin"]["source"] = deep_merge(rt["cabin"]["source"], parse_source_arg(a.source))
    if a.fast:
        rt["cabin"]["source"]["realtime"] = False
    if a.simulate_nir:
        rt["cabin"]["simulate_nir"] = True
    if a.no_beep:
        rt["alert"]["type"] = "none"
    show = rt.get("ui", {}).get("show", True) and not a.no_show

    sim = setup_camera_sim(rt, "cabin", a.camera_profile, a.phone_hfov, rt["cabin"]["source"])
    if sim is not None and sim.nir:
        # kamera target NIR mono: CLAHE dilakukan pipeline (sama seperti nanti di Jetson), bukan oleh simulasi
        rt["cabin"]["simulate_nir"] = True
    win = window_title(rt.get("ui", {}).get("window", "cabin"), sim)
    max_side = rt.get("ui", {}).get("display_max_side")

    src = make_source(rt["cabin"]["source"])
    if sim is not None:
        src = SimulatedSource(src, sim)
    if a.fps_only:
        with src:
            run_fps_only(src, show, a.max_frames, win, max_side)
        return

    from src.cabin.pipeline import CabinPipeline
    from src.io.alert import make_alerter
    from src.io.clips import ClipRecorder
    from src.io.sinks import make_sinks
    from src.ui.overlay import draw_cabin

    pipe = CabinPipeline(rt, cfg.thresholds, cfg.events, use_yolo=False if a.no_yolo else None)
    alerter = make_alerter(rt.get("alert", {}))
    sinks = make_sinks(rt.get("sink", {}), rt.get("device_id", "laptop-dev"))
    rec_cfg = rt.get("recording", {})
    recorder = ClipRecorder(rec_cfg, rt["sink"]["file"].get("dir", "events")) \
        if rec_cfg.get("enabled", True) and not a.no_record else None

    fps, n, t0 = 0.0, 0, time.monotonic()
    last = None
    debug = False
    try:
        with src:
            for frame in src:
                st = pipe.process(frame)
                last = st
                for act in st.actions:
                    if act.kind == "buzz":
                        alerter.buzz(act.payload["level"], act.payload["code"], act.payload["first"])
                    elif act.kind == "event":
                        ev = act.payload
                        if ev["phase"] == "start" and recorder is not None:
                            ev.update(recorder.on_event_start(ev, st.image))
                        sinks.send(ev)
                        print(f"[event] {ev['phase']:5s} {ev['code']} {ev['level']} id={ev['event_id']} "
                              f"t={ev.get('t_video_s')}")
                n += 1
                el = time.monotonic() - t0
                if el >= 1.0:
                    fps, n, t0 = n / el, 0, time.monotonic()
                vis = draw_cabin(st, pipe, fps, pipe.decision.open, debug) if (show or recorder) else None
                if recorder is not None:
                    recorder.push(vis, frame.ts_ms)
                if show:
                    show_image(win, vis, max_side)
                    k = cv2.waitKey(1) & 0xFF
                    if k == ord("q"):
                        break
                    if k == ord("c"):
                        pipe.recalibrate()
                    if k == ord("p"):
                        pipe.calibration.set_population()
                    if k == ord("d"):
                        debug = not debug
                if a.max_frames and frame.index + 1 >= a.max_frames:
                    break
    finally:
        if last is not None:
            for act in pipe.finish(last.frame.ts_ms, last.frame.video_t_s):
                sinks.send(act.payload)
        if recorder is not None:
            recorder.flush()
        sinks.close()
        alerter.close()
        if show:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
