"""Entry point runtime kabin.

Contoh:
  python -m src.main --source webcam
  python -m src.main --source webcam --simulate-nir
  python -m src.main --source path/ke/video.mp4 --no-show
  python -m src.main --source webcam --fps-only          # L1: hanya buka kamera + FPS
  python -m src.main --source http://192.168.1.5:8080/video --camera-profile imx219_77ir   # HP + simulasi kamera
  python -m src.main --source http://192.168.1.5:8080/video --camera-profile imx219_77ir+forklift --phone-hfov 68
  python -m src.main --overlay configs/jetson.yaml --stdin-keys --log-fps 5           # Jetson (tanpa keyboard)
Hotkey jendela: q = keluar, c = kalibrasi ulang, p = pakai baseline populasi, d = panel debug (angka mentah).
Dengan --stdin-keys, hotkey yang sama bisa diketik di terminal (+ Enter), mis. lewat `tmux send-keys`.
"""
from __future__ import annotations

import argparse
import time

import cv2

from src.capture import make_source, parse_source_arg
from src.capture.camera_sim import SimulatedSource, setup_camera_sim, window_title
from src.config import Config, deep_merge, load_yaml
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
    ap.add_argument("--overlay", action="append", default=[],
                    help="YAML tambahan yang menimpa config (bagian runtime/thresholds/events), "
                         "mis. configs/jetson.yaml; boleh diulang")
    ap.add_argument("--stdin-keys", action="store_true",
                    help="hotkey juga dibaca dari stdin (q/c/p/d + Enter) — untuk perangkat tanpa keyboard, "
                         "mis. Jetson yang dikendalikan lewat SSH/tmux")
    ap.add_argument("--web", type=int, default=None, metavar="PORT",
                    help="tampilkan overlay di browser: http://IP-PERANGKAT:PORT (+ tombol hotkey). "
                         "Default: runtime.yaml ui.web.port jika ui.web.enabled")
    ap.add_argument("--log-fps", type=float, default=0.0,
                    help="cetak FPS + waktu per tahap ke terminal tiap N detik (0 = mati)")
    return ap.parse_args(argv)


def start_stdin_keys(q):
    """Thread pembaca stdin: setiap karakter yang diketik (diakhiri Enter) masuk antrean hotkey `q`."""
    import sys
    import threading

    def reader():
        for line in sys.stdin:
            for ch in line.strip():
                q.put(ch)

    threading.Thread(target=reader, daemon=True).start()
    return q


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


def handle_actions(actions, alerter, sinks, recorder, image, now_ms=None) -> list[dict]:
    """Jalankan aksi keputusan satu frame. Buzzer DULU, baru event, supaya event `start` membawa
    `ts_buzzer_ms` (waktu buzzer kabin benar-benar diperintahkan; bahan ukur K5 kabin)."""
    clock = now_ms or (lambda: int(time.time() * 1000))
    buzz_ts: dict[str, int] = {}
    for act in actions:
        if act.kind == "buzz":
            alerter.buzz(act.payload["level"], act.payload["code"], act.payload["first"])
            if act.payload["first"]:
                buzz_ts[act.payload["code"]] = clock()
    sent = []
    for act in actions:
        if act.kind != "event":
            continue
        ev = act.payload
        if ev["phase"] == "start":
            if ev["code"] in buzz_ts:
                ev["ts_buzzer_ms"] = buzz_ts[ev["code"]]
            if recorder is not None and image is not None:
                ev.update(recorder.on_event_start(ev, image))
        elif ev["phase"] == "end" and recorder is not None:
            recorder.on_event_end(ev)
        sinks.send(ev)
        sent.append(ev)
        print(f"[event] {ev['phase']:5s} {ev['code']} {ev['level']} id={ev['event_id']} "
              f"t={ev.get('t_video_s')}", flush=True)
    return sent


def main(argv=None):
    a = parse_args(argv)
    overrides: dict = {}
    for path in a.overlay:
        overrides = deep_merge(overrides, load_yaml(path))
    cfg = Config.load(a.runtime, a.thresholds, a.events, overrides=overrides or None)
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
    if recorder is not None:
        recorder.on_done = sinks.media            # info bukti (klip selesai) -> media.jsonl / topik media
    from src.io.heartbeat import Heartbeat
    hb = Heartbeat(rt.get("heartbeat"), rt.get("device_id", "laptop-dev"), rt.get("model_versions"))

    fps, n, t0 = 0.0, 0, time.monotonic()
    last = None
    debug = False
    cal_status = None
    import queue
    keys_in: queue.Queue = queue.Queue()     # hotkey dari stdin dan/atau tombol web
    if a.stdin_keys:
        start_stdin_keys(keys_in)
    web_cfg = rt.get("ui", {}).get("web", {}) or {}
    web_port = a.web if a.web is not None else (web_cfg.get("port") if web_cfg.get("enabled") else None)
    web = None
    if web_port:
        from src.io.web_view import WebView
        web = WebView(web_port, keys_in, title=win, fps=web_cfg.get("fps", 10),
                      max_side=web_cfg.get("max_side", 960), jpeg_quality=web_cfg.get("jpeg_quality", 75))
        print(f"[web] tampilan browser: http://<IP-perangkat>:{web_port}", flush=True)
    t_log, n_log = time.monotonic(), 0
    face_log: list = []
    if show and rt.get("ui", {}).get("fullscreen", False):
        cv2.namedWindow(win, cv2.WINDOW_NORMAL)
        cv2.setWindowProperty(win, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    try:
        with src:
            for frame in src:
                st = pipe.process(frame)
                last = st
                if pipe.calibration.status != cal_status:   # log setiap perubahan status kalibrasi
                    cal_status = pipe.calibration.status
                    print(f"[kalibrasi] t={frame.video_t_s:.1f}s {cal_status}", flush=True)
                handle_actions(st.actions, alerter, sinks, recorder, st.image)
                now_ms = int(time.time() * 1000)
                if hb.due(now_ms):
                    op = pipe.decision.open
                    sinks.heartbeat(hb.build(
                        now_ms, fps=fps, camera_ok=not st.blocked_frame, face_present=st.sig.face_present,
                        calibration=pipe.calibration.status, yolo=pipe.yolo_status,
                        open_events=[op.code] if op is not None else []))
                n += 1
                n_log += 1
                if st.sig.face_present:
                    face_log.append((st.sig.yaw, st.sig.pitch, st.sig.roll, st.sig.openness))
                el = time.monotonic() - t0
                if el >= 1.0:
                    fps, n, t0 = n / el, 0, time.monotonic()
                if a.log_fps and time.monotonic() - t_log >= a.log_fps:
                    dt = time.monotonic() - t_log
                    tm = " ".join(f"{k} {v:.0f}ms" for k, v in st.timings_ms.items())
                    print(f"[fps] {n_log / dt:5.1f} FPS | {tm} | yolo {pipe.yolo_status}", flush=True)
                    if face_log:   # diagnosis jarak jauh: seberapa sering wajah terlihat + pose/mata median
                        import numpy as np
                        y, p, r, o = np.nanmedian(np.array(face_log, dtype=float), axis=0)
                        print(f"[wajah] terlihat {100 * len(face_log) / max(n_log, 1):3.0f}% | yaw {y:+.0f} "
                              f"pitch {p:+.0f} roll {r:+.0f} | openness {o:.2f} | {pipe.calibration.status}",
                              flush=True)
                    else:
                        print(f"[wajah] terlihat 0% | {pipe.calibration.status}", flush=True)
                    t_log, n_log = time.monotonic(), 0
                    face_log = []
                vis = draw_cabin(st, pipe, fps, pipe.decision.open, debug) \
                    if (show or recorder or web) else None
                if recorder is not None:
                    recorder.push(vis, frame.ts_ms)
                if web is not None:
                    web.publish(vis)
                keys = []
                if show:
                    show_image(win, vis, max_side)
                    keys.append(cv2.waitKey(1) & 0xFF)
                while not keys_in.empty():
                    keys.append(ord(keys_in.get()))
                if ord("q") in keys:
                    break
                for k in keys:
                    if k == ord("c"):
                        pipe.recalibrate()
                        print("[kontrol] kalibrasi ulang", flush=True)
                    if k == ord("p"):
                        pipe.calibration.set_population()
                    if k == ord("d"):
                        debug = not debug
                if a.max_frames and frame.index + 1 >= a.max_frames:
                    break
    finally:
        if last is not None:
            handle_actions(pipe.finish(last.frame.ts_ms, last.frame.video_t_s), alerter, sinks, recorder, None)
        if recorder is not None:
            recorder.flush()
        sinks.close()
        alerter.close()
        if web is not None:
            web.close()
        if show:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
