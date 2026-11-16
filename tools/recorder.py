"""Rekam skenario Lembar A dari webcam + tandai kebenaran dengan hotkey (L2).

Hasil:
  eval/scenarios/<sesi>.mp4          video mentah (tanpa overlay)
  eval/scenarios/<sesi>.mp4.ts.csv   timestamp asli per frame (dipakai replay)
  eval/scenarios/truth.csv           baris: video,start_s,end_s,expected_code,expected_level,peraga,cahaya,scenario

Hotkey:
  ] / [      skenario berikutnya / sebelumnya (1..20, lihat configs/scenarios.yaml)
  SPASI      mulai / akhiri penandaan skenario terpilih
  n          mulai / akhiri segmen NORMAL (untuk hitung alarm palsu)
  x          batalkan penandaan yang sedang berjalan
  l          ganti label cahaya (normal / redup)
  q / ESC    selesai

Contoh:
  python -m tools.recorder --peraga P1 --session gladi1_P1
  python -m tools.recorder --peraga P2 --cahaya redup --source webcam:1
  python -m tools.recorder --peraga P1 --source http://192.168.1.5:8080/video --camera-profile imx219_77ir
Dengan --camera-profile, layar menampilkan frame simulasi kamera target; yang DIREKAM juga frame simulasi
(replay tanpa profil = hasil kamera target). Pakai --save-raw untuk merekam frame HP mentah (bisa dibandingkan
antar profil dengan tools/compare_profiles.py). Profil yang dipakai dicatat di <video>.camera.json.
Privasi: rekam hanya orang yang setuju direkam; file video tidak di-commit dan tidak diunggah.
"""
from __future__ import annotations

import argparse
import csv
import time
from datetime import datetime
from pathlib import Path

import cv2

from src.capture import make_source, parse_source_arg, sidecar_ts_path
from src.capture.camera_sim import setup_camera_sim, window_title, write_camera_sidecar
from src.config import REPO_ROOT, load_yaml
from src.preprocess import resize_max_side

FIELDS = ["video", "start_s", "end_s", "expected_code", "expected_level", "peraga", "cahaya", "scenario"]


def append_truth(path: Path, row: dict):
    new = not path.exists()
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "a", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=FIELDS)
        if new:
            w.writeheader()
        w.writerow(row)


def load_guide(path: str) -> list[dict]:
    """Langkah panduan dengan waktu mulai/akhir kumulatif. Langkah boleh punya `code`/`level`/`scenario`
    (kebenaran untuk replay; ditulis otomatis ke truth.csv)."""
    steps, t = [], 0.0
    for s in load_yaml(path)["steps"]:
        steps.append({"start": t, "end": t + float(s["dur_s"]), "say": str(s["say"]),
                      "code": s.get("code"), "level": s.get("level", ""), "scenario": s.get("scenario", "")})
        t += float(s["dur_s"])
    return steps


def guide_truth_rows(steps: list[dict], t_done_s: float, video: str, peraga: str, cahaya: str) -> list[dict]:
    """Baris truth.csv untuk langkah ber-kode yang sudah selesai direkam (sampai t_done_s)."""
    return [{"video": video, "start_s": f"{s['start']:.2f}", "end_s": f"{s['end']:.2f}",
             "expected_code": s["code"], "expected_level": s["level"] if s["code"] != "NONE" else "",
             "peraga": peraga, "cahaya": cahaya, "scenario": s["scenario"]}
            for s in steps if s.get("code") and s["end"] <= t_done_s]


def step_beep(freq_hz: int = 1000, dur_ms: int = 150):
    """Bunyi pendek saat ganti langkah (peraga yang sedang pejam mata tidak bisa membaca layar)."""
    import threading

    def _b():
        try:
            import winsound
            winsound.Beep(freq_hz, dur_ms)
        except Exception:
            print("\a", end="", flush=True)
    threading.Thread(target=_b, daemon=True).start()


def guide_at(steps: list[dict], t_s: float) -> tuple[int, dict | None]:
    for i, s in enumerate(steps):
        if s["start"] <= t_s < s["end"]:
            return i, s
    return len(steps), None


def _center_text(img, text, y, scale, color, thick):
    (w, _), _ = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)
    x = max(10, (img.shape[1] - w) // 2)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, (0, 0, 0), thick + 6, cv2.LINE_AA)
    cv2.putText(img, text, (x, y), cv2.FONT_HERSHEY_SIMPLEX, scale, color, thick, cv2.LINE_AA)


def _fit_scale(text: str, max_w: int, scale: float, thick: int) -> float:
    w = cv2.getTextSize(text, cv2.FONT_HERSHEY_SIMPLEX, scale, thick)[0][0]
    return min(scale, scale * max_w / max(1, w))


def draw_guide(vis, steps: list[dict], t_s: float):
    """Panel instruksi di BAWAH layar (wajah tetap terlihat): langkah sekarang besar + hitung mundur + bar,
    lalu langkah berikutnya dan kapan mulainya."""
    i, s = guide_at(steps, t_s)
    h, w = vis.shape[:2]
    k = h / 720.0                                   # skala teks relatif tinggi frame
    top = int(h - 230 * k)
    band = vis[top:]
    band[:] = (band * 0.25).astype(band.dtype)
    if s is None:
        _center_text(vis, "SELESAI - terima kasih", int(h - 110 * k), 1.6 * k, (0, 255, 0), 3)
        return
    left = s["end"] - t_s
    _center_text(vis, f"Langkah {i + 1}/{len(steps)}   (total sisa {steps[-1]['end'] - t_s:4.0f} s)",
                 int(top + 32 * k), 0.75 * k, (200, 200, 200), 2)
    _center_text(vis, s["say"], int(top + 90 * k), _fit_scale(s["say"], w - 40, 1.4 * k, 3), (0, 255, 255), 3)
    _center_text(vis, f"{left:3.0f} detik lagi", int(top + 145 * k), 1.1 * k, (255, 255, 255), 3)
    # bar progres langkah ini
    x0, x1, y = int(w * 0.2), int(w * 0.8), int(top + 162 * k)
    cv2.rectangle(vis, (x0, y), (x1, y + int(10 * k)), (90, 90, 90), -1)
    frac = 1.0 - left / max(1e-6, s["end"] - s["start"])
    cv2.rectangle(vis, (x0, y), (x0 + int((x1 - x0) * frac), y + int(10 * k)), (0, 255, 255), -1)
    if i + 1 < len(steps):
        nxt = steps[i + 1]
        txt = f"BERIKUTNYA (dalam {left:2.0f} s, selama {nxt['end'] - nxt['start']:.0f} s): {nxt['say']}"
        col = (0, 200, 255) if left <= 3 else (180, 180, 180)      # menyala 3 detik sebelum ganti
        _center_text(vis, txt, int(top + 210 * k), _fit_scale(txt, w - 40, 0.75 * k, 2), col, 2)


def main(argv=None):
    ap = argparse.ArgumentParser(description="Recorder skenario Lembar A")
    ap.add_argument("--source", default="webcam")
    ap.add_argument("--peraga", required=True, help="ID peraga, mis. P1 (jangan nama lengkap)")
    ap.add_argument("--cahaya", default="normal")
    ap.add_argument("--session", default=None)
    ap.add_argument("--out-dir", default="eval/scenarios")
    ap.add_argument("--truth", default=None, help="default <out-dir>/truth.csv")
    ap.add_argument("--fps", type=float, default=30.0, help="fps nominal file mp4 (timestamp asli tetap disimpan)")
    ap.add_argument("--calib-s", type=float, default=6.0, help="detik awal: peraga menatap depan (kalibrasi saat replay)")
    ap.add_argument("--guide", default=None,
                    help="YAML panduan langkah (mis. configs/guides/diag_eyes_cover_lap.yaml): instruksi besar di "
                         "layar + hitung mundur, berhenti otomatis di akhir")
    ap.add_argument("--camera-profile", default=None,
                    help="simulasikan kamera target (configs/camera_profiles.yaml); 'none' = matikan. "
                         "Default: runtime.yaml cabin.camera_profile")
    ap.add_argument("--phone-hfov", type=float, default=None, help="HFOV kamera HP/webcam sumber (derajat)")
    ap.add_argument("--save-raw", action="store_true",
                    help="dengan profil: rekam frame HP mentah (layar tetap menampilkan simulasi)")
    a = ap.parse_args(argv)

    scen = load_yaml("configs/scenarios.yaml")["scenarios"]
    guide = load_guide(a.guide) if a.guide else None
    out_dir = REPO_ROOT / a.out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    truth = Path(a.truth) if a.truth else out_dir / "truth.csv"
    session = a.session or f"sesi_{a.peraga}_{datetime.now():%Y%m%d_%H%M%S}"
    vid_path = out_dir / f"{session}.mp4"

    rt = load_yaml("configs/runtime.yaml")
    src_cfg = {**parse_source_arg(a.source), "width": 1280, "height": 720}
    sim = setup_camera_sim(rt, "cabin", a.camera_profile, a.phone_hfov, src_cfg)
    src = make_source(src_cfg)
    writer = None
    ts_f = open(sidecar_ts_path(vid_path), "w", encoding="utf-8")
    ts_f.write("frame,ts_ms\n")
    sel = 0
    cahaya = a.cahaya
    mark: dict | None = None      # {"kind": "scen"/"normal", "start": t_s, "scen": dict}
    t0_ms = None
    t_last = 0.0
    guide_idx = -1
    n_rows = 0
    win = window_title(rt.get("ui", {}).get("window_recorder", "Recorder Lembar A"), sim)
    max_side = rt.get("ui", {}).get("display_max_side")
    record_raw = sim is None or a.save_raw
    try:
        with src:
            for raw in src:
                f = sim.apply(raw) if sim is not None else raw     # None = dibuang batas FPS profil
                rec = raw if record_raw else f
                if rec is not None:
                    if writer is None:
                        h, w = rec.image.shape[:2]
                        writer = cv2.VideoWriter(str(vid_path), cv2.VideoWriter_fourcc(*"mp4v"), a.fps, (w, h))
                        t0_ms = rec.ts_ms
                    writer.write(rec.image)
                    ts_f.write(f"{rec.index},{rec.ts_ms}\n")
                if f is None:
                    continue
                t_s = (f.ts_ms - t0_ms) / 1000.0

                vis = f.image.copy()
                s = scen[sel]
                t_last = t_s
                if guide is not None:
                    # mode panduan: hanya instruksi di bawah + baris REC kecil; tanpa panel hotkey
                    draw_guide(vis, guide, t_s)
                    gi = guide_at(guide, t_s)[0]
                    if gi != guide_idx:
                        if guide_idx >= 0:
                            step_beep()
                        guide_idx = gi
                    roi = vis[:44]
                    roi[:] = (roi * 0.35).astype(roi.dtype)
                    cv2.putText(vis, f"REC {t_s:6.1f}s   q = berhenti", (12, 31), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                                (80, 80, 255), 2, cv2.LINE_AA)
                    cv2.imshow(win, resize_max_side(vis, max_side) if max_side else vis)
                    k = cv2.waitKey(1) & 0xFF
                    if t_s >= guide[-1]["end"] or k in (ord("q"), 27):
                        break
                    continue
                elif t_s < a.calib_s:
                    # replay mengkalibrasi ulang di awal setiap video -> peraga harus menatap depan dulu
                    msg = f"TATAP LURUS KE DEPAN (kalibrasi) {a.calib_s - t_s:4.1f}s"
                    bot = vis[vis.shape[0] - 60:]
                    bot[:] = (bot * 0.35).astype(bot.dtype)
                    cv2.putText(vis, msg, (12, vis.shape[0] - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 255), 2,
                                cv2.LINE_AA)
                lines = [f"REC {t_s:7.1f}s  peraga {a.peraga}  cahaya {cahaya}  baris truth: {n_rows}",
                         f"Skenario [{s['no']:2d}] {s['code']} {s['level']} ({s['duration_s']} s): {s['desc']}",
                         "[ ] pilih  SPASI tandai  n normal  x batal  l cahaya  q selesai"]
                if mark:
                    el = t_s - mark["start"]
                    tgt = mark["scen"]["duration_s"] if mark["kind"] == "scen" else None
                    lines.append(f">>> MENANDAI {'NORMAL' if mark['kind']=='normal' else mark['scen']['code']} "
                                 f"{el:5.1f}s" + (f" / {tgt}s" if tgt else ""))
                # panel gelap semi-transparan di belakang teks: tetap terbaca di latar terang
                panel_h = 14 + 34 * len(lines)
                roi = vis[:panel_h]
                roi[:] = (roi * 0.35).astype(roi.dtype)
                for i, ln in enumerate(lines):
                    col = (80, 80, 255) if (mark and i == 3) else (255, 255, 255)
                    cv2.putText(vis, ln, (12, 34 + 34 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.8, col, 2, cv2.LINE_AA)
                cv2.imshow(win, resize_max_side(vis, max_side) if max_side else vis)
                k = cv2.waitKey(1) & 0xFF
                if k in (ord("q"), 27):
                    break
                elif k == ord("]"):
                    sel = (sel + 1) % len(scen)
                elif k == ord("["):
                    sel = (sel - 1) % len(scen)
                elif k == ord("l"):
                    cahaya = "redup" if cahaya == "normal" else "normal"
                elif k == ord("x"):
                    mark = None
                elif k == ord(" ") or k == ord("n"):
                    kind = "normal" if k == ord("n") else "scen"
                    if mark is None:
                        mark = {"kind": kind, "start": t_s, "scen": s}
                    else:
                        m = mark
                        row = {"video": vid_path.name, "start_s": f"{m['start']:.2f}", "end_s": f"{t_s:.2f}",
                               "expected_code": "NONE" if m["kind"] == "normal" else m["scen"]["code"],
                               "expected_level": "" if m["kind"] == "normal" else m["scen"]["level"],
                               "peraga": a.peraga, "cahaya": cahaya,
                               "scenario": "" if m["kind"] == "normal" else m["scen"]["no"]}
                        append_truth(truth, row)
                        n_rows += 1
                        print(f"[truth] {row}")
                        mark = None
                        if m["kind"] == "scen":
                            sel = (sel + 1) % len(scen)
    finally:
        if writer is not None:
            writer.release()
        ts_f.close()
        if guide is not None:
            for row in guide_truth_rows(guide, t_last, vid_path.name, a.peraga, cahaya):
                append_truth(truth, row)
                n_rows += 1
        if sim is not None:
            write_camera_sidecar(vid_path, sim.name, record_raw, sim.phone_hfov_deg)
        cv2.destroyAllWindows()
    print(f"Video: {vid_path}\nTruth: {truth} (+{n_rows} baris)")
    if sim is not None:
        print(f"Profil kamera: {sim.name} ({'frame HP mentah' if record_raw else 'frame simulasi'} direkam)")
    print("Catatan: segmen NORMAL dihitung alarm palsu hanya jika videonya berisi segmen normal saja; "
          "rekam sesi 'normal' terpisah untuk ukuran alarm palsu per jam.")


if __name__ == "__main__":
    main()
