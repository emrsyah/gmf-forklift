"""Bandingkan pembacaan mata pada dua ukuran wajah (resolusi kamera kabin lama vs baru) di webcam laptop.

Setiap frame webcam: grayscale + CLAHE (seperti Jetson), lalu diperkecil sampai tinggi wajah = target tiap varian
(lama ~130 px = forklift 2026-11-24 di 960x720; baru = lama x rasio mode sensor penuh 1440x1080) dan dijalankan
lewat model wajah yang sama dengan Jetson (YuNet + crop + Face Landmarker). Hanya efek jumlah piksel; buram lensa
dan sudut kamera forklift tidak ikut disimulasikan.

  python -m tools.res_compare [--cam 0] [--old-face 130] [--ratio 1.5]

3 detik pertama: tatap lurus ke layar (baseline EAR tiap varian). Tombol:
  SPASI = tandai/lepas "MEREM" (tahan tanda selama mata benar-benar tertutup)
  c     = kalibrasi ulang 3 detik      q / ESC = keluar + ringkasan
"""
from __future__ import annotations

import argparse
import statistics as st
import time

import cv2
import numpy as np

from src.cabin.face import make_face_model
from src.cabin.face_crop import YuNetDetector
from src.cabin.signals import LEFT_EYE, RIGHT_EYE, eye_aspect_ratio
from src.config import Config, deep_merge, load_yaml
from src.preprocess import simulate_nir

CAL_S = 3.0
VIEW_H = 420


class Variant:
    def __init__(self, name: str, face_px: float, face_cfg: dict, th: dict):
        self.name, self.face_px = name, face_px
        self.model = make_face_model(face_cfg)
        self.open_max = th.get("closed_openness_max", 0.30)
        self.blink_min = th.get("closed_blink_min", 0.50)
        self.logic = th.get("closed_logic", "and")
        self.base: float | None = None
        self.cal: list[float] = []
        self.log: list[tuple[bool, float | None, float | None, bool]] = []   # (ditandai, openness, blink, closed)

    def step(self, img: np.ndarray, scale: float, ts_ms: int, calibrating: bool, marked: bool):
        h, w = img.shape[:2]
        small = cv2.resize(img, (max(1, int(w * scale)), max(1, int(h * scale))), interpolation=cv2.INTER_AREA)
        obs = self.model.process(small, ts_ms, (0, 0))
        op = bl = None
        closed = False
        if obs.present and not obs.pose_only and obs.landmarks_px is not None:
            lm = obs.landmarks_px
            ear = (eye_aspect_ratio(lm[LEFT_EYE]) + eye_aspect_ratio(lm[RIGHT_EYE])) / 2
            bs = obs.blendshapes
            bl = max(bs.get("eyeBlinkLeft", 0.0), bs.get("eyeBlinkRight", 0.0))
            if calibrating:
                self.cal.append(ear)
            elif self.base is None and self.cal:
                self.base = st.median(self.cal)
            if self.base:
                op = ear / self.base
                by_o, by_b = op < self.open_max, bl > self.blink_min
                closed = (by_o and by_b) if self.logic == "and" else (by_o or by_b) if self.logic == "or" \
                    else by_b if self.logic == "blink_only" else by_o
        if not calibrating:
            self.log.append((marked, op, bl, closed))
        return small, obs, op, bl, closed

    def summary(self) -> str:
        def part(rows):
            ok = [r for r in rows if r[1] is not None]
            if not rows:
                return "-"
            if not ok:
                return f"{len(rows)} frame, mata tak terbaca"
            return (f"{len(rows)} frame, terbaca {len(ok)/len(rows):.0%}, openness median {st.median(r[1] for r in ok):.2f}"
                    f" min {min(r[1] for r in ok):.2f}, blink median {st.median(r[2] for r in ok):.2f}"
                    f", CLOSED {sum(r[3] for r in rows)/len(rows):.0%}")
        return (f"[{self.name} wajah {self.face_px:.0f} px]\n  MEREM (ditandai): {part([r for r in self.log if r[0]])}"
                f"\n  melek           : {part([r for r in self.log if not r[0]])}")


def panel(small: np.ndarray, obs, v: Variant, op, bl, closed) -> np.ndarray:
    k = VIEW_H / small.shape[0]
    view = cv2.resize(small, (int(small.shape[1] * k), VIEW_H), interpolation=cv2.INTER_NEAREST)
    if obs.present and obs.landmarks_px is not None and not obs.pose_only:
        for i in LEFT_EYE + RIGHT_EYE:
            x, y = obs.landmarks_px[i] * k
            cv2.circle(view, (int(x), int(y)), 2, (0, 255, 0), -1)
    col = (0, 0, 255) if closed else (0, 255, 255)
    cv2.rectangle(view, (0, 0), (view.shape[1], 64), (0, 0, 0), -1)
    cv2.putText(view, f"{v.name}: wajah {v.face_px:.0f}px", (8, 24), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
    txt = "mata tak terbaca" if op is None and bl is None else \
        f"open {op:.2f}  blink {bl:.2f}" if op is not None else f"kalibrasi... blink {bl:.2f}"
    cv2.putText(view, txt + ("  CLOSED" if closed else ""), (8, 54), cv2.FONT_HERSHEY_SIMPLEX, 0.65, col, 2)
    return view


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--cam", type=int, default=0)
    ap.add_argument("--old-face", type=float, default=130.0, help="tinggi wajah kamera lama (px)")
    ap.add_argument("--ratio", type=float, default=1.5, help="pembesaran mode baru (1440/960)")
    a = ap.parse_args()

    cfg = Config.load(overrides=deep_merge({}, load_yaml("configs/jetson.yaml")))
    face_cfg = cfg.runtime["cabin"]["face"]
    sig_th = cfg.thresholds.get("signals", {})
    cab = cfg.runtime["cabin"]
    variants = [Variant("lama", a.old_face, face_cfg, sig_th), Variant("baru", a.old_face * a.ratio, face_cfg, sig_th)]
    det = YuNetDetector(face_cfg.get("crop") or {})

    cap = cv2.VideoCapture(a.cam, cv2.CAP_DSHOW)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    face_h = None
    marked = False
    t_cal = time.monotonic()
    print("Tatap lurus 3 detik (kalibrasi). SPASI = tandai merem, c = kalibrasi ulang, q = keluar.")
    while True:
        ok, raw = cap.read()
        if not ok:
            break
        ts = int(time.monotonic() * 1000)
        img = simulate_nir(raw, cab.get("clahe_clip", 2.0), cab.get("clahe_tile", 8))
        faces = det.detect(img)
        if faces:
            f = max(faces, key=lambda b: (b[2] - b[0]) * (b[3] - b[1]))
            h = f[3] - f[1]
            face_h = h if face_h is None else 0.8 * face_h + 0.2 * h    # skala stabil, tidak lompat per frame
        if face_h is None:
            cv2.imshow("res_compare", img)
            if cv2.waitKey(1) & 0xFF in (ord("q"), 27):
                break
            continue
        calibrating = time.monotonic() - t_cal < CAL_S
        views = []
        for v in variants:
            scale = min(1.0, v.face_px / face_h)
            small, obs, op, bl, closed = v.step(img, scale, ts, calibrating, marked)
            views.append(panel(small, obs, v, op, bl, closed))
        out = np.hstack(views)
        if marked:
            cv2.putText(out, "MEREM", (out.shape[1] // 2 - 60, VIEW_H - 16), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        if calibrating:
            cv2.putText(out, "KALIBRASI: tatap lurus", (10, VIEW_H - 16), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
        cv2.imshow("res_compare", out)
        k = cv2.waitKey(1) & 0xFF
        if k in (ord("q"), 27):
            break
        if k == ord(" "):
            marked = not marked
        if k == ord("c"):
            t_cal = time.monotonic()
            for v in variants:
                v.base, v.cal = None, []
    cap.release()
    cv2.destroyAllWindows()
    print(f"\nambang closed: openness < {variants[0].open_max} {variants[0].logic} blink > {variants[0].blink_min}")
    for v in variants:
        print(v.summary())


if __name__ == "__main__":
    main()
