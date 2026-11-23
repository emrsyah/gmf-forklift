"""A9: bandingkan classifier mata dengan eye_closed MediaPipe (referensi) pada rekaman, juga setelah gambar
diperkecil (meniru wajah kecil di kamera kabin forklift). Tanpa label baru: MediaPipe di resolusi penuh = acuan.

    python eval/eye_cls_agreement.py eval/scenarios/sim_P1c.mp4 [--scales 1.0 0.5 0.35] [--step 2]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.cabin.calibration import Calibration  # noqa: E402
from src.cabin.eye_cls import EyeClassifier  # noqa: E402
from src.cabin.face import make_face_model  # noqa: E402
from src.cabin.ppe_cls import cut  # noqa: E402
from src.cabin.signals import LEFT_EYE, RIGHT_EYE, SignalExtractor  # noqa: E402
from src.config import Config  # noqa: E402
from src.preprocess import simulate_nir  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("videos", nargs="+")
    ap.add_argument("--scales", type=float, nargs="+", default=[1.0, 0.5, 0.35])
    ap.add_argument("--step", type=int, default=2)
    ap.add_argument("--set", action="append", default=[], help="override signals.eye_classifier.<k>=<v>")
    ap.add_argument("--dump", default=None, help="simpan contoh potongan (baris atas = acuan tertutup) ke file ini")
    a = ap.parse_args()
    cfg = Config.load()
    ec = dict(cfg.thresholds["signals"].get("eye_classifier", {}), enabled=True)
    for s in a.set:
        k, v = s.split("=", 1)
        ec[k] = type(ec.get(k, 0.0))(v) if k in ec else float(v)
    clf = EyeClassifier(ec)
    fc = dict(cfg.runtime["cabin"]["face"], crop={"enabled": False})
    res = {s: {"tp": 0, "fp": 0, "fn": 0, "tn": 0, "none": 0} for s in a.scales}
    dumps = {True: [], False: []}
    eye_px = {s: [] for s in a.scales}
    for v in a.videos:
        face = make_face_model(fc)
        sig = SignalExtractor(cfg.thresholds["signals"], Calibration(cfg.thresholds.get("calibration", {})))
        cap = cv2.VideoCapture(v)
        fps = cap.get(cv2.CAP_PROP_FPS) or 30
        i = 0
        while True:
            ok, img = cap.read()
            if not ok:
                break
            i += 1
            ts = int(i / fps * 1000)
            nir = simulate_nir(img, 2.0, 8)
            obs = face.process(nir, ts)
            s = sig.update(obs, ts)          # setiap frame (median filter + kalibrasi seperti live)
            if i % a.step or not (obs.present and s.eye_valid and not s.pose_only):
                continue
            lm = obs.landmarks_px
            eyes = np.stack([lm[RIGHT_EYE].mean(0)[:2], lm[LEFT_EYE].mean(0)[:2]])
            ref = bool(s.eye_closed_strict if s.eye_closed_strict is not None else s.eye_closed)
            for sc in a.scales:
                im = img if sc == 1.0 else cv2.resize(img, None, fx=sc, fy=sc, interpolation=cv2.INTER_AREA)
                im_n = simulate_nir(im, 2.0, 8)
                p = clf.p_closed(im_n, eyes * sc)
                if a.dump and sc == 1.0 and len(dumps[ref]) < 12:
                    b = clf.boxes(eyes)[0]
                    dumps[ref].append(cv2.putText(cv2.resize(cut(im_n, b, ec.get("input_px", 32)), (96, 96)),
                                                  f"{p:.2f}" if p is not None else "-", (2, 12), 0, 0.4, 255, 1))
                eye_px[sc].append(float(np.hypot(*(eyes[0] - eyes[1]))) * sc)
                r = res[sc]
                if p is None:
                    r["none"] += 1
                    continue
                pred = p >= ec.get("enter_p", 0.6)
                r["tp" if pred and ref else "fp" if pred else "fn" if ref else "tn"] += 1
    if a.dump:
        rows = []
        for k in (True, False):
            t = dumps[k] + [np.zeros((96, 96), np.uint8)] * (12 - len(dumps[k]))
            rows.append(np.hstack(t))
        cv2.imwrite(a.dump, np.vstack(rows))
    print("Acuan = eye_closed MediaPipe (resolusi penuh, eye_valid). 'Tertutup' classifier = P >= enter_p.")
    print("| Skala | Jarak mata (px, median) | Recall tertutup | Terbuka -> tertutup | Setuju | n | tak dinilai |")
    print("|---|---|---|---|---|---|---|")
    for sc in a.scales:
        r = res[sc]
        n = r["tp"] + r["fp"] + r["fn"] + r["tn"]
        rec = r["tp"] / max(1, r["tp"] + r["fn"])
        fpr = r["fp"] / max(1, r["fp"] + r["tn"])
        agree = (r["tp"] + r["tn"]) / max(1, n)
        print(f"| {sc} | {np.median(eye_px[sc]) if eye_px[sc] else 0:.0f} | {rec:.3f} | {fpr:.3f} | {agree:.3f} | {n} "
              f"| {r['none']} |")


if __name__ == "__main__":
    main()
