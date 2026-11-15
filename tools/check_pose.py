"""Cek 30 detik: arah tanda yaw/pitch/roll dan kalibrasi, langsung di depan webcam.

  python -m tools.check_pose            (q = keluar, c = kalibrasi ulang)

Ikuti instruksi di layar. Setiap langkah ditandai LULUS jika tanda sinyal sesuai konvensi:
  menunduk         -> pitch NEGATIF   (penting: D-04 anggukan dan D-05 pangkuan bergantung pada ini)
  miring ke bahu   -> |roll| > 25
  menoleh kanan/kiri -> |yaw| > 30
Jika pitch terbalik, ubah `cabin.face.pitch_sign: -1` di configs/runtime.yaml.
"""
from __future__ import annotations

import argparse

import cv2

from src.capture import make_source, parse_source_arg
from src.config import Config, deep_merge

STEPS = [
    ("Tatap lurus ke depan (kalibrasi 5 s)", lambda s, cal: cal.done),
    ("Menunduk ke pangkuan", lambda s, cal: s.pitch < -20),
    ("Tegak lagi, lalu MENDONGAK", lambda s, cal: s.pitch > 10),
    ("Menoleh ke KANAN", lambda s, cal: abs(s.yaw) > 30),
    ("Menoleh ke KIRI", lambda s, cal: abs(s.yaw) > 30),
    ("Miringkan kepala ke bahu", lambda s, cal: abs(s.roll) > 25),
    ("Pejamkan mata 2 detik", lambda s, cal: s.eye_closed),
]


def main(argv=None):
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", default="webcam")
    ap.add_argument("--simulate-nir", action="store_true")
    a = ap.parse_args(argv)
    cfg = Config.load()
    rt = cfg.runtime
    rt["cabin"]["source"] = deep_merge(rt["cabin"]["source"], parse_source_arg(a.source))
    rt["cabin"]["simulate_nir"] = a.simulate_nir
    from src.cabin.pipeline import CabinPipeline

    pipe = CabinPipeline(rt, cfg.thresholds, cfg.events, use_yolo=False)
    step, hold, yaw_right = 0, 0, None
    results = []
    with make_source(rt["cabin"]["source"]) as src:
        for f in src:
            st = pipe.process(f)
            s = st.sig
            img = st.image.copy()
            if step < len(STEPS):
                name, ok = STEPS[step]
                passed = s.face_present and ok(s, pipe.calibration)
                hold = hold + 1 if passed else 0
                if hold >= 10:
                    if step == 3:
                        yaw_right = s.yaw
                    results.append((name, f"yaw {s.yaw:+.0f} pitch {s.pitch:+.0f} roll {s.roll:+.0f}"))
                    step, hold = step + 1, 0
                msg = f"[{step + 1}/{len(STEPS)}] {name}"
            else:
                msg = "SELESAI - tekan q"
            lines = [msg, f"yaw {s.yaw:+6.1f}  pitch {s.pitch:+6.1f}  roll {s.roll:+6.1f}  "
                          f"open {s.openness:.2f}  mata {'TUTUP' if s.eye_closed else 'buka'}",
                     f"kalibrasi: {pipe.calibration.status}"]
            for i, (n, v) in enumerate(results):
                lines.append(f"OK  {n}: {v}")
            for i, ln in enumerate(lines):
                cv2.putText(img, ln, (10, 25 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 0, 0), 3)
                cv2.putText(img, ln, (10, 25 + 22 * i), cv2.FONT_HERSHEY_SIMPLEX, 0.55, (0, 255, 0), 1)
            cv2.imshow(rt.get("ui", {}).get("window_pose", "check_pose"), img)
            k = cv2.waitKey(1) & 0xFF
            if k == ord("q"):
                break
            if k == ord("c"):
                pipe.recalibrate()
    cv2.destroyAllWindows()
    print("\nHasil cek pose:")
    for n, v in results:
        print(f"  OK  {n}: {v}")
    if yaw_right is not None:
        print(f"  Info: menoleh kanan memberi yaw {'positif' if yaw_right > 0 else 'negatif'} "
              f"(D-05 memakai |yaw|, jadi tanda yaw tidak memengaruhi deteksi).")
    if len(results) < 3:
        print("  Langkah menunduk/mendongak belum lulus: jika pitch naik saat menunduk, set "
              "cabin.face.pitch_sign: -1 di configs/runtime.yaml")


if __name__ == "__main__":
    main()
