"""Bandingkan Face Landmarker biasa vs detektor + crop: berapa % frame wajahnya ditemukan, dan ms/frame.

Sumber: video (frame berurutan) atau gambar diam (diulang `--repeat` kali). `--forklift-sim` meniru kondisi
uji forklift 2026-11-23 dari video laptop: wajah diperkecil, diletakkan di tepi bawah (dagu terpotong),
latar atas terang (langit), wajah lebih gelap.

  python -m tools.face_bench --image snap.jpg
  python -m tools.face_bench --video eval/scenarios/sim_P1.mp4 --frames 300 --forklift-sim
"""
from __future__ import annotations

import argparse
import time

import cv2
import numpy as np

from src.cabin.face import make_face_model
from src.config import Config


def forklift_sim(img: np.ndarray, scale: float = 0.45, cut_px: int = 25, sky: int = 235,
                 face_gain: float = 0.55) -> np.ndarray:
    h, w = img.shape[:2]
    small = cv2.resize(img, (int(w * scale), int(h * scale)), interpolation=cv2.INTER_AREA)
    small = np.clip(small.astype(np.float32) * face_gain, 0, 255).astype(np.uint8)
    out = np.full((h, w, 3), 90, np.uint8)
    out[: h // 2] = sky                                        # langit terang di separuh atas
    sh, sw = small.shape[:2]
    x0, y0 = (w - sw) // 2, h - sh + cut_px                    # di tepi bawah, `cut_px` terpotong
    out[y0:h, x0:x0 + sw] = small[: h - y0]
    return out


def frames(a):
    if a.image:
        img = cv2.imread(a.image)
        for _ in range(a.repeat):
            yield img
        return
    cap = cv2.VideoCapture(a.video)
    n = 0
    while n < a.frames:
        ok, img = cap.read()
        if not ok:
            break
        n += 1
        if n % a.every:
            continue
        yield forklift_sim(img) if a.forklift_sim else img


def run(model, imgs) -> tuple[float, float, list]:
    found, t, srcs = 0, 0.0, []
    for i, img in enumerate(imgs):
        t0 = time.perf_counter()
        obs = model.process(img, i * 33)
        t += time.perf_counter() - t0
        found += obs.present
        srcs.append(getattr(model, "last_source", ""))
    n = max(1, len(imgs))
    return 100 * found / n, 1000 * t / n, srcs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--image")
    ap.add_argument("--video")
    ap.add_argument("--frames", type=int, default=300)
    ap.add_argument("--every", type=int, default=1)
    ap.add_argument("--repeat", type=int, default=30)
    ap.add_argument("--forklift-sim", action="store_true")
    ap.add_argument("--save", help="simpan contoh frame masukan (cek visual)")
    ap.add_argument("--margin", type=float, default=None, help="timpa crop.margin")
    a = ap.parse_args()
    imgs = list(frames(a))
    if a.save and imgs:
        cv2.imwrite(a.save, imgs[len(imgs) // 2])
    face_cfg = Config.load().runtime["cabin"]["face"]
    crop_cfg = dict(face_cfg.get("crop") or {}, enabled=True)
    if a.margin is not None:
        crop_cfg["margin"] = a.margin
    base = make_face_model(dict(face_cfg, crop={"enabled": False}))
    crop = make_face_model(dict(face_cfg, crop=crop_cfg))
    pb, mb, _ = run(base, imgs)
    pc, mc, srcs = run(crop, imgs)
    from collections import Counter
    print(f"frames: {len(imgs)}")
    print(f"biasa      : wajah {pb:5.1f}%  {mb:5.1f} ms/frame")
    print(f"deteksi+crop: wajah {pc:5.1f}%  {mc:5.1f} ms/frame  sumber {dict(Counter(srcs))}")


if __name__ == "__main__":
    main()
