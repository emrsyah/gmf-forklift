"""Augmentasi offline (Tech Spec bag. 7) untuk split TRAIN saja, sebelum simulasi NIR.

Transformasi (parameter di cabin_build.yaml -> augment):
  - Motion blur kernel 3-7, p=0.3            (getaran forklift)
  - Gaussian noise sigma 5-15, p=0.3          (sensor di cahaya rendah)
  - Gain gelap/terang 0.4-1.3                 (shift malam, lampu hangar)
  - Cutout strip di dahi/mata 10-20% tinggi kepala, p=0.2 (bayangan helm)
Semua transformasi bersifat piksel, jadi kotak tidak bergeser; tetap dilewatkan lewat BboxParams
Albumentations supaya kotak ikut divalidasi/di-clip.

Urutan di pipeline: split -> augment (train) -> NIR (semua). Gain dan noise diterapkan sebelum
CLAHE, sama seperti frame gelap kamera yang baru di-CLAHE di runtime.

Contoh:
    python training/scripts/augment.py --dir data/cabin_work/split/train --copies 1
"""
from __future__ import annotations

import argparse
import random
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import Box, imread, imwrite, list_images, load_build_cfg, read_yolo, rpath, write_yolo  # noqa: E402

AUG_PREFIX = "aug"


def build_pipeline(acfg: dict):
    import albumentations as A

    k0, k1 = acfg["motion_blur"]["kernel"]
    s0, s1 = acfg["gauss_noise"]["sigma"]
    g0, g1 = acfg["gain"]["range"]
    return A.Compose(
        [
            A.MultiplicativeNoise(multiplier=(g0, g1), per_channel=False, elementwise=False, p=acfg["gain"]["p"]),
            A.MotionBlur(blur_limit=(k0, k1), p=acfg["motion_blur"]["p"]),
            A.GaussNoise(std_range=(s0 / 255.0, s1 / 255.0), per_channel=False, p=acfg["gauss_noise"]["p"]),
        ],
        bbox_params=A.BboxParams(format="yolo", label_fields=["cls"], clip=True, min_visibility=0.0),
        seed=int(acfg.get("seed", 0)),
    )


def face_cutout(img: np.ndarray, boxes: list[Box], ccfg: dict, cls_ids: dict[int, str], rnd: random.Random) -> np.ndarray:
    """Strip horizontal di area dahi-mata (posisi relatif kotak kepala/helm, per kelas)."""
    H, W = img.shape[:2]
    for b in boxes:
        if b.cls not in cls_ids or rnd.random() >= ccfg["p"]:
            continue
        x0, y0, x1, y1 = b.xyxy()
        bh = (y1 - y0) * H
        sh = rnd.uniform(*ccfg["height_frac"]) * bh
        r0, r1 = ccfg["region"][cls_ids[b.cls]]
        top = y0 * H + rnd.uniform(r0, r1) * bh - sh / 2
        ya, yb = int(max(0, top)), int(min(H, top + sh))
        xa, xb = int(max(0, x0 * W)), int(min(W, x1 * W))
        if yb > ya and xb > xa:
            img[ya:yb, xa:xb] = ccfg["fill"]
    return img


def run(d: Path, cfg: dict, copies: int | None = None) -> int:
    acfg = cfg["augment"]
    copies = int(acfg["copies_per_image"] if copies is None else copies)
    if copies <= 0:
        return 0
    pipe = build_pipeline(acfg)
    rnd = random.Random(int(acfg.get("seed", 0)))
    names = cfg["classes"]
    cut_ids = {names.index(c): c for c in acfg["face_cutout"]["classes"]}
    n = 0
    for p in list_images(d / "images"):
        if p.name.startswith(AUG_PREFIX):
            continue
        img = imread(p)
        if img is None:
            continue
        boxes = read_yolo(d / "labels" / f"{p.stem}.txt")
        for k in range(copies):
            bb = [[min(max(b.cx, 0), 1), min(max(b.cy, 0), 1), min(max(b.w, 1e-6), 1), min(max(b.h, 1e-6), 1)] for b in boxes]
            r = pipe(image=img.copy(), bboxes=np.array(bb, dtype=np.float32).reshape(-1, 4), cls=[b.cls for b in boxes])
            out_boxes = [Box(int(c), *map(float, xywh)) for xywh, c in zip(r["bboxes"], r["cls"])]
            im2 = face_cutout(r["image"], out_boxes, acfg["face_cutout"], cut_ids, rnd)
            stem = f"{AUG_PREFIX}{k}_{p.stem}"
            imwrite(d / "images" / f"{stem}.jpg", im2, cfg["convert"]["jpeg_quality"])
            write_yolo(d / "labels" / f"{stem}.txt", out_boxes)
            n += 1
    print(f"augment: {n} gambar baru di {d}")
    return n


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dir", required=True, help="folder split train (berisi images/ dan labels/)")
    ap.add_argument("--copies", type=int, default=None)
    ap.add_argument("--config", default=None)
    a = ap.parse_args()
    run(rpath(a.dir), load_build_cfg(a.config), a.copies)


if __name__ == "__main__":
    main()
