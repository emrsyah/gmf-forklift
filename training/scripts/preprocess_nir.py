"""Simulasi NIR untuk data training: grayscale 3 channel + CLAHE (aturan kerja 7, Tech Spec bag. 5).

Memakai `simulate_nir` dari src/preprocess.py (fungsi yang sama dengan runtime) dengan clip/tile
dari configs/runtime.yaml (cabin.clahe_clip / cabin.clahe_tile).

Input : <src>/<split>/images|labels    (hasil split.py + augment.py)
Output: <dst>/images/<split>/*.jpg dan <dst>/labels/<split>/*.txt  (layout Ultralytics)

Contoh:
    python training/scripts/preprocess_nir.py --src data/cabin_work/split --dst data/cabin/cabin_v1
    python training/scripts/preprocess_nir.py --src data/cabin_work/split --dst data/cabin/cabin_v1_rgb --no-nir
"""
from __future__ import annotations

import argparse
import shutil
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import clahe_params, imread, imwrite, list_images, load_build_cfg, rpath  # noqa: E402
from src.preprocess import simulate_nir  # noqa: E402

SPLITS = ("train", "val", "test")


def run(src: Path, dst: Path, cfg: dict, nir: bool | None = None) -> dict[str, int]:
    nir = cfg["nir"]["enabled"] if nir is None else nir
    clip, tile = clahe_params(cfg)
    q = cfg["convert"]["jpeg_quality"]
    if dst.exists():
        shutil.rmtree(dst)
    counts = {}
    for sp in SPLITS:
        n = 0
        for p in list_images(src / sp / "images"):
            img = imread(p)
            if img is None:
                continue
            if nir:
                img = simulate_nir(img, clip, tile)
            imwrite(dst / "images" / sp / f"{p.stem}.jpg", img, q)
            lab = src / sp / "labels" / f"{p.stem}.txt"
            (dst / "labels" / sp).mkdir(parents=True, exist_ok=True)
            shutil.copyfile(lab, dst / "labels" / sp / f"{p.stem}.txt") if lab.exists() else \
                (dst / "labels" / sp / f"{p.stem}.txt").write_text("", encoding="utf-8")
            n += 1
        counts[sp] = n
    print(f"NIR={'on' if nir else 'off'} (clip={clip}, tile={tile}) -> {dst}: {counts}")
    return counts


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", default=None)
    ap.add_argument("--dst", default=None)
    ap.add_argument("--no-nir", action="store_true", help="salin tanpa NIR (untuk eksperimen pembanding)")
    ap.add_argument("--config", default=None)
    a = ap.parse_args()
    cfg = load_build_cfg(a.config)
    src = rpath(a.src or Path(cfg["paths"]["work"]) / "split")
    dst = rpath(a.dst or Path(cfg["paths"]["out"]) / cfg["version"])
    run(src, dst, cfg, False if a.no_nir else None)


if __name__ == "__main__":
    main()
