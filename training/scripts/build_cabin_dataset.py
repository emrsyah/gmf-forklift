"""Satu perintah: dataset publik -> dataset YOLO kabin siap training + training/datasets/cabin.yaml.

Urutan:
  1. download_public.py   (data/raw/<source>)
  2. convert_to_yolo.py   (class_map + pose-fill -> data/cabin_work/merged)
  3. split.py             (per grup + dHash -> data/cabin_work/split/{train,val,test})
  4. augment.py           (train saja, salinan tambahan)
  5. preprocess_nir.py    (grayscale 3ch + CLAHE -> data/cabin/<version>)
  6. tulis cabin.yaml (path absolut) + data/cabin/<version>/dataset_info.yaml

Contoh:
    # smoke (laptop, CPU)
    python training/scripts/build_cabin_dataset.py --limit 60
    # penuh (RunPod)
    python training/scripts/build_cabin_dataset.py
    # pakai data mentah yang sudah ada
    python training/scripts/build_cabin_dataset.py --skip-download
"""
from __future__ import annotations

import argparse
import datetime as dt
import shutil
import sys
import time
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import augment  # noqa: E402
import convert_to_yolo  # noqa: E402
import download_public  # noqa: E402
import preprocess_nir  # noqa: E402
import split  # noqa: E402
from common import load_build_cfg, rpath  # noqa: E402


def write_dataset_yaml(path: Path, root: Path, names: list[str], version: str, note: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = {
        "path": root.resolve().as_posix(),
        "train": "images/train",
        "val": "images/val",
        "test": "images/test",
        "names": {i: n for i, n in enumerate(names)},
    }
    header = (f"# DIBUAT OTOMATIS oleh training/scripts/build_cabin_dataset.py ({version}, {note}).\n"
              "# Path absolut khusus mesin ini; jalankan ulang build di mesin lain (mis. RunPod).\n")
    path.write_text(header + yaml.safe_dump(body, sort_keys=False), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=None, help="training/configs/cabin_build.yaml")
    ap.add_argument("--limit", type=int, default=None, help="maks gambar per sumber (smoke)")
    ap.add_argument("--only", nargs="*", default=None, help="hanya sumber tertentu")
    ap.add_argument("--version", default=None, help="nama versi data (default: version di config)")
    ap.add_argument("--skip-download", action="store_true")
    ap.add_argument("--no-pose-fill", action="store_true")
    ap.add_argument("--no-augment", action="store_true")
    ap.add_argument("--no-nir", action="store_true", help="hanya untuk eksperimen pembanding")
    ap.add_argument("--keep-work", action="store_true", help="simpan data/cabin_work (default: dihapus)")
    ap.add_argument("--dataset-yaml", default=None, help="path output data yaml (default: paths.dataset_yaml)")
    a = ap.parse_args()

    cfg = load_build_cfg(a.config)
    if a.no_pose_fill:
        cfg["pose_fill"]["enabled"] = False
    version = a.version or cfg["version"]
    raw = rpath(cfg["paths"]["raw"])
    work = rpath(cfg["paths"]["work"])
    out = rpath(cfg["paths"]["out"]) / version
    t0 = time.time()

    if not a.skip_download:
        print("== 1/5 download")
        download_public.run(raw, a.limit, a.only, cfg)
    print("== 2/5 convert + pose-fill")
    convert_to_yolo.run(raw, work / "merged", cfg, a.only)
    print("== 3/5 split per grup")
    split.run(work / "merged", work / "split", cfg)
    if cfg["augment"]["enabled"] and not a.no_augment:
        print("== 4/5 augment (train)")
        augment.run(work / "split" / "train", cfg)
    else:
        print("== 4/5 augment dilewati")
    print("== 5/5 NIR")
    counts = preprocess_nir.run(work / "split", out, cfg, False if a.no_nir else None)
    shutil.copyfile(work / "split" / "splits.csv", out / "splits.csv")

    raw_limits = sorted({str((yaml.safe_load((p / "source.yaml").read_text(encoding="utf-8")) or {}).get("limit"))
                         for p in raw.iterdir() if (p / "source.yaml").exists()})
    note = f"limit={a.limit}" if a.limit else f"raw limit={','.join(raw_limits)}"
    ds_yaml = rpath(a.dataset_yaml or cfg["paths"]["dataset_yaml"])
    write_dataset_yaml(ds_yaml, out, cfg["classes"], version, note)
    info = {
        "version": version,
        "built": dt.datetime.now().isoformat(timespec="seconds"),
        "limit": a.limit,
        "pose_fill": bool(cfg["pose_fill"]["enabled"]),
        "augment": bool(cfg["augment"]["enabled"] and not a.no_augment),
        "nir": not a.no_nir and bool(cfg["nir"]["enabled"]),
        "counts": counts,
        "classes": cfg["classes"],
        "note": "Hanya data publik. Bukan test set 'orang di luar tim' (gerbang L5).",
    }
    (out / "dataset_info.yaml").write_text(yaml.safe_dump(info, sort_keys=False), encoding="utf-8")
    if not a.keep_work:
        shutil.rmtree(work, ignore_errors=True)
    print(f"selesai dalam {time.time() - t0:.0f} s -> {ds_yaml}")


if __name__ == "__main__":
    main()
