"""Training YOLO11n kamera depan (Model 4) + export ONNX + laporan metrik per kelas.

Membungkus perintah Tech Spec bag. 6 (parameter di training/configs/train_front.yaml):
    yolo detect train model=yolo11n.pt data=training/datasets/front.yaml \
        imgsz=640 epochs=150 patience=30 batch=32 fliplr=0.5 mosaic=1.0 seed=0

Fungsi train/export/cek ONNX/val dipakai ulang dari train_cabin.py (export tanpa `nms` -> output mentah
(1, 4+nc, N) untuk src/backends/ort_backend.py). Tidak ada gerbang lulus: L9 hanya menuntut orang terdeteksi
dan dilacak stabil; laporan berisi P/R/mAP per kelas pada split test publik.

Contoh:
    python training/train_front.py --device 0                                   # penuh (GPU)
    python training/train_front.py --epochs 1 --batch 8 --imgsz 320 --device cpu --workers 0 \
        --name front_smoke --weights-name front_smoke                           # smoke (CPU)
    python training/train_front.py --skip-train --weights training/runs/front_v1/weights/best.pt
"""
from __future__ import annotations

import argparse
import datetime as dt
import shutil
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "training"))
import train_cabin as tc  # noqa: E402  (train, export_onnx, check_onnx, per_class_val, load_yaml, rp)
from common import list_images  # noqa: E402

DEFAULT_CFG = REPO / "training" / "configs" / "train_front.yaml"


def write_report(path: Path, run_name: str, weights: Path, data: Path, split: str, map50: float, map5095: float,
                 rows: dict, onnx_msg: str, args_line: str, counts: dict | None) -> None:
    L = [f"# Front model report — {run_name}", "",
         "> Split test **dataset publik** (COCO/LOCO/CrowdHuman/Apron), bukan hangar GMF. Tidak ada gerbang angka",
         "> untuk L9; kelas tanpa instance (mis. ladder_stand) belum bisa dinilai sampai ada data sintetis/rekaman.",
         "",
         f"- Tanggal: {dt.datetime.now().isoformat(timespec='seconds')}",
         f"- Bobot: `{weights}`", f"- Data: `{data}` (split `{split}`)", f"- Argumen: `{args_line}`",
         f"- ONNX: {onnx_msg}", "",
         "## Per kelas (Ultralytics val, P/R pada titik max-F1)", "",
         "| Kelas | Precision | Recall | mAP50 | mAP50-95 |", "|---|---|---|---|---|"]
    for name, r in rows.items():
        L.append(f"| {name} | " + (" | ".join(f"{x:.3f}" for x in r) if r else "tidak ada instance | — | — | —") + " |")
    L += ["", f"mAP50 = {map50:.3f}, mAP50-95 = {map5095:.3f}", ""]
    if counts:
        L += ["## Jumlah instance di dataset (dataset_info.yaml)", "", "```", str(counts), "```", ""]
    path.write_text("\n".join(L), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(DEFAULT_CFG))
    ap.add_argument("--data", default=None)
    ap.add_argument("--model", default=None)
    ap.add_argument("--name", default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--imgsz", type=int, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--fraction", type=float, default=None)
    ap.add_argument("--patience", type=int, default=None)
    ap.add_argument("--cache", default=None)
    ap.add_argument("--exist-ok", action="store_true")
    ap.add_argument("--weights-name", default=None, help="nama di models/weights (default: front_best)")
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--weights", default=None)
    ap.add_argument("--no-export", action="store_true")
    ap.add_argument("--no-plots", action="store_true")
    a = ap.parse_args()

    cfg = tc.load_yaml(a.config)
    if a.no_plots:
        cfg["train"]["plots"] = False
    data = tc.rp(a.data or cfg["data"])
    dcfg = tc.load_yaml(data)
    names = [dcfg["names"][i] for i in sorted(dcfg["names"])]
    imgsz = a.imgsz or cfg["train"]["imgsz"]
    device = a.device if a.device is not None else cfg["train"].get("device")
    split = cfg["report"]["split"]

    best = tc.rp(a.weights) if a.skip_train else tc.train(cfg, a)
    run_dir = best.parent.parent
    print(f"bobot terbaik: {best}")

    wname = a.weights_name or cfg["export"]["weights_name"]
    wdir = tc.rp(cfg["export"]["weights_dir"])
    wdir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(best, wdir / f"{wname}.pt")
    onnx_msg = "export dilewati"
    root = Path(dcfg["path"])
    test_imgs = list_images(root / dcfg[split])
    if not a.no_export:
        onnx = tc.export_onnx(best, cfg, a.imgsz or cfg["export"]["imgsz"])
        shutil.copyfile(onnx, wdir / f"{wname}.onnx")
        onnx_msg = tc.check_onnx(wdir / f"{wname}.onnx", names, test_imgs[0] if test_imgs else None,
                                 float(cfg["export"]["check_conf"]))
        print(onnx_msg)
    print(f"disalin ke {wdir / wname}.pt" + ("" if a.no_export else f" dan {wdir / wname}.onnx"))

    bs = a.batch or cfg["train"]["batch"]
    map50, map5095, rows = tc.per_class_val(best, data, split, imgsz, bs, device, names)
    print(f"\n{'kelas':16s} {'P':>6s} {'R':>6s} {'mAP50':>6s}")
    for n, r in rows.items():
        print(f"{n:16s} " + (" ".join(f"{x:6.3f}" for x in r[:3]) if r else "  (tidak ada instance)"))
    print(f"mAP50(all) = {map50:.3f}")
    info = root / "dataset_info.yaml"
    counts = tc.load_yaml(info).get("counts") if info.exists() else None
    rpt = run_dir / "front_report.md"
    write_report(rpt, run_dir.name, best, data, split, map50, map5095, rows, onnx_msg, " ".join(sys.argv[1:]), counts)
    print(f"\nlaporan: {rpt}  (split test data publik; L9 tanpa gerbang angka)")


if __name__ == "__main__":
    main()
