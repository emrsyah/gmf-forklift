"""Training YOLO11n kabin (Model 3) + export ONNX + laporan gerbang.

Membungkus perintah Tech Spec bag. 5 (parameter di training/configs/train_cabin.yaml):
    yolo detect train model=yolo11n.pt data=training/datasets/cabin.yaml imgsz=640 epochs=150 patience=30
        batch=32 hsv_h=0 hsv_s=0 hsv_v=0.4 degrees=10 translate=0.1 scale=0.4 fliplr=0.5 mosaic=1.0
        close_mosaic=15 seed=0

Setelah training:
  1. Export ONNX (opset 17, dynamic=False, imgsz 640, output mentah tanpa NMS) lalu cek dengan
     onnxruntime + src/backends/ort_backend.py.
  2. Salin best.pt / best.onnx -> models/weights/<weights_name>.pt / .onnx
  3. Evaluasi split test: P/R/mAP50 per kelas + recall pada conf 0.35 (phone, head_no_helmet,
     torso tanpa vest) -> <run>/gate_report.md

Contoh:
    # penuh (RunPod GPU)
    python training/train_cabin.py --device 0
    # smoke (laptop CPU)
    python training/train_cabin.py --epochs 1 --batch 8 --imgsz 320 --device cpu --workers 0 \
        --name smoke --weights-name cabin_smoke
    # hanya evaluasi + export dari bobot yang sudah ada
    python training/train_cabin.py --skip-train --weights training/runs/cabin_v1/weights/best.pt
"""
from __future__ import annotations

import argparse
import datetime as dt
import shutil
import sys
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "training" / "scripts"))
from common import iou, list_images, read_yolo  # noqa: E402

DEFAULT_CFG = REPO / "training" / "configs" / "train_cabin.yaml"


def rp(p) -> Path:
    p = Path(p)
    return p if p.is_absolute() else REPO / p


def load_yaml(p) -> dict:
    with open(p, encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


# ------------------------------------------------------------------ train
def train(cfg: dict, a) -> Path:
    from ultralytics import YOLO

    t = dict(cfg["train"])
    for k in ("epochs", "batch", "imgsz", "device", "workers", "fraction", "patience", "cache"):
        v = getattr(a, k, None)
        if v is not None:
            t[k] = v
    model = YOLO(str(rp(a.model or cfg["model"])))
    data = rp(a.data or cfg["data"])
    print(f"train: data={data} " + " ".join(f"{k}={v}" for k, v in t.items()))
    model.train(data=str(data), project=str(rp(cfg["project"])), name=a.name or cfg["name"], exist_ok=a.exist_ok, **t)
    save_dir = Path(model.trainer.save_dir)
    best = save_dir / "weights" / "best.pt"
    if not best.exists():
        best = save_dir / "weights" / "last.pt"
    return best


# ------------------------------------------------------------------ export + cek ONNX
def export_onnx(best: Path, cfg: dict, imgsz: int) -> Path:
    from ultralytics import YOLO

    e = cfg["export"]
    kw = dict(format=e["format"], imgsz=imgsz, opset=e["opset"], dynamic=e["dynamic"], simplify=e["simplify"])
    out = YOLO(str(best)).export(**kw)  # nms tidak di-set -> output mentah
    return Path(out)


def check_onnx(onnx_path: Path, names: list[str], sample_img: Path | None, conf: float) -> str:
    import onnxruntime as ort

    sess = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    ishape = sess.get_inputs()[0].shape
    oshape = sess.get_outputs()[0].shape
    expect = 4 + len(names)
    assert oshape[1] == expect, f"output ONNX {oshape}, harus (1, {expect}, N) — export tanpa NMS"
    msg = f"onnxruntime OK: input {ishape}, output {oshape}"
    if sample_img is not None:
        import cv2
        from src.backends.ort_backend import OrtYoloDetector

        det = OrtYoloDetector(onnx_path, imgsz=int(ishape[2]), names=names)
        img = cv2.imread(str(sample_img))
        dets = det.predict(img, conf=conf)
        msg += f"; OrtYoloDetector pada {sample_img.name}: {len(dets)} deteksi (conf>={conf})"
    return msg


# ------------------------------------------------------------------ evaluasi gerbang
def per_class_val(weights: Path, data: Path, split: str, imgsz: int, batch: int, device, names: list[str]) -> tuple:
    from ultralytics import YOLO

    m = YOLO(str(weights)).val(data=str(data), split=split, imgsz=imgsz, batch=batch, device=device,
                               plots=False, verbose=False)
    box = m.box
    rows = {}
    idx = list(getattr(box, "ap_class_index", []))
    for c, name in enumerate(names):
        if c in idx:
            i = idx.index(c)
            p, r, ap50, ap = box.class_result(i)
            rows[name] = (float(p), float(r), float(ap50), float(ap))
        else:
            rows[name] = None  # tidak ada instance di split ini
    return float(box.map50), float(box.map), rows


def recall_at_conf(weights: Path, img_dir: Path, lbl_dir: Path, names: list[str], g: dict, imgsz: int,
                   device, no_vest_iou: float) -> dict:
    """Recall per kelas pada conf tetap (bukan titik max-F1 seperti metrik Ultralytics)."""
    from ultralytics import YOLO

    model = YOLO(str(weights))
    C = {n: i for i, n in enumerate(names)}
    hit = {n: 0 for n in names} | {"no_vest": 0}
    tot = {n: 0 for n in names} | {"no_vest": 0}
    imgs = list_images(img_dir)
    nb = int(g["predict_batch"])
    for k in range(0, len(imgs), nb):
        chunk = imgs[k:k + nb]
        res = model.predict([str(p) for p in chunk], conf=g["recall_conf"], imgsz=imgsz, device=device, verbose=False)
        for p, r in zip(chunk, res):
            gt = [(b.cls, b.xyxy()) for b in read_yolo(lbl_dir / f"{p.stem}.txt")]
            pr = [(int(c), tuple(x)) for c, x in zip(r.boxes.cls.cpu().numpy(), r.boxes.xyxyn.cpu().numpy())]
            for c, name in enumerate(names):
                g_c = [x for cc, x in gt if cc == c]
                p_c = [x for cc, x in pr if cc == c]
                used = set()
                for gb in g_c:
                    tot[name] += 1
                    best, bj = 0.0, -1
                    for j, pb in enumerate(p_c):
                        if j not in used:
                            v = iou(gb, pb)
                            if v > best:
                                best, bj = v, j
                    if best >= g["match_iou"]:
                        used.add(bj)
                        hit[name] += 1
            # torso tanpa vest (aturan runtime): GT torso tanpa GT vest overlap; terdeteksi jika torso
            # prediksi cocok DAN tidak ada vest prediksi yang overlap >= no_vest_iou
            gv = [x for cc, x in gt if cc == C["vest"]]
            pv = [x for cc, x in pr if cc == C["vest"]]
            pt = [x for cc, x in pr if cc == C["torso"]]
            for cc, tb in gt:
                if cc != C["torso"] or any(iou(tb, v) >= no_vest_iou for v in gv):
                    continue
                tot["no_vest"] += 1
                m = [x for x in pt if iou(tb, x) >= g["match_iou"]]
                if m and not any(iou(m[0], v) >= no_vest_iou for v in pv):
                    hit["no_vest"] += 1
    return {k: (hit[k] / tot[k] if tot[k] else None, hit[k], tot[k]) for k in tot}


def write_report(path: Path, run_name: str, weights: Path, data: Path, g: dict, map50: float, map5095: float,
                 rows: dict, rec: dict, onnx_msg: str, args_line: str) -> bool:
    fmt = lambda v: "—" if v is None else f"{v:.3f}"  # noqa: E731
    checks = [("mAP50 semua kelas", map50, g["map50_min"])]
    for k, mn in g["recall_min"].items():
        checks.append((f"recall {k} @conf {g['recall_conf']}", rec.get(k, (None,))[0], mn))
    passed = all(v is not None and v >= mn for _, v, mn in checks)
    L = [
        f"# Gate report — {run_name}",
        "",
        "> **PUBLIC TEST SPLIT, NOT the L5 gate on people outside training.**",
        "> Angka ini dihitung pada split test dari dataset publik (gambar konstruksi/COCO, sudut dan",
        "> pencahayaan berbeda dari kamera kabin, label sebagian hasil pose-fill). Gerbang L5 yang sah",
        "> harus dihitung pada rekaman orang di luar tim dengan kamera kabin (CLAUDE.md L5, Tech Spec bag. 5).",
        "",
        f"- Tanggal: {dt.datetime.now().isoformat(timespec='seconds')}",
        f"- Bobot: `{weights}`",
        f"- Data: `{data}` (split `{g['split']}`)",
        f"- Argumen: `{args_line}`",
        f"- ONNX: {onnx_msg}",
        "",
        "## Ringkasan gerbang (indikatif)",
        "",
        "| Metrik | Nilai | Minimum | Status |",
        "|---|---|---|---|",
    ]
    for name, v, mn in checks:
        L.append(f"| {name} | {fmt(v)} | {mn:.2f} | {'OK' if (v is not None and v >= mn) else 'GAGAL'} |")
    L += [
        "",
        f"Status keseluruhan (data publik): **{'LOLOS (indikatif)' if passed else 'BELUM LOLOS'}**",
        "",
        "## Per kelas (Ultralytics val, P/R pada titik max-F1)",
        "",
        "| Kelas | Precision | Recall | mAP50 | mAP50-95 |",
        "|---|---|---|---|---|",
    ]
    for name, r in rows.items():
        L.append(f"| {name} | " + (" | ".join(f"{x:.3f}" for x in r) if r else "tidak ada instance | — | — | —") + " |")
    L += ["", f"mAP50 = {map50:.3f}, mAP50-95 = {map5095:.3f}", "",
          f"## Recall pada conf {g['recall_conf']} (IoU >= {g['match_iou']})", "",
          "| Kelas | Recall | Terdeteksi / total |", "|---|---|---|"]
    for k, (v, h, t) in rec.items():
        L.append(f"| {k} | {fmt(v)} | {h} / {t} |")
    L.append("")
    path.write_text("\n".join(L), encoding="utf-8")
    return passed


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(DEFAULT_CFG))
    ap.add_argument("--data", default=None)
    ap.add_argument("--model", default=None, help="bobot awal (default: model di config)")
    ap.add_argument("--name", default=None)
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--batch", type=int, default=None)
    ap.add_argument("--imgsz", type=int, default=None)
    ap.add_argument("--device", default=None, help="0 / 0,1 / cpu")
    ap.add_argument("--workers", type=int, default=None)
    ap.add_argument("--fraction", type=float, default=None, help="fraksi data train (smoke)")
    ap.add_argument("--patience", type=int, default=None)
    ap.add_argument("--cache", default=None)
    ap.add_argument("--exist-ok", action="store_true")
    ap.add_argument("--weights-name", default=None, help="nama file di models/weights (default: cabin_best)")
    ap.add_argument("--skip-train", action="store_true")
    ap.add_argument("--weights", default=None, help="bobot untuk --skip-train")
    ap.add_argument("--no-export", action="store_true")
    a = ap.parse_args()

    cfg = load_yaml(a.config)
    data = rp(a.data or cfg["data"])
    dcfg = load_yaml(data)
    names = [dcfg["names"][i] for i in sorted(dcfg["names"])]
    imgsz = a.imgsz or cfg["train"]["imgsz"]
    device = a.device if a.device is not None else cfg["train"].get("device")
    g = cfg["gate"]
    no_vest_iou = g.get("no_vest_iou")
    if no_vest_iou is None:
        no_vest_iou = float(load_yaml(rp(g["thresholds_config"]))["D10_ppe"]["vest_iou_max"])

    if a.skip_train:
        best = rp(a.weights)
    else:
        best = train(cfg, a)
    run_dir = best.parent.parent
    print(f"bobot terbaik: {best}")

    # export + salin
    wname = a.weights_name or cfg["export"]["weights_name"]
    wdir = rp(cfg["export"]["weights_dir"])
    wdir.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(best, wdir / f"{wname}.pt")
    onnx_msg = "export dilewati"
    root = Path(dcfg["path"])
    test_imgs = list_images(root / dcfg[g["split"]])
    if not a.no_export:
        onnx = export_onnx(best, cfg, a.imgsz or cfg["export"]["imgsz"])
        shutil.copyfile(onnx, wdir / f"{wname}.onnx")
        onnx_msg = check_onnx(wdir / f"{wname}.onnx", names, test_imgs[0] if test_imgs else None,
                              float(cfg["export"]["check_conf"]))
        print(onnx_msg)
    print(f"disalin ke {wdir / wname}.pt" + ("" if a.no_export else f" dan {wdir / wname}.onnx"))

    # evaluasi
    bs = a.batch or cfg["train"]["batch"]
    map50, map5095, rows = per_class_val(best, data, g["split"], imgsz, bs, device, names)
    lbl_dir = root / dcfg[g["split"]].replace("images", "labels", 1)
    rec = recall_at_conf(best, root / dcfg[g["split"]], lbl_dir, names, g, imgsz, device, no_vest_iou)
    print(f"\n{'kelas':16s} {'P':>6s} {'R':>6s} {'mAP50':>6s}")
    for n, r in rows.items():
        print(f"{n:16s} " + (" ".join(f"{x:6.3f}" for x in r[:3]) if r else "  (tidak ada instance)"))
    print(f"mAP50(all) = {map50:.3f}")
    for k, (v, h, t) in rec.items():
        print(f"recall@{g['recall_conf']} {k:16s} = {'—' if v is None else f'{v:.3f}'} ({h}/{t})")
    rpt = run_dir / "gate_report.md"
    args_line = " ".join(sys.argv[1:])
    ok = write_report(rpt, run_dir.name, best, data, g, map50, map5095, rows, rec, onnx_msg, args_line)
    print(f"\nlaporan: {rpt}  (PUBLIC TEST SPLIT, bukan gerbang L5) -> {'lolos indikatif' if ok else 'belum lolos'}")


if __name__ == "__main__":
    main()
