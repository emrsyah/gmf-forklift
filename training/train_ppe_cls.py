"""A8: train classifier helm (kepala) dan rompi (dada) dari dataset PPE publik ber-anotasi upstream.

Potongan dibuat dengan geometri runtime (src.cabin.ppe_cls.head_box / torso_box) dari kotak wajah semu:
  kepala: kotak helmet/head GT -> wajah semu (u = 0,75 x sisi kotak, di bagian bawah kotak)
  dada  : kotak vest/no-vest GT + kepala GT di bagian atasnya -> wajah semu dari kepala itu
           (tanpa kepala di dalam kotak -> tidak dipakai: posisi dada tidak bisa ditebak sama dengan runtime)
Gambar: grayscale + CLAHE (simulate_nir, sama dengan runtime Jetson). Split = splits.csv cabin_v0 (per grup gambar
asli; salinan augmentasi Roboflow tetap satu grup). CNN kecil dari nol (tanpa bobot pretrained -> tanpa lisensi
tambahan), CPU cukup.

    python training/train_ppe_cls.py --out training/runs/ppe_cls_v1 [--epochs 25]
Keluaran: head.onnx, torso.onnx, report.md (dibandingkan dengan aturan YOLO kabin pada kotak GT test yang sama).
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import random
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from src.cabin.ppe_cls import HEAD_GEOM, TORSO_GEOM, cut, head_box, torso_box  # noqa: E402
from src.preprocess import simulate_nir  # noqa: E402

sys.path.insert(0, str(ROOT / "training" / "scripts"))
from tiny_cls import pick_threshold, predict, stats, train  # noqa: E402

# sumber -> {nama kelas upstream: (tugas, label)}; label 1 = pelanggaran (tanpa helm / tanpa rompi)
CLASS_MAP = {
    "ppe_jhboyo": {"helmet": ("head", 0), "head": ("head", 1), "vest": ("torso", 0)},
    "ppe_vincent": {"helmet": ("head", 0), "no-helmet": ("head", 1), "vest": ("torso", 0), "no-vest": ("torso", 1)},
    "construction_keremberke": {"hardhat": ("head", 0), "no-hardhat": ("head", 1),
                                "safety vest": ("torso", 0), "no-safety vest": ("torso", 1)},
}
FACE_FROM_HEAD = 0.75       # u wajah landmark ~ 0,75 x sisi terpanjang kotak kepala/helm GT
TORSO_W_TO_U, TORSO_FACE_ABOVE_U = 0.55, 0.7    # = thresholds D06_phone.filter torso_width_to_u / torso_face_above_u


def face_from_head(b, rng=None, jit=0.0):
    x1, y1, x2, y2 = b
    u = FACE_FROM_HEAD * max(x2 - x1, y2 - y1)
    cx, cy = (x1 + x2) / 2, y2 - 0.5 * u
    if rng is not None and jit:
        u *= 1 + rng.uniform(-jit, jit)
        cx += rng.uniform(-jit, jit) * u
        cy += rng.uniform(-jit, jit) * u
    return cx - u / 2, cy - u / 2, cx + u / 2, cy + u / 2


def read_boxes(lbl: Path, names: list[str], w: int, h: int):
    out = []
    if lbl.exists():
        for ln in lbl.read_text().split("\n"):
            p = ln.split()
            if len(p) >= 5:
                k = int(p[0])
                cx, cy, bw, bh = (float(v) for v in p[1:5])
                out.append((names[k] if k < len(names) else str(k),
                            ((cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h)))
    return out


def head_inside(torso_b, heads):
    """Kepala GT yang pusatnya di dalam kotak dada dan di 40% atasnya (label 'vest' kadang = badan + kepala)."""
    x1, y1, x2, y2 = torso_b
    c = [hb for hb in heads if x1 <= (hb[0] + hb[2]) / 2 <= x2 and y1 - 0.5 * (hb[3] - hb[1]) <= (hb[1] + hb[3]) / 2
         <= y1 + 0.4 * (y2 - y1)]
    return min(c, key=lambda hb: abs((hb[0] + hb[2]) / 2 - (x1 + x2) / 2)) if c else None


def split_of(src: str, name: str, splits: dict, group_split: dict, group_rx: dict) -> str:
    """Split cabin_v0 jika file ada di sana; selain itu per grup (regex class map, mis. salinan augmentasi Roboflow
    = satu grup): grup yang sudah punya split ikut split itu, grup baru di-hash 80/10/10 (deterministik)."""
    s = splits.get(f"{src}__{name}")
    if s is not None:
        return s
    stem = Path(name).stem
    g = f"{src}:{stem}"
    for rx in group_rx.get(src, []):
        m = rx.search(stem)
        if m:
            g = f"{src}:{m.group('g')}"
            break
    if g in group_split:
        return group_split[g]
    h = int(hashlib.md5(g.encode()).hexdigest()[:8], 16) % 10
    return "test" if h == 0 else "val" if h == 1 else "train"


def build(raws: list[Path], splits: dict, group_split: dict, group_rx: dict, size: int, reps: int, seed: int,
          min_u: float):
    """-> {(task, split): (X uint8 [n,size,size], y, meta)}; meta = (file, box) untuk perbandingan YOLO."""
    rng = random.Random(seed)
    data: dict = {}
    for src, cmap in CLASS_MAP.items():
        d = next((r / src for r in raws if (r / src / "images").exists()), None)
        if d is None:
            print(f"  [lewati] {src}: tidak ada di {raws}")
            continue
        names = d.joinpath("classes.txt").read_text().split("\n")
        names = [n.strip() for n in names if n.strip()]
        for img_p in sorted((d / "images").iterdir()):
            split = split_of(src, img_p.name, splits, group_split, group_rx)
            img = cv2.imread(str(img_p))
            if img is None:
                continue
            H, W = img.shape[:2]
            nir = simulate_nir(img, 2.0, 8)
            boxes = read_boxes(d / "labels" / (img_p.stem + ".txt"), names, W, H)
            heads = [b for n, b in boxes if cmap.get(n, ("", 0))[0] == "head"]
            for n, b in boxes:
                if n not in cmap:
                    continue
                task, lab = cmap[n]
                from_torso = False
                if task == "head":
                    anchor = b
                else:
                    anchor = head_inside(b, heads)
                    if anchor is None:
                        # tanpa kepala GT: wajah semu dari tepi atas kotak dada (aturan cadangan PhoneFilter:
                        # u = 0,55 x lebar, pusat wajah 0,7 u di atas tepi atas)
                        u = TORSO_W_TO_U * (b[2] - b[0])
                        cx, cy = (b[0] + b[2]) / 2, b[1] - TORSO_FACE_ABOVE_U * u
                        anchor = (cx - u / 2 / FACE_FROM_HEAD, cy + 0.5 * u - u / FACE_FROM_HEAD,
                                  cx + u / 2 / FACE_FROM_HEAD, cy + 0.5 * u)
                        from_torso = True
                if FACE_FROM_HEAD * max(anchor[2] - anchor[0], anchor[3] - anchor[1]) < min_u:
                    continue
                n_rep = reps if split == "train" else 1
                if from_torso and split == "test":
                    continue          # test dada hanya dari pasangan kepala GT (geometri paling mirip runtime)
                for r in range(n_rep):
                    fb = face_from_head(anchor, rng if split == "train" else None, 0.12 if r else 0.0)
                    box = head_box(fb) if task == "head" else torso_box(fb)
                    crop = cut(nir, box, size)
                    if split == "train" and rng.random() < 0.5:
                        crop = crop[:, ::-1]
                    X, y, meta = data.setdefault((task, split), ([], [], []))
                    X.append(np.ascontiguousarray(crop))
                    y.append(lab)
                    meta.append((f"{src}__{img_p.name}", b, str(img_p)))
    return {k: (np.stack(v[0]), np.array(v[1]), v[2]) for k, v in data.items()}


def yolo_rule(meta, task, ycfg, d10):
    """Aturan YOLO kabin saat ini pada kotak GT yang sama: pelanggaran jika ada kotak head_no_helmet (kepala) /
    torso tanpa vest (dada) yang menumpuk kotak GT (IoU >= 0,3); bukan pelanggaran jika tidak."""
    from src.backends import make_detector
    from src.cabin.yolo_cabin import helmet_wins, inside_frac, iou

    det = make_detector(ycfg)
    conf = d10.get("conf", 0.4)
    cache, out = {}, []
    for f, b, path in meta:
        if path not in cache:
            img = cv2.imread(path)
            cache.clear()
            cache[path] = det.predict(simulate_nir(img, 2.0, 8), conf=0.25)
        dets = cache[path]
        if task == "head":
            helmets = [d for d in dets if d.cls_name == "helmet" and d.conf >= conf]
            heads = [d for d in dets if d.cls_name == "head_no_helmet" and d.conf >= conf and not helmet_wins(d, helmets, d10)]
            out.append(any(iou(d.xyxy, b) >= 0.3 for d in heads))
        else:
            vests = [d for d in dets if d.cls_name == "vest" and d.conf >= conf]
            torsos = [d for d in dets if d.cls_name == "torso" and d.conf >= conf and iou(d.xyxy, b) >= 0.3]
            out.append(any(not any(iou(t.xyxy, v.xyxy) >= d10.get("vest_iou_max", 0.3)
                                   or inside_frac(v.xyxy, t.xyxy) >= d10.get("vest_inside_min", 0.6) for v in vests)
                           for t in torsos))
    return np.array(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw_full,data/raw", help="folder raw, dipisah koma (yang pertama berisi sumber dipakai)")
    ap.add_argument("--splits", default="data/cabin/cabin_v0/splits.csv")
    ap.add_argument("--out", default="training/runs/ppe_cls_v1")
    ap.add_argument("--size", type=int, default=96)
    ap.add_argument("--reps", type=int, default=3, help="salinan jitter per kotak train")
    ap.add_argument("--min-u", type=float, default=12.0, help="abaikan wajah semu lebih kecil dari ini (px)")
    ap.add_argument("--epochs", type=int, default=25)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-yolo-compare", action="store_true")
    a = ap.parse_args()
    out = ROOT / a.out
    out.mkdir(parents=True, exist_ok=True)
    rows = list(csv.DictReader(open(ROOT / a.splits)))
    splits = {r["file"]: r["split"] for r in rows}
    group_split = {r["group"]: r["split"] for r in rows}
    import yaml
    group_rx = {}
    for src in CLASS_MAP:
        cm = yaml.safe_load(open(ROOT / "training/datasets/class_maps" / f"{src}.yaml", encoding="utf-8")) or {}
        group_rx[src] = [re.compile(r) for r in ((cm.get("group") or {}).get("regex") or [])]
    t0 = time.time()
    data = build([ROOT / r for r in a.raw.split(",")], splits, group_split, group_rx, a.size, a.reps, a.seed, a.min_u)
    print({f"{k[0]}/{k[1]}": (len(v[1]), int(v[1].sum())) for k, v in sorted(data.items())},
          f"build {time.time() - t0:.0f} s", flush=True)
    from src.config import Config
    cfg = Config.load()
    report = {"geometry": {"head": HEAD_GEOM, "torso": TORSO_GEOM, "face_from_head": FACE_FROM_HEAD},
              "input_px": a.size, "counts": {f"{k[0]}/{k[1]}": [len(v[1]), int(v[1].sum())] for k, v in data.items()}}
    for task in ("head", "torso"):
        Xtr, ytr, _ = data[(task, "train")]
        Xva, yva, _ = data[(task, "val")]
        Xte, yte, mte = data[(task, "test")]
        net, best = train(task, Xtr, ytr, Xva, yva, a.epochs, out, a.seed)
        pv, pt = predict(net, Xva), predict(net, Xte)
        thr = pick_threshold(yva, pv, max_fpr=0.05)      # ambang di VAL: alarm palsu <= 5%
        r = {"val_best_bal_acc": best, "thr": thr, "test_cls": stats(yte, pt >= thr)}
        if not a.no_yolo_compare:
            r["test_yolo_rule"] = stats(yte, yolo_rule(mte, task, cfg.runtime["cabin"]["yolo"], cfg.thresholds["D10_ppe"]))
        report[task] = r
        print(task, json.dumps(r, indent=1), flush=True)
    (out / "report.json").write_text(json.dumps(report, indent=1))
    lines = ["# Classifier APD (A8) — test split cabin_v0 (data publik, per kotak GT)", "",
             "| Tugas | Model | Recall pelanggaran | Alarm palsu | Bal-acc | TP/FP/FN/TN |", "|---|---|---|---|---|---|"]
    for task in ("head", "torso"):
        for k in ("test_cls", "test_yolo_rule"):
            if k in report[task]:
                s = report[task][k]
                lines.append(f"| {task} | {'classifier (thr ' + str(report[task]['thr']) + ')' if k == 'test_cls' else 'aturan YOLO kabin'}"
                             f" | {s['recall_pos']:.3f} | {s['false_pos_rate']:.3f} | {s['bal_acc']:.3f} | "
                             f"{s['tp']}/{s['fp']}/{s['fn']}/{s['tn']} |")
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
