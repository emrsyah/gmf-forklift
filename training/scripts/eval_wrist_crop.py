"""A7: recall/presisi `phone` di split test kabin dengan dan tanpa potongan tangan (WristCrop), backend runtime.

Pipeline sama dengan live (src.cabin.wrist_crop + PhoneCrop.merge); filter wajah tidak dipakai (foto publik).
    python training/scripts/eval_wrist_crop.py --data data/cabin/cabin_v0 --split test [--side-u 2.0]
"""
from __future__ import annotations

import argparse
import csv
import sys
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import iou  # noqa: E402

from src.backends import make_detector  # noqa: E402
from src.backends.pose import make_pose  # noqa: E402
from src.cabin.wrist_crop import WristCrop  # noqa: E402
from src.config import Config  # noqa: E402


def read_phones(lbl: Path, w: int, h: int, phone_id: int) -> list:
    out = []
    if lbl.exists():
        for ln in lbl.read_text().split("\n"):
            p = ln.split()
            if len(p) >= 5 and int(p[0]) == phone_id:
                cx, cy, bw, bh = (float(v) for v in p[1:5])
                out.append(((cx - bw / 2) * w, (cy - bh / 2) * h, (cx + bw / 2) * w, (cy + bh / 2) * h))
    return out


def score(gt, pr, thr):
    used = set()
    for g in gt:
        cands = [(iou(g, p), j) for j, p in enumerate(pr) if j not in used]
        if cands:
            v, j = max(cands)
            if v >= thr:
                used.add(j)
    return len(used), len(pr) - len(used), len(gt) - len(used)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/cabin/cabin_v0")
    ap.add_argument("--split", default="test")
    ap.add_argument("--conf", type=float, default=None, help="default: D06_phone.conf")
    ap.add_argument("--iou", type=float, default=0.5)
    ap.add_argument("--side-u", type=float, default=None)
    ap.add_argument("--out", default=None, help="CSV per gambar")
    a = ap.parse_args()
    cfg = Config.load()
    d06 = cfg.thresholds["D06_phone"]
    conf = a.conf if a.conf is not None else d06.get("conf", 0.35)
    wc_cfg = dict(d06.get("wrist_crop", {}), enabled=True)
    if a.side_u:
        wc_cfg["side_u"] = a.side_u
    det = make_detector(cfg.runtime["cabin"]["yolo"])
    pose = make_pose(cfg.runtime["cabin"]["pose"])
    wc = WristCrop(wc_cfg)
    phone_id = det.names.index("phone")
    data = ROOT / a.data
    imgs = sorted((data / "images" / a.split).glob("*.*"))
    tot = {"base": [0, 0, 0], "wrist": [0, 0, 0]}
    neg = {"base": 0, "wrist": 0}
    n_neg = n_crop = 0
    rows = []
    for p in imgs:
        img = cv2.imread(str(p))
        if img is None:
            continue
        h, w = img.shape[:2]
        lbl = p.parent.parent.parent / "labels" / p.parent.name / (p.stem + ".txt")
        gt = read_phones(lbl, w, h, phone_id)
        full = det.predict(img, conf=conf)
        both = wc.run(det, pose, img, full, None, conf)
        n_crop += bool(wc.last_boxes)
        r = {"image": p.name, "gt": len(gt), "crops": len(wc.last_boxes)}
        for k, dets in (("base", full), ("wrist", both)):
            pr = [d.xyxy for d in dets if d.cls_name == "phone" and d.conf >= conf]
            tp, fp, fn = score(gt, pr, a.iou)
            for i, v in enumerate((tp, fp, fn)):
                tot[k][i] += v
            if not gt:
                neg[k] += bool(pr)
            r.update({f"{k}_tp": tp, f"{k}_fp": fp})
        n_neg += not gt
        rows.append(r)
    print(f"{len(rows)} gambar ({a.split}), {sum(r['gt'] > 0 for r in rows)} berisi ponsel, potongan tangan dipakai di "
          f"{n_crop}; conf {conf}, IoU {a.iou}")
    for k in ("base", "wrist"):
        tp, fp, fn = tot[k]
        print(f"  {k:5s}: recall {tp / max(1, tp + fn):.3f}  precision {tp / max(1, tp + fp):.3f}  TP {tp} FP {fp} FN {fn}"
              f"  | gambar tanpa ponsel dengan alarm: {neg[k]}/{n_neg}")
    if a.out:
        with open(a.out, "w", newline="") as f:
            wr = csv.DictWriter(f, list(rows[0]))
            wr.writeheader()
            wr.writerows(rows)


if __name__ == "__main__":
    main()
