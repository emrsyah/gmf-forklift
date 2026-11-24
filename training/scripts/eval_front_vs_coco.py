"""Gerbang ganti bobot kamera depan: front_best vs YOLO11n COCO (bobot runtime saat ini) di split test yang SAMA.

COCO dipetakan ke kelas kita lewat `classes.aliases` di configs/front_camera.yaml (persis seperti runtime),
jadi perbandingannya adil untuk kelas yang bisa dilihat COCO (person, obstacle=kursi/bangku/pot/koper,
gse_vehicle, aircraft_part). Cone hanya ada di front_best. Metrik per kelas: AP50 dan recall/presisi pada
ambang runtime (front.person_conf / classes.obstacle_conf).

    python training/scripts/eval_front_vs_coco.py --data training/datasets/front.yaml \
        --new models/weights/front_best.pt --base models/weights/yolo11n.pt --out training/runs/front_v1/gate_vs_coco.md
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import list_images, read_yolo, rpath  # noqa: E402


def iou_m(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    if not len(a) or not len(b):
        return np.zeros((len(a), len(b)))
    x1 = np.maximum(a[:, None, 0], b[None, :, 0])
    y1 = np.maximum(a[:, None, 1], b[None, :, 1])
    x2 = np.minimum(a[:, None, 2], b[None, :, 2])
    y2 = np.minimum(a[:, None, 3], b[None, :, 3])
    inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
    aa = (a[:, 2] - a[:, 0]) * (a[:, 3] - a[:, 1])
    ab = (b[:, 2] - b[:, 0]) * (b[:, 3] - b[:, 1])
    return inter / (aa[:, None] + ab[None, :] - inter + 1e-9)


def match(preds: list[tuple[float, np.ndarray]], gts: np.ndarray, thr: float = 0.5) -> list[tuple[float, bool]]:
    """Greedy per gambar (urut confidence): (conf, true_positive)."""
    used = np.zeros(len(gts), bool)
    out = []
    for conf, box in sorted(preds, key=lambda x: -x[0]):
        if len(gts):
            ious = iou_m(box[None], gts)[0]
            ious[used] = 0
            j = int(ious.argmax())
            if ious[j] >= thr:
                used[j] = True
                out.append((conf, True))
                continue
        out.append((conf, False))
    return out


def ap_and_pr(hits: list[tuple[float, bool]], n_gt: int, conf_thr: float) -> dict:
    if n_gt == 0:
        return {"n_gt": 0, "ap50": None, "recall": None, "precision": None}
    hits = sorted(hits, key=lambda x: -x[0])
    tp = np.cumsum([h[1] for h in hits]) if hits else np.zeros(0)
    fp = np.cumsum([not h[1] for h in hits]) if hits else np.zeros(0)
    rec = tp / n_gt if len(tp) else np.zeros(0)
    prec = tp / np.maximum(tp + fp, 1e-9) if len(tp) else np.zeros(0)
    mrec = np.concatenate([[0.0], rec, [1.0]])
    mpre = np.concatenate([[1.0], prec, [0.0]])
    mpre = np.flip(np.maximum.accumulate(np.flip(mpre)))
    x = np.linspace(0, 1, 101)
    trap = getattr(np, "trapezoid", None) or np.trapz
    ap = float(trap(np.interp(x, mrec, mpre), x))
    sel = [h for h in hits if h[0] >= conf_thr]
    ntp = sum(h[1] for h in sel)
    return {"n_gt": n_gt, "ap50": ap, "recall": ntp / n_gt, "precision": ntp / len(sel) if sel else None}


def evaluate(weights: str, imgs: list[Path], lbl_dir: Path, ours: list[str], aliases: dict, conf_thr: dict,
             imgsz: int, device) -> dict:
    from ultralytics import YOLO

    m = YOLO(weights)
    names = m.names
    hits = {c: [] for c in ours}
    ngt = {c: 0 for c in ours}
    for i in range(0, len(imgs), 32):
        chunk = imgs[i:i + 32]
        res = m.predict([str(p) for p in chunk], imgsz=imgsz, conf=0.01, device=device, verbose=False)
        for p, r in zip(chunk, res):
            gts: dict[str, list] = {c: [] for c in ours}
            for b in read_yolo(lbl_dir / f"{p.stem}.txt"):
                gts[ours[b.cls]].append(b.xyxy())
            pr: dict[str, list] = {c: [] for c in ours}
            for c, xyxy, s in zip(r.boxes.cls.cpu().numpy(), r.boxes.xyxyn.cpu().numpy(), r.boxes.conf.cpu().numpy()):
                n = names[int(c)]
                n = aliases.get(n, n)
                if n in pr:
                    pr[n].append((float(s), np.asarray(xyxy, float)))
            for c in ours:
                g = np.array(gts[c], float).reshape(-1, 4)
                ngt[c] += len(g)
                hits[c] += match(pr[c], g)
    return {c: ap_and_pr(hits[c], ngt[c], conf_thr.get(c, conf_thr["_default"])) for c in ours}


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data", default="training/datasets/front.yaml")
    ap.add_argument("--split", default="test")
    ap.add_argument("--new", default="models/weights/front_best.pt")
    ap.add_argument("--base", default="models/weights/yolo11n.pt")
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--device", default=None)
    ap.add_argument("--out", default="training/runs/front_v1/gate_vs_coco.md")
    # front_v2 vs front_v1: --base front_v1.pt --gate person,obstacle --tol 0.01 --min-ap50 box=0.5
    ap.add_argument("--gate", default="person,obstacle", help="kelas yang AP50-nya tidak boleh turun vs base")
    ap.add_argument("--tol", type=float, default=0.0, help="toleransi penurunan AP50 kelas --gate")
    ap.add_argument("--min-ap50", default="", help="ambang AP50 mutlak untuk kelas baru, mis. box=0.5,obstacle=0.6")
    a = ap.parse_args()

    d = yaml.safe_load(rpath(a.data).read_text(encoding="utf-8"))
    root = Path(d["path"])
    img_dir = root / d[a.split]
    lbl_dir = Path(str(img_dir).replace("/images/", "/labels/").replace("\\images\\", "\\labels\\"))
    ours = [d["names"][i] for i in sorted(d["names"])] if isinstance(d["names"], dict) else list(d["names"])
    cam = yaml.safe_load(rpath("configs/front_camera.yaml").read_text(encoding="utf-8"))
    thr = yaml.safe_load(rpath("configs/thresholds.yaml").read_text(encoding="utf-8")).get("front", {})
    obst = float(cam["classes"].get("obstacle_conf", 0.35))
    conf_thr = {"person": float(thr.get("person_conf", 0.4)), "_default": obst}
    imgs = list_images(img_dir)
    print(f"{len(imgs)} gambar {a.split}; ambang {conf_thr}")
    base = evaluate(a.base, imgs, lbl_dir, ours, dict(cam["classes"].get("aliases") or {}), conf_thr, a.imgsz, a.device)
    new = evaluate(a.new, imgs, lbl_dir, ours, {}, conf_thr, a.imgsz, a.device)

    def f(x):
        return "-" if x is None else f"{x:.3f}"

    rows = ["| kelas | n_gt | AP50 COCO | AP50 baru | recall COCO | recall baru | presisi COCO | presisi baru |",
            "|---|---|---|---|---|---|---|---|"]
    for c in ours:
        b, n = base[c], new[c]
        rows.append(f"| {c} | {n['n_gt']} | {f(b['ap50'])} | {f(n['ap50'])} | {f(b['recall'])} | {f(n['recall'])} | "
                    f"{f(b['precision'])} | {f(n['precision'])} |")
    gate = [c for c in a.gate.split(",") if c and new[c]["ap50"] is not None and base[c]["ap50"] is not None]
    mins = {k: float(v) for k, v in (x.split("=") for x in a.min_ap50.split(",") if x)}
    rel_ok = all(new[c]["ap50"] >= base[c]["ap50"] - a.tol for c in gate)
    abs_ok = all(new[c]["ap50"] is not None and new[c]["ap50"] >= v for c, v in mins.items())
    passed = rel_ok and abs_ok
    txt = (f"# Gerbang {a.new} vs {a.base} ({a.split}, {len(imgs)} gambar)\n\n"
           f"Recall/presisi pada ambang runtime {conf_thr}. COCO dipetakan lewat alias runtime.\n\n"
           + "\n".join(rows)
           + f"\n\nGerbang: AP50 baru >= base - {a.tol} untuk {gate} ({'ya' if rel_ok else 'TIDAK'})"
           + (f"; AP50 mutlak {mins} ({'ya' if abs_ok else 'TIDAK'})" if mins else "")
           + f" -> **{'LULUS' if passed else 'GAGAL'}**\n"
           "Catatan: split test = data publik (label pseudo teacher COCO ikut di sini), bukan rekaman simulasi kita.\n")
    out = rpath(a.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(txt, encoding="utf-8")
    print(txt)


if __name__ == "__main__":
    main()
