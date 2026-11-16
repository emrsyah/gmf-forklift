"""Bandingkan beberapa bobot YOLO kabin pada split yang SAMA: TP/FP/FN, precision, recall per kelas
pada conf tetap (default = gate.recall_conf), alarm palsu phone di gambar tanpa ponsel, dan mAP50
(Ultralytics val) per subset.

Subset:
  legacy  = gambar dari split.anchor_sources (= split test cabin_v1, tidak pernah dilatih v1 maupun v2)
  new     = gambar dari sumber baru (ponsel Open Images, negatif keras COCO)
  all     = semuanya

Contoh (RunPod):
    python training/scripts/compare_cabin.py --weights cabin_v1=models/weights/cabin_v1_ref.pt \
        cabin_v2=training/runs/cabin_v2/weights/best.pt --out training/runs/cabin_v2/compare
"""
from __future__ import annotations

import argparse
import json
import sys
from collections import defaultdict
from pathlib import Path

import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import iou, list_images, load_build_cfg, load_yaml, read_yolo, rpath  # noqa: E402

TRAIN_CFG = rpath("training/configs/train_cabin.yaml")


def source_of(stem: str) -> str:
    return stem.split("__", 1)[0] if "__" in stem else "?"


def match(gt: list, pr: list, thr: float) -> tuple[int, int, int, list[bool]]:
    """Greedy per GT (sama dengan train_cabin.recall_at_conf). Kembalikan TP, FP, FN, flag TP per prediksi."""
    used: set[int] = set()
    for gb in gt:
        best, bj = 0.0, -1
        for j, pb in enumerate(pr):
            if j not in used:
                v = iou(gb, pb)
                if v > best:
                    best, bj = v, j
        if best >= thr:
            used.add(bj)
    tp = len(used)
    return tp, len(pr) - tp, len(gt) - tp, [j in used for j in range(len(pr))]


def evaluate(weights: Path, imgs: list[Path], lbl_dir: Path, names: list[str], conf: float, thr: float,
             imgsz: int, device, batch: int, legacy: set[str]) -> dict:
    from ultralytics import YOLO

    model = YOLO(str(weights))
    st = defaultdict(lambda: defaultdict(lambda: [0, 0, 0]))     # subset -> class -> [tp, fp, fn]
    neg = defaultdict(lambda: [0, 0, 0])                         # subset -> [n_neg_img, img_with_fp, n_fp]
    for k in range(0, len(imgs), batch):
        chunk = imgs[k:k + batch]
        res = model.predict([str(p) for p in chunk], conf=conf, imgsz=imgsz, device=device, verbose=False)
        for p, r in zip(chunk, res):
            src = source_of(p.stem)
            subsets = ("all", "legacy" if src in legacy else "new", f"src:{src}")
            gt = [(b.cls, b.xyxy()) for b in read_yolo(lbl_dir / f"{p.stem}.txt")]
            pr = [(int(c), tuple(x)) for c, x in zip(r.boxes.cls.cpu().numpy(), r.boxes.xyxyn.cpu().numpy())]
            for c, name in enumerate(names):
                tp, fp, fn, _ = match([x for cc, x in gt if cc == c], [x for cc, x in pr if cc == c], thr)
                for s in subsets:
                    v = st[s][name]
                    v[0] += tp; v[1] += fp; v[2] += fn
            ph = names.index("phone")
            if not any(cc == ph for cc, _ in gt):
                nfp = sum(1 for cc, _ in pr if cc == ph)
                for s in subsets:
                    neg[s][0] += 1
                    neg[s][1] += int(nfp > 0)
                    neg[s][2] += nfp
    out = {}
    for s, cls in st.items():
        out[s] = {"classes": {}, "phone_neg": dict(zip(("neg_images", "images_with_fp", "phone_fp"), neg[s]))}
        for name, (tp, fp, fn) in cls.items():
            out[s]["classes"][name] = {"tp": tp, "fp": fp, "fn": fn,
                                       "precision": tp / (tp + fp) if tp + fp else None,
                                       "recall": tp / (tp + fn) if tp + fn else None}
    return out


def map50(weights: Path, img_list: list[Path], names: list[str], imgsz: int, device, batch: int, tmp: Path) -> dict:
    from ultralytics import YOLO

    tmp.mkdir(parents=True, exist_ok=True)
    lst = tmp / "subset.txt"
    lst.write_text("".join(f"{p.resolve().as_posix()}\n" for p in img_list), encoding="utf-8")
    dy = tmp / "subset.yaml"
    dy.write_text(yaml.safe_dump({"path": tmp.resolve().as_posix(), "train": lst.resolve().as_posix(),
                                  "val": lst.resolve().as_posix(), "names": dict(enumerate(names))}), encoding="utf-8")
    m = YOLO(str(weights)).val(data=str(dy), split="val", imgsz=imgsz, batch=batch, device=device, plots=False,
                               verbose=False)
    idx = list(getattr(m.box, "ap_class_index", []))
    res = {"all": float(m.box.map50)}
    for c, n in enumerate(names):
        if c in idx:
            res[n] = float(m.box.class_result(idx.index(c))[2])
    return res


def fmt(v, d=3):
    return "—" if v is None else f"{v:.{d}f}"


def report(results: dict, maps: dict, names: list[str], conf: float, thr: float, split: str) -> str:
    L = [f"Split `{split}`, conf {conf}, IoU >= {thr}. Precision = TP/(TP+FP), recall = TP/(TP+FN).", ""]
    models = list(results)
    subsets = ["legacy", "new", "all"] + sorted(s for s in results[models[0]] if s.startswith("src:"))
    for s in subsets:
        if s not in results[models[0]]:
            continue
        L += [f"### Subset `{s}`", "", "| Kelas | " + " | ".join(f"{m} P | {m} R | {m} TP/FP/FN" for m in models) + " |",
              "|---|" + "---|---|---|" * len(models)]
        for n in names:
            cells = []
            for m in models:
                c = results[m][s]["classes"].get(n, {})
                cells += [fmt(c.get("precision")), fmt(c.get("recall")), f"{c.get('tp', 0)}/{c.get('fp', 0)}/{c.get('fn', 0)}"]
            L.append(f"| {n} | " + " | ".join(cells) + " |")
        neg = [results[m][s]["phone_neg"] for m in models]
        L.append("")
        L.append("Gambar tanpa ponsel (GT): " + "; ".join(
            f"{m}: {x['phone_fp']} phone FP pada {x['images_with_fp']}/{x['neg_images']} gambar "
            f"({x['phone_fp'] / x['neg_images']:.3f}/gambar)" if x["neg_images"] else f"{m}: —"
            for m, x in zip(models, neg)))
        if maps and s in maps.get(models[0], {}):
            L.append("")
            L.append("mAP50 (Ultralytics val): " + "; ".join(
                f"{m}: all {fmt(maps[m][s].get('all'))}, " + ", ".join(f"{n} {fmt(maps[m][s].get(n))}" for n in names)
                for m in models))
        L.append("")
    return "\n".join(L)


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--weights", nargs="+", required=True, help="nama=path.pt ...")
    ap.add_argument("--data", default=None, help="data yaml (default: data di train_cabin.yaml)")
    ap.add_argument("--split", default="test")
    ap.add_argument("--conf", type=float, default=None)
    ap.add_argument("--device", default=None)
    ap.add_argument("--batch", type=int, default=32)
    ap.add_argument("--no-map", action="store_true")
    ap.add_argument("--out", required=True, help="folder output (compare.md + compare.json)")
    a = ap.parse_args()

    tcfg = load_yaml(TRAIN_CFG)
    g = tcfg["gate"]
    conf = a.conf if a.conf is not None else float(g["recall_conf"])
    thr = float(g["match_iou"])
    imgsz = int(tcfg["train"]["imgsz"])
    data = rpath(a.data or tcfg["data"])
    dcfg = load_yaml(data)
    names = [dcfg["names"][i] for i in sorted(dcfg["names"])]
    root = Path(dcfg["path"])
    img_dir = root / dcfg[a.split]
    lbl_dir = root / dcfg[a.split].replace("images", "labels", 1)
    imgs = list_images(img_dir)
    legacy = set(load_build_cfg().get("split", {}).get("anchor_sources") or [])
    out = rpath(a.out)
    out.mkdir(parents=True, exist_ok=True)
    print(f"{len(imgs)} gambar di {img_dir}; legacy={sorted(legacy)}")

    results, maps = {}, {}
    subsets_imgs = {"all": imgs, "legacy": [p for p in imgs if source_of(p.stem) in legacy],
                    "new": [p for p in imgs if source_of(p.stem) not in legacy]}
    for spec in a.weights:
        name, path = spec.split("=", 1)
        w = rpath(path)
        print(f"== {name}: {w}")
        results[name] = evaluate(w, imgs, lbl_dir, names, conf, thr, imgsz, a.device, a.batch, legacy)
        if not a.no_map:
            maps[name] = {s: map50(w, li, names, imgsz, a.device, a.batch, out / f"_tmp_{s}")
                          for s, li in subsets_imgs.items() if li}
    md = report(results, maps, names, conf, thr, a.split)
    counts = {s: len(v) for s, v in subsets_imgs.items()}
    md = f"# Perbandingan bobot kabin\n\nJumlah gambar: {counts}\n\n" + md
    (out / "compare.md").write_text(md, encoding="utf-8")
    (out / "compare.json").write_text(json.dumps({"conf": conf, "iou": thr, "counts": counts,
                                                  "results": results, "map50": maps}, indent=1), encoding="utf-8")
    print(md)


if __name__ == "__main__":
    main()
