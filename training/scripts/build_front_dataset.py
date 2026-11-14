"""Satu perintah: dataset publik -> dataset YOLO kamera DEPAN siap training + training/datasets/front.yaml.

Kelas (urutan = ID, Tech Spec bag. 6): person, forklift, pallet_jack, gse_vehicle, aircraft_part,
ladder_stand, obstacle. Parameter di training/configs/front_build.yaml.

Urutan:
  1. download   front_sources.py  -> data/front_raw/<source>   (label upstream + classes.txt)
  2. convert    class_map (training/datasets/class_maps/front_*.yaml) -> kelas kita;
                teacher-fill (YOLO11n COCO) untuk kelas yang TIDAK dilabel sumber (mis. person di LOCO),
                ditandai pseudo di manifest; gambar besar (Apron 4K) diperkecil -> data/front_work/merged
  3. split      per GRUP (sekuens/sesi/lokasi dari regex class map) + dHash near-duplicate,
                aturan tetap (split.fixed) dulu, sisanya hash stabil. Tidak ada frame satu sekuens
                di dua split (diassert).
  4. tulis      data/front/<version>/images|labels/<split>  (hardlink)
  5. augment    train saja: motion blur, noise, gain (dipakai ulang dari augment.py) + strip
                oklusi vertikal (tiang mast forklift)
  TIDAK ada preprocessing NIR: kamera depan adalah kamera RGB biasa.

Contoh:
    python training/scripts/build_front_dataset.py --limit 50          # smoke
    python training/scripts/build_front_dataset.py                     # penuh (GPU box)
    python training/scripts/build_front_dataset.py --skip-download     # pakai data/front_raw yang ada
"""
from __future__ import annotations

import argparse
import csv
import datetime as dt
import random
import re
import shutil
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
import front_sources  # noqa: E402
from common import Box, imread, imwrite, iou, list_images, load_yaml, read_yolo, rpath, stable_unit, write_yolo  # noqa: E402
from convert_to_yolo import ClassMap  # noqa: E402
from split import UF, dhash, link_or_copy  # noqa: E402

DEFAULT_CFG = rpath("training/configs/front_build.yaml")
SPLITS = ("train", "val", "test")
AUG_PREFIX = "aug"
NOMATCH = "__nomatch__"


# ------------------------------------------------------------------ grup
def group_of(m: ClassMap, cm: dict, source: str, stem: str) -> str:
    """Grup split. Jika regex grup ada tapi tidak cocok: on_no_match=train -> satu grup '<src>:__nomatch__'
    (diarahkan ke train oleh split.fixed) supaya frame video tidak pernah di-split per frame."""
    g = cm.get("group") or {}
    if m.whole_source or not m.group_regex or any(rx.search(stem) for rx in m.group_regex):
        return m.group(source, stem)
    mode = g.get("on_no_match", "stem")
    if mode == "error":
        raise ValueError(f"[{source}] stem '{stem}' tidak cocok regex grup {g.get('regex')}")
    return f"{source}:{NOMATCH}" if mode == "train" else f"{source}:{stem}"


# ------------------------------------------------------------------ teacher fill
def load_teacher(cfg: dict):
    from ultralytics import YOLO

    return YOLO(str(rpath(cfg["teacher_fill"]["weights"])))


def run_teacher(model, paths: list[Path], t: dict) -> dict[str, list[tuple[str, tuple, float]]]:
    out: dict[str, list] = {}
    for i in range(0, len(paths), int(t["batch"])):
        chunk = paths[i:i + int(t["batch"])]
        res = model.predict([str(p) for p in chunk], imgsz=t["imgsz"], conf=t["conf"], device=t.get("device"),
                            verbose=False)
        for p, r in zip(chunk, res):
            names = r.names
            out[str(p)] = [(names[int(c)], tuple(map(float, x)), float(s)) for c, x, s in
                           zip(r.boxes.cls.cpu().numpy(), r.boxes.xyxyn.cpu().numpy(), r.boxes.conf.cpu().numpy())]
    return out


def teacher_boxes(preds: list[tuple[str, tuple, float]], fill: list[str], existing: list[Box], ours: list[str],
                  tmap: dict[str, list[str]], dedupe_iou: float) -> list[Box]:
    """Prediksi teacher (nama COCO) -> kotak pseudo kelas kita, hanya untuk kelas di `fill`.
    Dibuang jika IoU >= dedupe_iou dengan kotak mana pun yang sudah ada (kelas apa pun): objek yang sudah
    dilabel sumber tidak boleh mendapat label kedua (mis. forklift LOCO yang oleh COCO disebut 'truck')."""
    rev = {cn: our for our, cns in tmap.items() if our in fill for cn in cns}
    out: list[Box] = []
    for cname, xyxy, _ in preds:
        our = rev.get(cname)
        if our is None:
            continue
        if max((iou(xyxy, b.xyxy()) for b in existing + out), default=0.0) < dedupe_iou:
            out.append(Box.from_xyxy(ours.index(our), *xyxy, pseudo=True))
    return out


# ------------------------------------------------------------------ convert
def convert_source(name: str, raw: Path, out: Path, cfg: dict, cm: dict, teacher, writer) -> int:
    ours = list(cfg["classes"])
    upstream = [x.strip() for x in (raw / "classes.txt").read_text(encoding="utf-8").splitlines() if x.strip()]
    m = ClassMap(cm, upstream, ours)
    cc, tf = cfg["convert"], cfg["teacher_fill"]
    labelled = {ours[i] for i in m.labelled_default}
    fill = [c for c in ((cm.get("derive") or {}).get("teacher_fill") or []) if c not in labelled]
    imgs = list_images(raw / "images")
    if not imgs:
        print(f"[{name}] tidak ada gambar di {raw}")
        return 0
    preds = run_teacher(teacher, imgs, tf) if (fill and teacher is not None) else {}
    print(f"[{name}] {len(imgs)} gambar; dilabel sumber: {sorted(labelled)}; teacher-fill: "
          f"{fill if preds else ('MATI' if fill else '-')}")
    n = nomatch = 0
    max_side = int(cc.get("max_side") or 0)
    for p in imgs:
        img = imread(p)
        if img is None:
            print(f"  lewati (tidak terbaca): {p.name}")
            continue
        H, W = img.shape[:2]
        boxes: list[Box] = []
        for b in read_yolo(raw / "labels" / f"{p.stem}.txt"):
            for t in m.map.get(b.cls, []):
                if min(b.w * W, b.h * H) >= cc["min_box_px"]:
                    boxes.append(Box(t, b.cx, b.cy, b.w, b.h))
        if fill and str(p) in preds:
            for nb in teacher_boxes(preds[str(p)], fill, boxes, ours, tf["map"], tf["dedupe_iou"]):
                if min(nb.w * W, nb.h * H) >= cc["min_box_px"]:
                    boxes.append(nb)
        if not boxes and cc.get("drop_empty_images"):
            continue
        if max_side and max(H, W) > max_side:
            import cv2
            r = max_side / max(H, W)
            img = cv2.resize(img, (round(W * r), round(H * r)), interpolation=cv2.INTER_AREA)
        stem = f"{name}__{p.stem}"
        dst = out / "images" / f"{stem}{cc['image_ext']}"
        if not imwrite(dst, img, cc["jpeg_quality"]):
            continue
        write_yolo(out / "labels" / f"{stem}.txt", boxes)
        grp = group_of(m, cm, name, p.stem)
        nomatch += grp.endswith(NOMATCH)
        writer.writerow([dst.name, name, grp, *[sum(b.cls == i for b in boxes) for i in range(len(ours))],
                         *[sum(b.cls == i and b.pseudo for b in boxes) for i in range(len(ours))],
                         "|".join(sorted(labelled))])
        n += 1
    if nomatch:
        print(f"  [peringatan] {nomatch} gambar {name} tidak cocok regex grup -> semua ke train. "
              f"Sesuaikan group.regex di class map dengan penamaan file sebenarnya.")
    return n


def convert_all(raw_root: Path, out: Path, cfg: dict, active: list[str], teacher) -> Path:
    ours = cfg["classes"]
    sources = load_yaml(rpath(cfg["paths"]["sources"]))["sources"]
    if out.exists():
        shutil.rmtree(out)
    (out / "images").mkdir(parents=True)
    (out / "labels").mkdir(parents=True)
    manifest = out / "manifest.csv"
    with open(manifest, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["file", "source", "group", *[f"n_{c}" for c in ours], *[f"pseudo_{c}" for c in ours], "labelled"])
        for name in active:
            d = raw_root / name
            if not (d / "classes.txt").exists():
                print(f"[{name}] belum ada data mentah di {d} -> dilewati")
                continue
            cm = load_yaml(rpath(cfg["paths"]["class_maps"]) / sources[name]["class_map"])
            convert_source(name, d, out, cfg, cm, teacher, w)
    return manifest


# ------------------------------------------------------------------ split
def assign(rows: list[dict], images_dir: Path | None, scfg: dict, seq_sources: set[str] | frozenset = frozenset()) -> dict[str, str]:
    """file -> split. Grup (regex class map) + dHash digabung (union-find); aturan split.fixed
    (regex pada id grup, urutan = prioritas) lalu hash stabil untuk sisanya.
    dHash TIDAK dijalankan untuk sumber sekuens (`seq_sources`, grup dari regex): frame dashcam/gudang
    yang mirip bisa merantai sekuens berbeda jadi satu grup raksasa; grupnya sudah ditentukan regex."""
    uf = UF()
    for r in rows:
        uf.find(r["group"])
    dh = scfg.get("dhash") or {}
    if dh.get("enabled") and images_dir is not None:
        size, maxh, bands = int(dh["size"]), int(dh["max_hamming"]), int(dh["bands"])
        bw = size * size // bands
        hashes = {}
        for r in rows:
            if r.get("source") in seq_sources:
                continue
            h = dhash(images_dir / r["file"], size)
            if h is not None:
                hashes[r["file"]] = (h, r["group"])
        buckets: dict[tuple, list[str]] = defaultdict(list)
        for f, (h, _) in hashes.items():
            for b in range(bands):
                buckets[(b, (h >> (b * bw)) & ((1 << bw) - 1))].append(f)
        merged = 0
        for files in buckets.values():
            for i in range(len(files)):
                hi, gi = hashes[files[i]]
                for j in range(i + 1, len(files)):
                    hj, gj = hashes[files[j]]
                    if bin(hi ^ hj).count("1") <= maxh and uf.find(gi) != uf.find(gj):
                        uf.union(gi, gj)
                        merged += 1
        print(f"dHash: {merged} penggabungan grup near-duplicate (dilewati untuk sumber sekuens: {sorted(seq_sources)})")
    members: dict[str, set[str]] = defaultdict(set)
    for r in rows:
        members[uf.find(r["group"])].add(r["group"])
    rules = [(re.compile(x["match"]), x["split"]) for x in (scfg.get("fixed") or [])]
    ratios = scfg["ratios"]
    cut_train, cut_val = ratios["train"], ratios["train"] + ratios["val"]
    root_split: dict[str, str] = {}
    for root, gs in members.items():
        hits = [sp for rx, sp in rules if any(rx.search(g) for g in gs)]
        hit = hits[0] if hits else None
        if len(set(hits)) > 1:
            print(f"[peringatan] grup gabungan {root} ({len(gs)} grup) cocok aturan split berbeda {hits} -> {hit}")
        if hit is None:
            u = stable_unit(root, int(scfg["seed"]))
            hit = "train" if u < cut_train else ("val" if u < cut_val else "test")
        root_split[root] = hit
    out = {r["file"]: root_split[uf.find(r["group"])] for r in rows}
    g2s = defaultdict(set)
    for r in rows:
        g2s[r["group"]].add(out[r["file"]])
    leaks = [g for g, s in g2s.items() if len(s) > 1]
    assert not leaks, f"grup bocor lintas split: {leaks[:5]}"
    size = Counter(uf.find(r["group"]) for r in rows)
    warn = int(scfg.get("warn_group_size") or 0)
    big = [(g, n) for g, n in size.most_common(5) if warn and n > warn]
    if big:
        print(f"[peringatan] grup akhir > {warn} gambar: {big} -> cek regex grup / max_hamming")
    by_src: dict[str, Counter] = defaultdict(Counter)
    for r in rows:
        by_src[r.get("source", "?")][r["group"]] += 1
    for s, c in sorted(by_src.items()):
        sizes = sorted(c.values())
        print(f"  grup {s:20s}: {len(sizes)} grup, median {sizes[len(sizes) // 2]} gambar, maks {sizes[-1]}")
        if s in seq_sources and len(sizes) > 1 and sizes[len(sizes) // 2] <= 1:
            print(f"  [peringatan] {s}: grup sekuens rata-rata 1 gambar -> regex grup kemungkinan memecah per frame")
    return out


def sequence_sources(cfg: dict, active: list[str]) -> set[str]:
    """Sumber yang grupnya ditentukan regex/whole_source (sekuens video, lokasi) -> tanpa dHash."""
    sources = load_yaml(rpath(cfg["paths"]["sources"]))["sources"]
    out = set()
    for name in active:
        g = load_yaml(rpath(cfg["paths"]["class_maps"]) / sources[name]["class_map"]).get("group") or {}
        if g.get("regex") or g.get("whole_source"):
            out.add(name)
    return out


def materialize(merged: Path, rows: list[dict], split_of: dict, out: Path, hardlink: bool) -> None:
    if out.exists():
        shutil.rmtree(out)
    for sp in SPLITS:
        (out / "images" / sp).mkdir(parents=True, exist_ok=True)
        (out / "labels" / sp).mkdir(parents=True, exist_ok=True)
    for r in rows:
        sp, stem = split_of[r["file"]], Path(r["file"]).stem
        link_or_copy(merged / "images" / r["file"], out / "images" / sp / r["file"], hardlink)
        link_or_copy(merged / "labels" / f"{stem}.txt", out / "labels" / sp / f"{stem}.txt", hardlink)
    with open(out / "splits.csv", "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()) + ["split"])
        w.writeheader()
        for r in rows:
            w.writerow({**r, "split": split_of[r["file"]]})


# ------------------------------------------------------------------ augmentasi
def mast_strip(img: np.ndarray, boxes: list[Box], mc: dict, rnd: random.Random) -> tuple[np.ndarray, list[Box]]:
    """Strip vertikal setinggi gambar, lebar 5-15% lebar gambar (tiang mast forklift, Tech Spec bag. 7).
    Kotak yang tertutup >= drop_covered_frac (fraksi lebarnya) dibuang: objeknya praktis tidak terlihat."""
    if rnd.random() >= float(mc["p"]):
        return img, boxes
    H, W = img.shape[:2]
    sw = max(1, int(round(rnd.uniform(*mc["width_frac"]) * W)))
    x0 = rnd.randint(0, max(0, W - sw))
    lo, hi = mc["fill_gray"]
    img[:, x0:x0 + sw] = rnd.randint(int(lo), int(hi))
    s0, s1 = x0 / W, (x0 + sw) / W
    keep = []
    for b in boxes:
        bx0, _, bx1, _ = b.xyxy()
        cover = max(0.0, min(bx1, s1) - max(bx0, s0)) / max(bx1 - bx0, 1e-9)
        if cover < float(mc["drop_covered_frac"]):
            keep.append(b)
    return img, keep


def augment_train(img_dir: Path, lbl_dir: Path, acfg: dict, jpeg_quality: int) -> int:
    from augment import build_pipeline

    copies = int(acfg["copies_per_image"])
    if copies <= 0:
        return 0
    pipe = build_pipeline(acfg)
    rnd = random.Random(int(acfg.get("seed", 0)))
    n = 0
    for p in list_images(img_dir):
        if p.name.startswith(AUG_PREFIX):
            continue
        img = imread(p)
        if img is None:
            continue
        boxes = read_yolo(lbl_dir / f"{p.stem}.txt")
        for k in range(copies):
            bb = [[min(max(b.cx, 0), 1), min(max(b.cy, 0), 1), min(max(b.w, 1e-6), 1), min(max(b.h, 1e-6), 1)] for b in boxes]
            r = pipe(image=img.copy(), bboxes=np.array(bb, dtype=np.float32).reshape(-1, 4), cls=[b.cls for b in boxes])
            ob = [Box(int(c), *map(float, xywh)) for xywh, c in zip(r["bboxes"], r["cls"])]
            im2, ob = mast_strip(r["image"], ob, acfg["mast_strip"], rnd)
            stem = f"{AUG_PREFIX}{k}_{p.stem}"
            imwrite(img_dir / f"{stem}.jpg", im2, jpeg_quality)
            write_yolo(lbl_dir / f"{stem}.txt", ob)
            n += 1
    print(f"augment: {n} gambar baru di {img_dir}")
    return n


# ------------------------------------------------------------------ laporan
def check_sources(rows: list[dict], classes: list[str]) -> None:
    """Instance per sumber x kelas. Berhenti jika sumber punya gambar tetapi NOL instance untuk semua kelas
    yang katanya dilabel (konversi gagal diam-diam: header CSV beda, salah nama di class map, zip tidak cocok)."""
    print(f"{'sumber':20s} {'img':>6s} " + " ".join(f"{c:>13s}" for c in classes))
    bad = []
    for s in sorted({r["source"] for r in rows}):
        rs = [r for r in rows if r["source"] == s]
        cnt = {c: sum(int(r[f"n_{c}"]) - int(r[f"pseudo_{c}"]) for r in rs) for c in classes}
        print(f"{s:20s} {len(rs):6d} " + " ".join(f"{cnt[c]:13d}" for c in classes))
        lab = [c for c in (rs[0]["labelled"].split("|") if rs[0]["labelled"] else []) if c]
        zero = [c for c in lab if cnt[c] == 0]
        if lab and len(zero) == len(lab):
            bad.append(s)
        elif zero:
            print(f"  [peringatan] {s}: kelas dilabel tanpa instance: {zero} (wajar hanya untuk --limit kecil)")
    if bad:
        sys.exit(f"sumber {bad} punya gambar tapi nol label untuk semua kelas 'labelled' -> cek format/class map")


def class_counts(lbl_dir: Path, nc: int) -> tuple[int, list[int]]:
    files = sorted(lbl_dir.glob("*.txt"))
    cnt = [0] * nc
    for f in files:
        for b in read_yolo(f):
            if 0 <= b.cls < nc:
                cnt[b.cls] += 1
    return len(files), cnt


def report(out: Path, rows: list[dict], split_of: dict, classes: list[str]) -> dict:
    print(f"{'split':6s} {'img':>6s} " + " ".join(f"{c:>13s}" for c in classes))
    counts = {}
    for sp in SPLITS:
        n, cnt = class_counts(out / "labels" / sp, len(classes))
        counts[sp] = {"images": n, **{c: v for c, v in zip(classes, cnt)}}
        print(f"{sp:6s} {n:6d} " + " ".join(f"{x:13d}" for x in cnt))
    per_src = Counter((r["source"], split_of[r["file"]]) for r in rows)
    for s in sorted({r["source"] for r in rows}):
        print(f"  {s:20s} " + " ".join(f"{sp}={per_src[(s, sp)]}" for sp in SPLITS))
    pseudo = {c: sum(int(r[f"pseudo_{c}"]) for r in rows) for c in classes}
    if any(pseudo.values()):
        print("label pseudo (teacher-fill): " + ", ".join(f"{c}={v}" for c, v in pseudo.items() if v))
    empty = [c for c in classes if counts["train"][c] == 0]
    if empty:
        print(f"[PERINGATAN] kelas tanpa instance di train: {empty} -> butuh data sintetis / rekaman tim "
              "(sources_front.yaml: synthetic, roboflow_*).")
    for sp in ("val", "test"):
        miss = [c for c in classes if counts[sp][c] == 0 and counts["train"][c] > 0]
        if miss:
            print(f"[peringatan] kelas tanpa instance di {sp}: {miss} (metrik kelas ini tidak terukur)")
    return counts


def write_dataset_yaml(path: Path, root: Path, names: list[str], version: str, note: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    body = {"path": root.resolve().as_posix(), "train": "images/train", "val": "images/val", "test": "images/test",
            "names": {i: n for i, n in enumerate(names)}}
    header = (f"# DIBUAT OTOMATIS oleh training/scripts/build_front_dataset.py ({version}, {note}).\n"
              "# Path absolut khusus mesin ini; jalankan ulang build di mesin lain (mis. RunPod).\n")
    path.write_text(header + yaml.safe_dump(body, sort_keys=False), encoding="utf-8")


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default=str(DEFAULT_CFG))
    ap.add_argument("--sources", default=None, help="ganti sources_front.yaml (mis. untuk smoke)")
    ap.add_argument("--limit", type=int, default=None, help="maks gambar per sumber (smoke)")
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--version", default=None)
    ap.add_argument("--skip-download", action="store_true")
    ap.add_argument("--keep-archives", action="store_true")
    ap.add_argument("--no-teacher-fill", action="store_true")
    ap.add_argument("--no-augment", action="store_true")
    ap.add_argument("--keep-work", action="store_true")
    ap.add_argument("--dataset-yaml", default=None)
    a = ap.parse_args()

    cfg = load_yaml(a.config)
    if a.sources:
        cfg["paths"]["sources"] = a.sources
    version = a.version or cfg["version"]
    raw, work = rpath(cfg["paths"]["raw"]), rpath(cfg["paths"]["work"])
    out = rpath(cfg["paths"]["out"]) / version
    sources = load_yaml(rpath(cfg["paths"]["sources"]))["sources"]
    from download_public import enabled_sources
    active = list(enabled_sources(sources, a.only))
    t0 = time.time()

    if not a.skip_download:
        print("== 1/5 download")
        front_sources.run(raw, a.limit, a.only, cfg, a.keep_archives)
    print("== 2/5 convert (class map + teacher-fill)")
    use_teacher = cfg["teacher_fill"].get("enabled") and not a.no_teacher_fill
    teacher = load_teacher(cfg) if use_teacher else None
    manifest = convert_all(raw, work / "merged", cfg, active, teacher)
    with open(manifest, encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    if not rows:
        sys.exit("tidak ada gambar terkonversi -> cek download / sources_front.yaml")
    print("== 3/5 split per grup (sekuens/sesi/lokasi)")
    check_sources(rows, cfg["classes"])
    split_of = assign(rows, work / "merged" / "images", cfg["split"], sequence_sources(cfg, active))
    print("== 4/5 tulis dataset")
    materialize(work / "merged", rows, split_of, out, bool(cfg["split"].get("hardlink", True)))
    if cfg["augment"]["enabled"] and not a.no_augment:
        print("== 5/5 augment (train)")
        augment_train(out / "images" / "train", out / "labels" / "train", cfg["augment"], cfg["convert"]["jpeg_quality"])
    else:
        print("== 5/5 augment dilewati")
    counts = report(out, rows, split_of, cfg["classes"])
    for sp in ("train", "val"):
        if counts[sp]["images"] == 0:
            sys.exit(f"split {sp} kosong -> cek split.fixed / ratios di front_build.yaml")

    note = f"limit={a.limit}" if a.limit else "build penuh"
    ds_yaml = rpath(a.dataset_yaml or cfg["paths"]["dataset_yaml"])
    write_dataset_yaml(ds_yaml, out, cfg["classes"], version, note)
    info = {"version": version, "built": dt.datetime.now().isoformat(timespec="seconds"), "limit": a.limit,
            "sources": sorted({r["source"] for r in rows}), "teacher_fill": bool(teacher is not None),
            "augment": bool(cfg["augment"]["enabled"] and not a.no_augment), "nir": False,
            "counts": counts, "classes": cfg["classes"],
            "note": "Hanya data publik. Bukan data hangar GMF; kelas ladder_stand/aircraft_part butuh sintetis/rekaman."}
    (out / "dataset_info.yaml").write_text(yaml.safe_dump(info, sort_keys=False), encoding="utf-8")
    if not a.keep_work:
        shutil.rmtree(work, ignore_errors=True)
    print(f"selesai dalam {time.time() - t0:.0f} s -> {ds_yaml}")


if __name__ == "__main__":
    main()
