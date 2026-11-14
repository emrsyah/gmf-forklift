"""Auto-label dengan YOLO-World (prompt teks) -> label YOLO untuk dikoreksi di CVAT / Label Studio.

Alat opsional (Tech Spec bag. 7 langkah 3). Tidak dipanggil oleh build_cabin_dataset.py: label publik
yang ada tetap dipakai; auto-label untuk gambar baru tanpa label, atau mengisi kelas yang tidak
dilabel di suatu sumber (lalu WAJIB dikoreksi manual).

Output: <out>/labels/*.txt (YOLO, ID kelas kabin) + <out>/obj.names (urutan kelas untuk impor
"YOLO 1.1" di CVAT). Dengan --merge, kotak lama dari --labels dipertahankan dan hanya kotak baru
untuk --classes yang tidak tumpang-tindih yang ditambahkan.

Contoh:
    # label baru untuk folder gambar
    python training/scripts/autolabel.py --images data/new_frames --out data/autolabel/new_frames
    # isi kelas vest yang tidak dilabel di sumber ppe_jhboyo ds1 (lalu koreksi di CVAT)
    python training/scripts/autolabel.py --images data/raw_merged/images --labels data/raw_merged/labels \
        --merge --classes vest --out data/autolabel/jhboyo_vest
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import Box, iou, list_images, load_build_cfg, load_yaml, read_yolo, rpath, write_yolo  # noqa: E402

DEFAULT_CFG = Path(__file__).resolve().parents[1] / "configs" / "autolabel.yaml"


def build_model(acfg: dict, classes: list[str], keep: list[str]):
    from ultralytics import YOLOWorld

    w = rpath(acfg["model"])
    if not w.exists():
        import requests
        w.parent.mkdir(parents=True, exist_ok=True)
        r = requests.get(acfg["model_url"], timeout=300)
        r.raise_for_status()
        w.write_bytes(r.content)
    model = YOLOWorld(str(w))
    prompts, owner = [], []
    for cname in keep:
        for p in acfg["prompts"].get(cname, []):
            prompts.append(p)
            owner.append(classes.index(cname))
    model.set_classes(prompts)
    return model, owner


def run(images: Path, out: Path, acfg: dict, classes: list[str], keep: list[str],
        labels: Path | None = None, merge: bool = False, limit: int | None = None) -> int:
    model, owner = build_model(acfg, classes, keep)
    (out / "labels").mkdir(parents=True, exist_ok=True)
    (out / "obj.names").write_text("\n".join(classes) + "\n", encoding="utf-8")
    imgs = list_images(images)[:limit] if limit else list_images(images)
    n_new = 0
    for p in imgs:
        r = model.predict(str(p), imgsz=acfg["imgsz"], conf=acfg["conf"], iou=acfg["iou"],
                          device=acfg.get("device"), verbose=False)[0]
        new = [Box.from_xyxy(owner[int(c)], *map(float, xy), pseudo=True)
               for c, xy in zip(r.boxes.cls.cpu().numpy(), r.boxes.xyxyn.cpu().numpy())]
        old = read_yolo(labels / f"{p.stem}.txt") if (merge and labels) else []
        kept = list(old)
        for b in new:
            if any(o.cls == b.cls and iou(o.xyxy(), b.xyxy()) >= acfg["merge_iou"] for o in kept):
                continue
            kept.append(b)
            n_new += 1
        write_yolo(out / "labels" / f"{p.stem}.txt", kept)
    print(f"autolabel: {len(imgs)} gambar, {n_new} kotak baru -> {out} (koreksi di CVAT sebelum dipakai)")
    return n_new


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--images", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--labels", default=None, help="label lama (untuk --merge)")
    ap.add_argument("--merge", action="store_true")
    ap.add_argument("--classes", nargs="*", default=None, help="kelas kabin yang di-auto-label (default: semua)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--config", default=str(DEFAULT_CFG))
    ap.add_argument("--build-config", default=None)
    a = ap.parse_args()
    acfg = load_yaml(a.config)
    classes = load_build_cfg(a.build_config)["classes"]
    keep = a.classes or classes
    bad = [c for c in keep if c not in classes]
    if bad:
        ap.error(f"kelas tidak dikenal: {bad}")
    run(rpath(a.images), rpath(a.out), acfg, classes, keep, rpath(a.labels) if a.labels else None, a.merge, a.limit)


if __name__ == "__main__":
    main()
