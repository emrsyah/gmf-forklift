"""Gabungkan data mentah semua sumber menjadi satu dataset YOLO dengan 5 kelas kabin.

Langkah per sumber:
  1. Petakan kelas upstream -> kelas kita (training/datasets/class_maps/<source>.yaml).
  2. (opsional) Isi label yang tidak dilabel upstream dengan YOLO pose:
       - torso dari keypoint bahu-pinggul,
       - head_no_helmet dari keypoint wajah (hanya jika tidak tertutup helmet/head yang ada).
     Label hasil isi ditandai "pseudo" di manifest.
  3. Fallback torso dari kotak person (jika ada dan belum ada torso di dalamnya).
  4. Simpan gambar (JPG, warna asli) + label + manifest.csv (sumber, grup, jumlah per kelas).

Contoh:
    python training/scripts/convert_to_yolo.py --raw data/raw --out data/cabin_work/merged
"""
from __future__ import annotations

import argparse
import csv
import re
import shutil
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import (Box, class_names, imread, imwrite, ioa, iou, list_images, load_build_cfg,  # noqa: E402
                    load_yaml, read_yolo, rpath, write_yolo)

# indeks keypoint COCO-17
NOSE, LEYE, REYE, LEAR, REAR = 0, 1, 2, 3, 4
LSH, RSH, LHIP, RHIP = 5, 6, 11, 12


# ------------------------------------------------------------------ class map
class ClassMap:
    def __init__(self, cm: dict, upstream: list[str], ours: list[str]):
        self.ours = ours
        cid = {n: i for i, n in enumerate(ours)}
        self.map: dict[int, list[int]] = {}
        self.person_ids: set[int] = set()
        m = cm.get("map", {}) or {}
        person_names = set((cm.get("derive", {}) or {}).get("torso_from_person", []) or [])
        for uid, uname in enumerate(upstream):
            self.map[uid] = [cid[t] for t in (m.get(uname) or [])]
            if uname in person_names:
                self.person_ids.add(uid)
        unknown = [k for k in m if k not in upstream]
        if unknown:
            print(f"  [peringatan] nama di class_map tidak ada di upstream: {unknown}")
        self.labelled_default = {cid[n] for n in cm.get("labelled", [])}
        self.subsets = [(re.compile(s["match"]), {cid[n] for n in s.get("labelled", [])})
                        for s in (cm.get("subsets") or [])]
        d = cm.get("derive", {}) or {}
        self.torso_from_pose = bool(d.get("torso_from_pose", False))
        self.head_from_pose = bool(d.get("head_from_pose", False))
        g = cm.get("group", {}) or {}
        self.group_regex = [re.compile(r) for r in (g.get("regex") or [])]
        self.whole_source = bool(g.get("whole_source", False))

    def labelled(self, stem: str) -> set[int]:
        for rx, lab in self.subsets:
            if rx.search(stem):
                return lab
        return self.labelled_default

    def group(self, source: str, stem: str) -> str:
        if self.whole_source:
            return f"{source}:*"
        for rx in self.group_regex:
            m = rx.search(stem)
            if m:
                return f"{source}:{m.group('g')}"
        return f"{source}:{stem}"


# ------------------------------------------------------------------ geometri dari keypoint
def torso_from_kpts(xy: np.ndarray, cf: np.ndarray, W: int, H: int, c: dict, kp_conf: float):
    """Kotak torso (piksel) dari bahu (+pinggul jika terlihat)."""
    if cf[LSH] < kp_conf or cf[RSH] < kp_conf:
        return None
    sh = xy[[LSH, RSH]]
    sw = abs(sh[0, 0] - sh[1, 0])
    if sw < c["min_shoulder_width_px"]:
        return None
    hips = [xy[i] for i in (LHIP, RHIP) if cf[i] >= kp_conf]
    y_sh = float(sh[:, 1].mean())
    if hips:
        hp = np.array(hips)
        y_hip = float(hp[:, 1].mean())
        xs = np.concatenate([sh[:, 0], hp[:, 0]])
        th = max(y_hip - y_sh, 1.0)
        y0, y1 = y_sh - c["pad_top"] * th, y_hip + c["pad_bottom"] * th
    else:
        xs = sh[:, 0]
        th = c["height_per_shoulder_width"] * sw
        y0, y1 = y_sh - c["pad_top"] * th, y_sh + th
    x0, x1 = float(xs.min()) - c["pad_x"] * sw, float(xs.max()) + c["pad_x"] * sw
    x0, x1 = max(0.0, x0), min(float(W), x1)
    y0, y1 = max(0.0, y0), min(float(H), y1)
    if x1 - x0 < 2 or y1 - y0 < 2:
        return None
    return x0, y0, x1, y1


def head_from_kpts(xy: np.ndarray, cf: np.ndarray, W: int, H: int, c: dict, kp_conf: float):
    vis = [i for i in (NOSE, LEYE, REYE, LEAR, REAR) if cf[i] >= kp_conf]
    if len(vis) < c["min_points"]:
        return None
    if cf[LEAR] >= kp_conf and cf[REAR] >= kp_conf:
        w = c["width_per_ear_dist"] * abs(xy[LEAR, 0] - xy[REAR, 0])
    elif cf[LEYE] >= kp_conf and cf[REYE] >= kp_conf:
        w = c["width_per_eye_dist"] * abs(xy[LEYE, 0] - xy[REYE, 0])
    else:
        return None
    if w < c["min_width_px"]:
        return None
    h = c["height_per_width"] * w
    cx = float(xy[vis, 0].mean())
    cy = float(xy[vis, 1].mean()) - c["shift_up"] * h
    x0, y0, x1, y1 = max(0.0, cx - w / 2), max(0.0, cy - h / 2), min(float(W), cx + w / 2), min(float(H), cy + h / 2)
    if x1 - x0 < 2 or y1 - y0 < 2:
        return None
    return x0, y0, x1, y1


# ------------------------------------------------------------------ pose model
def load_pose(cfg: dict):
    from ultralytics import YOLO

    w = rpath(cfg["paths"]["pose_weights"])
    if not w.exists():
        import requests
        w.parent.mkdir(parents=True, exist_ok=True)
        print(f"  unduh bobot pose -> {w}")
        r = requests.get(cfg["paths"]["pose_weights_url"], timeout=120)
        r.raise_for_status()
        w.write_bytes(r.content)
    return YOLO(str(w))


def run_pose(model, paths: list[Path], pcfg: dict) -> dict[str, list[tuple[np.ndarray, np.ndarray]]]:
    out: dict[str, list] = {}
    for i in range(0, len(paths), pcfg["batch"]):
        chunk = paths[i:i + pcfg["batch"]]
        res = model.predict([str(p) for p in chunk], imgsz=pcfg["imgsz"], conf=pcfg["conf"],
                            device=pcfg.get("device"), verbose=False)
        for p, r in zip(chunk, res):
            people = []
            if r.keypoints is not None and r.keypoints.xy is not None and len(r.keypoints.xy):
                xy = r.keypoints.xy.cpu().numpy()
                cf = r.keypoints.conf.cpu().numpy() if r.keypoints.conf is not None else np.ones(xy.shape[:2])
                people = [(xy[k], cf[k]) for k in range(len(xy))]
            out[str(p)] = people
    return out


_SCREEN_MODELS: dict = {}


def screen_images(name: str, imgs: list[Path], sc: dict | None, cfg: dict, pcfg: dict) -> set[str]:
    """Gambar yang dibuang karena model pembanding melihat kelas yang tidak dilabel sumber ini.

    Masalah label parsial: sumber yang hanya melabel `phone` (Open Images, negatif COCO) bisa berisi
    helm/rompi tanpa label -> model belajar helm/rompi = latar. Gambar seperti itu dibuang.
    Input model pembanding diberi preprocessing yang sama dengan training (grayscale+CLAHE).
    """
    if not sc:
        return set()
    from ultralytics import YOLO

    from common import clahe_params
    from src.preprocess import simulate_nir

    w = rpath(sc["weights"])
    if not w.exists():
        raise FileNotFoundError(f"[{name}] bobot screen tidak ada: {w} (salin cabin_v1 ke path ini)")
    model = _SCREEN_MODELS.setdefault(str(w), YOLO(str(w)))
    want = set(sc["classes"])
    clip, tile = clahe_params(cfg)
    drop: set[str] = set()
    bs = int(pcfg["batch"])
    for i in range(0, len(imgs), bs):
        chunk = imgs[i:i + bs]
        arrs, keep = [], []
        for p in chunk:
            im = imread(p)
            if im is not None:
                arrs.append(simulate_nir(im, clip, tile))
                keep.append(p)
        if not arrs:
            continue
        res = model.predict(arrs, imgsz=pcfg["imgsz"], conf=float(sc["conf"]), device=pcfg.get("device"), verbose=False)
        for p, r in zip(keep, res):
            names = {model.names[int(c)] for c in r.boxes.cls.cpu().numpy()}
            if names & want:
                drop.add(str(p))
    print(f"[{name}] screen {sorted(want)} conf>={sc['conf']}: {len(drop)} dari {len(imgs)} gambar dibuang")
    return drop


# ------------------------------------------------------------------ konversi
def convert_source(name: str, raw: Path, out: Path, cfg: dict, cm: dict, pose_model, writer, stats: dict) -> int:
    ours = class_names(cfg)
    C = {n: i for i, n in enumerate(ours)}
    upstream = [x.strip() for x in (raw / "classes.txt").read_text(encoding="utf-8").splitlines() if x.strip()]
    m = ClassMap(cm, upstream, ours)
    ccfg, pcfg = cfg["convert"], cfg["pose_fill"]
    imgs = list_images(raw / "images")
    if not imgs:
        print(f"[{name}] tidak ada gambar di {raw}")
        return 0

    need_pose = []
    for p in imgs:
        lab = m.labelled(p.stem)
        if pcfg.get("enabled") and ((m.torso_from_pose and C["torso"] not in lab) or
                                    (m.head_from_pose and C["head_no_helmet"] not in lab)):
            need_pose.append(p)
    poses = run_pose(pose_model, need_pose, pcfg) if (need_pose and pose_model is not None) else {}
    print(f"[{name}] {len(imgs)} gambar, pose-fill pada {len(poses)}")
    dropped = screen_images(name, imgs, cm.get("screen"), cfg, pcfg)

    n = 0
    for p in imgs:
        if str(p) in dropped:
            continue
        img = imread(p)
        if img is None:
            print(f"  lewati (tidak terbaca): {p.name}")
            continue
        H, W = img.shape[:2]
        lab = m.labelled(p.stem)
        boxes: list[Box] = []
        persons: list[tuple] = []
        for b in read_yolo(raw / "labels" / f"{p.stem}.txt"):
            if b.cls in m.person_ids:
                persons.append(b.xyxy())
            for t in m.map.get(b.cls, []):
                nb = Box(t, b.cx, b.cy, b.w, b.h)
                if min(nb.w * W, nb.h * H) >= ccfg["min_box_px"]:
                    boxes.append(nb)

        # --- isi dari pose
        native_torso = [b.xyxy() for b in boxes if b.cls == C["torso"]]
        hat_heads = [b.xyxy() for b in boxes if b.cls in (C["helmet"], C["head_no_helmet"])]
        for xy, cf in poses.get(str(p), []):
            if m.torso_from_pose and C["torso"] not in lab:
                t = torso_from_kpts(xy, cf, W, H, pcfg["torso"], pcfg["kp_conf"])
                if t is not None:
                    tn = (t[0] / W, t[1] / H, t[2] / W, t[3] / H)
                    best = max((iou(tn, nt) for nt in native_torso), default=0.0)
                    if best > pcfg["stats_same_person_iou"]:  # orang yang sama: geometri torso pose vs label
                        stats.setdefault("pose_vs_label_torso_iou", []).append(best)
                    if best < pcfg["dedupe"]["torso_iou"]:
                        boxes.append(Box.from_xyxy(C["torso"], *tn, pseudo=True))
                        native_torso.append(tn)
            if m.head_from_pose and C["head_no_helmet"] not in lab:
                h = head_from_kpts(xy, cf, W, H, pcfg["head"], pcfg["kp_conf"])
                if h is not None:
                    hn = (h[0] / W, h[1] / H, h[2] / W, h[3] / H)
                    if max((ioa(hn, hb) for hb in hat_heads), default=0.0) < pcfg["dedupe"]["head_ioa"]:
                        boxes.append(Box.from_xyxy(C["head_no_helmet"], *hn, pseudo=True))
                        hat_heads.append(hn)

        # --- fallback torso dari kotak person
        pt = cfg["pose_fill"]["person_torso"]
        if C["torso"] not in lab:
            for (x0, y0, x1, y1) in persons:
                if any(x0 <= (t[0] + t[2]) / 2 <= x1 and y0 <= (t[1] + t[3]) / 2 <= y1 for t in native_torso):
                    continue
                pw, ph = x1 - x0, y1 - y0
                if ph < pt["min_height_frac"] or pw < pt["min_width_frac"] or ph / max(pw, 1e-6) < pt["min_aspect_h_w"]:
                    continue  # potongan kecil / sempit (tangan, lengan) bukan badan
                tn = (x0 + pt["x_margin"] * pw, y0 + pt["y0"] * ph, x1 - pt["x_margin"] * pw, y0 + pt["y1"] * ph)
                boxes.append(Box.from_xyxy(C["torso"], *tn, pseudo=True))
                native_torso.append(tn)
                key = f"person_fallback_torso[{name}]"
                stats[key] = stats.get(key, 0) + 1

        # --- statistik IoU torso-vs-vest pada label (cek aturan no_vest IoU<0,3)
        vests = [b.xyxy() for b in boxes if b.cls == C["vest"]]
        for b in boxes:
            if b.cls == C["torso"] and b.pseudo and vests:
                stats.setdefault("pseudo_torso_vs_vest_iou", []).append(max(iou(b.xyxy(), v) for v in vests))

        stem = f"{name}__{p.stem}"
        dst_img = out / "images" / f"{stem}{ccfg['image_ext']}"
        if not imwrite(dst_img, img, ccfg["jpeg_quality"]):
            continue
        if not boxes and ccfg.get("drop_empty_images"):
            dst_img.unlink(missing_ok=True)
            continue
        write_yolo(out / "labels" / f"{stem}.txt", boxes)
        counts = [sum(1 for b in boxes if b.cls == i) for i in range(len(ours))]
        pseudo = [sum(1 for b in boxes if b.cls == i and b.pseudo) for i in range(len(ours))]
        writer.writerow([dst_img.name, name, m.group(name, p.stem), *counts, *pseudo,
                         "|".join(ours[i] for i in sorted(lab))])
        n += 1
    return n


def summarize(stats: dict) -> None:
    for k in ("pose_vs_label_torso_iou", "pseudo_torso_vs_vest_iou"):
        v = np.array(stats.get(k, []))
        if v.size:
            print(f"  {k}: n={v.size} median={np.median(v):.2f} frac>=0.3={np.mean(v >= 0.3):.2f}")
    for k, v in sorted(stats.items()):
        if k.startswith("person_fallback_torso"):
            print(f"  {k}: {v}")


def run(raw_root: Path, out: Path, cfg: dict, only: list[str] | None = None) -> Path:
    ours = class_names(cfg)
    sources = load_yaml(rpath(cfg["paths"]["sources"]))["sources"]
    if out.exists():
        shutil.rmtree(out)
    (out / "images").mkdir(parents=True)
    (out / "labels").mkdir(parents=True)
    pose_model = load_pose(cfg) if cfg["pose_fill"].get("enabled") else None
    manifest = out / "manifest.csv"
    stats: dict = {}
    total = 0
    with open(manifest, "w", newline="", encoding="utf-8") as f:
        w = csv.writer(f)
        w.writerow(["file", "source", "group", *[f"n_{c}" for c in ours], *[f"pseudo_{c}" for c in ours], "labelled"])
        from download_public import enabled_sources
        active = enabled_sources(sources, only)
        for name in sorted(p.name for p in raw_root.iterdir() if p.is_dir()):
            if name not in active:
                print(f"[{name}] tidak aktif di sources.yaml (enabled/--only) -> dilewati")
                continue
            cm = load_yaml(rpath(cfg["paths"]["class_maps"]) / sources[name]["class_map"])
            total += convert_source(name, raw_root / name, out, cfg, cm, pose_model, w, stats)
    print(f"total {total} gambar -> {out}")
    summarize(stats)
    return manifest


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--raw", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--only", nargs="*", default=None)
    ap.add_argument("--no-pose-fill", action="store_true")
    ap.add_argument("--config", default=None)
    a = ap.parse_args()
    cfg = load_build_cfg(a.config)
    if a.no_pose_fill:
        cfg["pose_fill"]["enabled"] = False
    raw = rpath(a.raw or cfg["paths"]["raw"])
    out = rpath(a.out or (Path(cfg["paths"]["work"]) / "merged"))
    run(raw, out, cfg, a.only)


if __name__ == "__main__":
    main()
