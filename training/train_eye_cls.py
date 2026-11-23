"""A9: classifier mata terbuka/tertutup dari MRL Eye Dataset (inframerah, 37 orang), split PER ORANG.

Nama file MRL: s0001_00001_0_0_0_0_0_01.png = subjek, id, gender, kacamata, keadaan mata (0 tertutup, 1 terbuka),
pantulan, cahaya, sensor. Label kita: 1 = tertutup.
Augmentasi: kontras/gamma/noise + perkecil-perbesar (mata pengemudi di kamera kabin hanya ~12-25 px).
Lisensi: lihat training/datasets/LICENSES.md (tanpa lisensi eksplisit -> hanya PoC internal).

    python training/train_eye_cls.py --raw data/raw_eye --out training/runs/eye_cls_v1 [--epochs 12]
Keluaran: eye.onnx, report.json/md (test = orang yang tidak ada di train/val).
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "training" / "scripts"))
from tiny_cls import pick_threshold, predict, stats, train  # noqa: E402


def subject_split(subjects: list[str]) -> dict:
    """Deterministik per orang: tiap orang ke-6 (mulai 0) -> test, (mulai 3) -> val, sisanya train."""
    s = sorted(subjects)
    return {x: "test" if i % 6 == 0 else "val" if i % 6 == 3 else "train" for i, x in enumerate(s)}


def load(raw: Path, size: int, max_per_subject: int, seed: int):
    files = sorted(raw.rglob("s*_*.png"))
    by_subj: dict[str, list[Path]] = {}
    for f in files:
        by_subj.setdefault(f.stem.split("_")[0], []).append(f)
    split = subject_split(list(by_subj))
    rng = np.random.default_rng(seed)
    data = {k: ([], [], []) for k in ("train", "val", "test")}
    for subj, fs in by_subj.items():
        if max_per_subject and len(fs) > max_per_subject:
            fs = [fs[i] for i in sorted(rng.choice(len(fs), max_per_subject, replace=False))]
        X, y, m = data[split[subj]]
        for f in fs:
            p = f.stem.split("_")
            if len(p) < 8:
                continue
            img = cv2.imread(str(f), cv2.IMREAD_GRAYSCALE)
            if img is None:
                continue
            X.append(cv2.resize(img, (size, size), interpolation=cv2.INTER_AREA))
            y.append(1 - int(p[4]))
            m.append({"subject": subj, "glasses": int(p[3]), "light": int(p[6]), "sensor": p[7]})
    return {k: (np.stack(v[0]), np.array(v[1]), v[2]) for k, v in data.items() if v[0]}, split


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", default="data/raw_eye")
    ap.add_argument("--out", default="training/runs/eye_cls_v1")
    ap.add_argument("--size", type=int, default=32)
    ap.add_argument("--max-per-subject", type=int, default=2500, help="batas gambar per orang (0 = semua)")
    ap.add_argument("--epochs", type=int, default=12)
    ap.add_argument("--blur-min-scale", type=float, default=0.4, help="perkecil acak sampai skala ini")
    ap.add_argument("--max-fpr", type=float, default=0.03, help="maks. mata terbuka dinilai tertutup di val")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()
    out = ROOT / a.out
    t0 = time.time()
    data, split = load(ROOT / a.raw, a.size, a.max_per_subject, a.seed)
    counts = {k: [len(v[1]), int(v[1].sum())] for k, v in data.items()}
    print("counts [n, tertutup]:", counts, f"load {time.time() - t0:.0f} s", flush=True)
    Xtr, ytr, _ = data["train"]
    Xva, yva, _ = data["val"]
    Xte, yte, mte = data["test"]
    net, best = train("eye", Xtr, ytr, Xva, yva, a.epochs, out, a.seed, blocks=3, width=16,
                      blur_max_scale=a.blur_min_scale)
    pv, pt = predict(net, Xva), predict(net, Xte)
    thr = pick_threshold(yva, pv, a.max_fpr)
    rep = {"input_px": a.size, "counts": counts, "val_best_bal_acc": best, "thr": thr,
           "test": stats(yte, pt >= thr),
           "test_subjects": sorted(s for s, v in split.items() if v == "test")}
    # per kondisi (kacamata / cahaya buruk) dan pada mata kecil (perkecil ke 14 px lalu perbesar)
    for key, val, nm in (("glasses", 1, "test_glasses"), ("light", 0, "test_bad_light")):
        idx = np.array([m[key] == val for m in mte])
        if idx.any():
            rep[nm] = stats(yte[idx], pt[idx] >= thr)
    small = np.stack([cv2.resize(cv2.resize(x, (14, 14), interpolation=cv2.INTER_AREA), (a.size, a.size))
                      for x in Xte])
    rep["test_small_14px"] = stats(yte, predict(net, small) >= thr)
    (out / "report.json").write_text(json.dumps(rep, indent=1))
    lines = ["# Classifier mata (A9) — MRL Eye, test = orang di luar train/val", "",
             f"Ambang {thr} (dipilih di val, mata terbuka dinilai tertutup <= {a.max_fpr:.0%}).", "",
             "| Subset | Recall tertutup | Terbuka -> tertutup | Bal-acc | TP/FP/FN/TN |", "|---|---|---|---|---|"]
    for k in ("test", "test_glasses", "test_bad_light", "test_small_14px"):
        if k in rep:
            s = rep[k]
            lines.append(f"| {k} | {s['recall_pos']:.3f} | {s['false_pos_rate']:.3f} | {s['bal_acc']:.3f} | "
                         f"{s['tp']}/{s['fp']}/{s['fn']}/{s['tn']} |")
    (out / "report.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    print("\n".join(lines))


if __name__ == "__main__":
    main()
