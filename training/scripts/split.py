"""Split train/val/test per GRUP (bukan acak per frame).

Grup = kunci dari class_map (frame video / salinan augmentasi Roboflow / sesi / satu sumber),
lalu digabung lagi dengan dHash: gambar yang hampir identik (Hamming <= max_hamming) masuk grup
yang sama walaupun namanya beda. Split ditentukan oleh hash stabil dari id grup, sehingga
menambah sumber baru tidak mengacak ulang split lama.

Untuk data publik tidak ada id orang. Aturan "test = orang di luar training" (Tech Spec bag. 7)
hanya bisa dipenuhi dengan rekaman orang luar tim; split ini hanya mencegah kebocoran duplikat.

Contoh:
    python training/scripts/split.py --merged data/cabin_work/merged --out data/cabin_work/split
"""
from __future__ import annotations

import argparse
import csv
import os
import shutil
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import imread, load_build_cfg, rpath, stable_unit  # noqa: E402

SPLITS = ("train", "val", "test")


def dhash(path: Path, size: int) -> int | None:
    img = imread(path)
    if img is None:
        return None
    g = cv2.resize(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), (size + 1, size), interpolation=cv2.INTER_AREA)
    bits = (g[:, 1:] > g[:, :-1]).flatten()
    return int("".join("1" if b else "0" for b in bits), 2)


class UF:
    def __init__(self):
        self.p: dict[str, str] = {}

    def find(self, x: str) -> str:
        self.p.setdefault(x, x)
        while self.p[x] != x:
            self.p[x] = self.p[self.p[x]]
            x = self.p[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra != rb:
            self.p[max(ra, rb)] = min(ra, rb)


def assign(rows: list[dict], images_dir: Path, cfg: dict) -> dict[str, str]:
    scfg = cfg["split"]
    uf = UF()
    for r in rows:
        uf.find(r["group"])
    dh = scfg.get("dhash", {})
    if dh.get("enabled"):
        size, maxh, bands = int(dh["size"]), int(dh["max_hamming"]), int(dh["bands"])
        nbits = size * size
        bw = nbits // bands
        hashes = {}
        for r in rows:
            h = dhash(images_dir / r["file"], size)
            if h is not None:
                hashes[r["file"]] = (h, r["group"])
        buckets: dict[tuple, list[str]] = defaultdict(list)
        for f, (h, _) in hashes.items():
            for b in range(bands):
                buckets[(b, (h >> (b * bw)) & ((1 << bw) - 1))].append(f)
        merged = 0
        for files in buckets.values():
            if len(files) < 2:
                continue
            for i in range(len(files)):
                hi, gi = hashes[files[i]]
                for j in range(i + 1, len(files)):
                    hj, gj = hashes[files[j]]
                    if bin(hi ^ hj).count("1") <= maxh and uf.find(gi) != uf.find(gj):
                        uf.union(gi, gj)
                        merged += 1
        print(f"dHash: {merged} penggabungan grup near-duplicate")
    ratios = scfg["ratios"]
    cut_train = ratios["train"]
    cut_val = ratios["train"] + ratios["val"]
    out = {}
    for r in rows:
        u = stable_unit(uf.find(r["group"]), scfg["seed"])
        out[r["file"]] = "train" if u < cut_train else ("val" if u < cut_val else "test")
    # cek kebocoran: satu grup akhir hanya di satu split
    g2s = defaultdict(set)
    for r in rows:
        g2s[uf.find(r["group"])].add(out[r["file"]])
    leaks = [g for g, s in g2s.items() if len(s) > 1]
    assert not leaks, f"grup bocor lintas split: {leaks[:5]}"
    print(f"{len(g2s)} grup akhir dari {len(rows)} gambar")
    size = Counter(uf.find(r["group"]) for r in rows)
    top = size.most_common(5)
    print("grup terbesar: " + ", ".join(f"{g} ({n} img, {next(iter(g2s[g]))})" for g, n in top))
    warn = int(scfg.get("warn_group_size", 0))
    big = [(g, n) for g, n in top if warn and n > warn]
    if big:
        print(f"[peringatan] grup > {warn} gambar (kemungkinan rantai dHash / whole_source): {big} "
              "-> cek max_hamming atau regex grup")
    return out


def link_or_copy(src: Path, dst: Path, hardlink: bool) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    if hardlink:
        try:
            os.link(src, dst)
            return
        except OSError:
            pass
    shutil.copy2(src, dst)


def run(merged: Path, out: Path, cfg: dict) -> Path:
    with open(merged / "manifest.csv", encoding="utf-8") as f:
        rows = list(csv.DictReader(f))
    split_of = assign(rows, merged / "images", cfg)
    if out.exists():
        shutil.rmtree(out)
    hl = bool(cfg["split"].get("hardlink", True))
    for r in rows:
        sp = split_of[r["file"]]
        stem = Path(r["file"]).stem
        link_or_copy(merged / "images" / r["file"], out / sp / "images" / r["file"], hl)
        link_or_copy(merged / "labels" / f"{stem}.txt", out / sp / "labels" / f"{stem}.txt", hl)
    with open(out / "splits.csv", "w", newline="", encoding="utf-8") as f:
        fields = list(rows[0].keys()) + ["split"] if rows else ["file", "split"]
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        for r in rows:
            w.writerow({**r, "split": split_of[r["file"]]})
    report(rows, split_of, cfg)
    return out


def report(rows: list[dict], split_of: dict, cfg: dict) -> None:
    classes = cfg["classes"]
    print(f"{'split':6s} {'img':>6s} " + " ".join(f"{c:>15s}" for c in classes))
    for sp in SPLITS:
        rs = [r for r in rows if split_of[r["file"]] == sp]
        cnt = [sum(int(r[f"n_{c}"]) for r in rs) for c in classes]
        print(f"{sp:6s} {len(rs):6d} " + " ".join(f"{x:15d}" for x in cnt))
    per_src = Counter((r["source"], split_of[r["file"]]) for r in rows)
    for s in sorted({r["source"] for r in rows}):
        print(f"  {s:28s} " + " ".join(f"{sp}={per_src[(s, sp)]}" for sp in SPLITS))


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--merged", default=None)
    ap.add_argument("--out", default=None)
    ap.add_argument("--config", default=None)
    a = ap.parse_args()
    cfg = load_build_cfg(a.config)
    work = Path(cfg["paths"]["work"])
    run(rpath(a.merged or work / "merged"), rpath(a.out or work / "split"), cfg)


if __name__ == "__main__":
    main()
