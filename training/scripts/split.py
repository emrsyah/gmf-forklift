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


def _hashes(rows: list[dict], images_dir: Path, size: int) -> dict[str, tuple[int, str]]:
    out = {}
    for r in rows:
        h = dhash(images_dir / r["file"], size)
        if h is not None:
            out[r["file"]] = (h, r["group"])
    return out


def _near_pairs(hashes: dict[str, tuple[int, str]], dh: dict):
    """Pasangan file near-duplicate (Hamming <= max_hamming) lewat LSH per pita."""
    size, maxh, bands = int(dh["size"]), int(dh["max_hamming"]), int(dh["bands"])
    bw = size * size // bands
    buckets: dict[tuple, list[str]] = defaultdict(list)
    for f, (h, _) in hashes.items():
        for b in range(bands):
            buckets[(b, (h >> (b * bw)) & ((1 << bw) - 1))].append(f)
    for files in buckets.values():
        if len(files) < 2:
            continue
        for i in range(len(files)):
            hi = hashes[files[i]][0]
            for j in range(i + 1, len(files)):
                if bin(hi ^ hashes[files[j]][0]).count("1") <= maxh:
                    yield files[i], files[j]


def assign(rows: list[dict], images_dir: Path, cfg: dict) -> dict[str, str]:
    """Split per grup. Dengan split.anchor_sources, sumber lama dibagi PERSIS seperti build sebelumnya
    (algoritma yang sama hanya pada baris sumber lama), lalu sumber baru ditempelkan tanpa memindahkan
    satu pun file lama: komponen baru yang near-duplicate dengan file lama ikut split file lama itu;
    jika file lama di komponen itu berada di split berbeda, file baru dibuang ("drop")."""
    anchor = set(cfg["split"].get("anchor_sources") or [])
    if not anchor:
        return _assign_core(rows, images_dir, cfg)
    old = [r for r in rows if r["source"] in anchor]
    new = [r for r in rows if r["source"] not in anchor]
    print(f"anchor_sources={sorted(anchor)}: {len(old)} file lama (split dipertahankan), {len(new)} file baru")
    out = _assign_core(old, images_dir, cfg) if old else {}
    if not new:
        return out
    scfg = cfg["split"]
    uf = UF()
    for r in new:
        uf.find(r["group"])
    dh = scfg.get("dhash", {})
    links: dict[str, set[str]] = defaultdict(set)   # grup baru -> file lama near-duplicate
    if dh.get("enabled"):
        hn = _hashes(new, images_dir, int(dh["size"]))
        ho = _hashes(old, images_dir, int(dh["size"]))
        allh = {**{f: (h, "N" + g) for f, (h, g) in hn.items()}, **{f: (h, "O" + g) for f, (h, g) in ho.items()}}
        n_new = n_link = 0
        for a, b in _near_pairs(allh, dh):
            ga, gb = allh[a][1], allh[b][1]
            if ga[0] == "N" and gb[0] == "N":
                if uf.find(ga[1:]) != uf.find(gb[1:]):
                    uf.union(ga[1:], gb[1:])
                    n_new += 1
            elif ga[0] == "N" and gb[0] == "O":
                links[ga[1:]].add(b)
                n_link += 1
            elif ga[0] == "O" and gb[0] == "N":
                links[gb[1:]].add(a)
                n_link += 1
        print(f"dHash baru: {n_new} penggabungan antar-file baru, {n_link} pasangan baru~lama")
    comp_old: dict[str, set[str]] = defaultdict(set)
    for g, files in links.items():
        comp_old[uf.find(g)] |= {out[f] for f in files if f in out}
    ratios = scfg["ratios"]
    cut_train, cut_val = ratios["train"], ratios["train"] + ratios["val"]
    dropped = 0
    for r in new:
        root = uf.find(r["group"])
        sp = comp_old.get(root)
        if sp:
            if len(sp) == 1:
                out[r["file"]] = next(iter(sp))
            else:
                out[r["file"]] = "drop"
                dropped += 1
        else:
            u = stable_unit(root, scfg["seed"])
            out[r["file"]] = "train" if u < cut_train else ("val" if u < cut_val else "test")
    print(f"file baru dibuang (near-duplicate dengan file lama di split berbeda): {dropped}")
    return out


def _assign_core(rows: list[dict], images_dir: Path, cfg: dict) -> dict[str, str]:
    scfg = cfg["split"]
    uf = UF()
    for r in rows:
        uf.find(r["group"])
    dh = scfg.get("dhash", {})
    if dh.get("enabled"):
        hashes = _hashes(rows, images_dir, int(dh["size"]))
        merged = 0
        for a, b in _near_pairs(hashes, dh):
            ga, gb = hashes[a][1], hashes[b][1]
            if uf.find(ga) != uf.find(gb):
                uf.union(ga, gb)
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
        if sp not in SPLITS:  # "drop" (lihat assign / anchor_sources)
            continue
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
