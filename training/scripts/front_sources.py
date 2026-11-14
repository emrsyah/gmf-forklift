"""Parser + pengunduh sumber data kamera DEPAN (Model 4, Tech Spec bag. 6) -> format raw seragam.

Hasil per sumber (sama dengan pipeline kabin, label masih memakai ID kelas upstream):
    <raw>/<source>/images/*.jpg|png
    <raw>/<source>/labels/*.txt      (YOLO, ID upstream)
    <raw>/<source>/classes.txt
    <raw>/<source>/source.yaml

Nama file (stem) membawa id sekuens/sesi supaya split per sekuens bisa dilakukan lewat regex grup
di class map (mis. LOCO: "subset-5__2020-03-05_1__RealSense__color__0__1583416214257_48").

Tipe sumber (training/datasets/sources_front.yaml -> type):
    coco_ultralytics  : dipakai ulang dari download_public.py (COCO person)
    roboflow          : dipakai ulang dari download_public.py (butuh ROBOFLOW_API_KEY)
    loco              : JSON COCO dari GitHub tum-fml/loco + gambar dari dataset.zip (HTTP range,
                        hanya gambar yang dipilih yang diunduh) atau zip lokal
    crowdhuman        : .odgt + zip gambar (HF sshao0516/CrowdHuman, HTTP range) atau folder lokal
    apron_local       : Apron Dataset (manual, butuh registrasi) -> CSV per gambar
    yolo_local        : folder YOLO lokal (data sintetis / rekaman tim)
"""
from __future__ import annotations

import csv
import io
import json
import os
import re
import shutil
import sys
import threading
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests

sys.path.insert(0, str(Path(__file__).resolve().parent))
import download_public as dp  # noqa: E402
from common import IMG_EXTS, load_yaml, rpath, sample  # noqa: E402

_SAFE = re.compile(r"[^A-Za-z0-9._-]+")


def sanitize(s: str) -> str:
    """Nama file aman lintas OS (koma di nama LOCO/CrowdHuman -> '_')."""
    return _SAFE.sub("_", s).strip("_")


def _clip_xyxy(x1, y1, x2, y2, W, H):
    x1, x2 = max(0.0, min(x1, x2)), min(float(W), max(x1, x2))
    y1, y2 = max(0.0, min(y1, y2)), min(float(H), max(y1, y2))
    return x1, y1, x2, y2


def yolo_line(cls: int, x1: float, y1: float, x2: float, y2: float, W: float, H: float) -> str | None:
    x1, y1, x2, y2 = _clip_xyxy(x1, y1, x2, y2, W, H)
    if x2 <= x1 or y2 <= y1:
        return None
    return f"{cls} {(x1 + x2) / 2 / W:.6f} {(y1 + y2) / 2 / H:.6f} {(x2 - x1) / W:.6f} {(y2 - y1) / H:.6f}"


# ------------------------------------------------------------------ parser (murni, dites di tests/)
def coco_index(j: dict) -> tuple[list[str], dict[int, int], dict[int, list]]:
    """COCO json -> (nama kelas urut id, id_kategori->indeks upstream, image_id->anotasi)."""
    cats = sorted(j["categories"], key=lambda c: int(c["id"]))
    names = [c["name"] for c in cats]
    idx = {int(c["id"]): i for i, c in enumerate(cats)}
    anns: dict[int, list] = {}
    for a in j.get("annotations", []):
        anns.setdefault(a["image_id"], []).append(a)
    return names, idx, anns


def coco_anns_to_yolo(anns: list, W: float, H: float, idx: dict[int, int], skip_crowd: bool = True) -> list[str]:
    out = []
    for a in anns:
        if skip_crowd and a.get("iscrowd") in (1, True):
            continue
        x, y, w, h = a["bbox"]
        ln = yolo_line(idx[int(a["category_id"])], x, y, x + w, y + h, W, H)
        if ln:
            out.append(ln)
    return out


def loco_member_and_stem(path: str) -> tuple[str, str]:
    """'/dataset/subset-5/2020-03-05_1/RealSense/color/0/1583416214257,48.jpg' ->
    ('dataset/subset-5/.../1583416214257,48.jpg', 'subset-5__2020-03-05_1__RealSense__color__0__1583416214257_48')."""
    member = path.lstrip("/")
    parts = member.split("/")
    if parts and parts[0] == "dataset":
        parts = parts[1:]
    parts[-1] = Path(parts[-1]).stem
    return member, "__".join(sanitize(p) for p in parts)


def apron_csv_to_yolo(text: str, W: float, H: float) -> list[str]:
    """CSV Apron (satu per gambar): kolom label, tlx, tly, brx, bry (+ meta). Urutan kolom tidak diasumsikan.
    Header tanpa kolom wajib -> ValueError (jangan diam-diam menghasilkan gambar tanpa label)."""
    text = text.lstrip("﻿")
    rd = csv.DictReader(io.StringIO(text))
    need = {"label", "tlx", "tly", "brx", "bry"}
    cols = {c.strip() for c in (rd.fieldnames or [])}
    if text.strip() and not need <= cols:
        raise ValueError(f"CSV Apron tanpa kolom {sorted(need - cols)} (header: {rd.fieldnames})")
    out = []
    for row in rd:
        row = {(k or "").strip(): v for k, v in row.items()}
        try:
            ln = yolo_line(int(float(row["label"])), float(row["tlx"]), float(row["tly"]),
                           float(row["brx"]), float(row["bry"]), W, H)
        except (TypeError, ValueError):
            continue
        if ln:
            out.append(ln)
    return out


def crowdhuman_to_yolo(rec: dict, W: float, H: float, box: str = "vbox") -> list[str]:
    """Satu baris .odgt -> label YOLO kelas upstream 0 ('person').
    tag 'mask' (kerumunan/pantulan/patung) dan extra.ignore=1 dilewati. vbox = bagian yang terlihat
    (konsisten dengan kotak COCO); fbox bisa keluar gambar -> di-clip."""
    out = []
    for g in rec.get("gtboxes", []):
        if g.get("tag") != "person" or int((g.get("extra") or {}).get("ignore", 0)) == 1:
            continue
        b = g.get(box)
        if not b:
            continue
        x, y, w, h = b
        ln = yolo_line(0, x, y, x + w, y + h, W, H)
        if ln:
            out.append(ln)
    return out


def crowdhuman_stem(image_id: str) -> str:
    return sanitize(image_id)


def image_size(data: bytes | None = None, path: Path | None = None) -> tuple[int, int]:
    from PIL import Image

    with Image.open(io.BytesIO(data) if data is not None else path) as im:
        return im.size  # (W, H)


# ------------------------------------------------------------------ zip jarak jauh (HTTP Range)
class HttpRangeFile(io.RawIOBase):
    """File seekable di atas HTTP Range: zipfile hanya membaca central directory + member yang dipilih."""

    def __init__(self, url: str, retries: int = 6, timeout: float = 60):
        self.url, self.retries, self.timeout = url, retries, timeout
        self._resolve()
        self.pos = 0

    def _resolve(self) -> None:
        r = requests.head(self.url, allow_redirects=True, timeout=self.timeout, headers=dp.UA)
        r.raise_for_status()
        self.final = r.url  # URL CDN bertanda tangan (HF) bisa kedaluwarsa -> di-resolve ulang saat 403
        self.size = int(r.headers["Content-Length"])

    def readable(self): return True
    def seekable(self): return True
    def tell(self): return self.pos

    def seek(self, off, whence=0):
        self.pos = off if whence == 0 else (self.pos + off if whence == 1 else self.size + off)
        return self.pos

    def readinto(self, b) -> int:
        if self.pos >= self.size:
            return 0
        end = min(self.size, self.pos + len(b)) - 1
        last = None
        for i in range(self.retries):
            try:
                r = requests.get(self.final, headers={**dp.UA, "Range": f"bytes={self.pos}-{end}"}, timeout=self.timeout)
                if r.status_code in (401, 403, 410):
                    self._resolve()
                    raise RuntimeError(f"HTTP {r.status_code}")
                if r.status_code != 206:
                    raise RuntimeError(f"HTTP {r.status_code} (server tidak mendukung Range?)")
                d = r.content
                b[:len(d)] = d
                self.pos += len(d)
                return len(d)
            except Exception as e:  # noqa: BLE001
                last = e
                import time
                time.sleep(min(30, 2 ** i))
        raise RuntimeError(f"gagal baca range {self.url}: {last}")


class ZipSource:
    """ZipFile per thread dari path lokal ATAU URL (HTTP Range)."""

    def __init__(self, local: Path | None, url: str | None, dcfg: dict):
        self.local, self.url, self.dcfg = local, url, dcfg
        self._tl = threading.local()

    def zf(self) -> zipfile.ZipFile:
        z = getattr(self._tl, "zf", None)
        if z is None:
            if self.local is not None:
                z = zipfile.ZipFile(self.local)
            else:
                raw = HttpRangeFile(self.url, self.dcfg["retries"], self.dcfg["timeout_s"])
                z = zipfile.ZipFile(io.BufferedReader(raw, 1 << 18))
            self._tl.zf = z
        return z


def _local(p) -> Path | None:
    if not p:
        return None
    q = rpath(p)
    return q if q.exists() else None


def _fetch_cached(url: str, cache: Path, dcfg: dict) -> Path:
    dst = cache / Path(url.split("?")[0]).name
    if not dst.exists():
        dp.http_get(url, dst, dcfg["retries"], max(dcfg["timeout_s"], 300))
    return dst


def _parallel(fn, items, workers: int) -> int:
    with ThreadPoolExecutor(workers) as ex:
        return sum(ex.map(fn, items))


# ------------------------------------------------------------------ handler: LOCO
def dl_loco(name: str, s: dict, root: Path, limit: int | None, dcfg: dict, keep_archives: bool) -> int:
    cache = root / "_archives"
    names: list[str] | None = None
    entries: dict[str, tuple] = {}  # path -> (W, H, lines)
    for af in s["annotation_files"]:
        local = _local(Path(s["local_dir"]) / af) if s.get("local_dir") else None
        jp = local or _fetch_cached(s["annotation_url_template"].format(file=af), cache, dcfg)
        j = json.loads(Path(jp).read_text(encoding="utf-8"))
        n_, idx, anns = coco_index(j)
        if names is None:
            names = n_
        elif n_ != names:
            raise RuntimeError(f"kategori LOCO berbeda antar file: {n_} vs {names}")
        for im in j["images"]:
            entries[im["path"]] = (im["width"], im["height"], coco_anns_to_yolo(anns.get(im["id"], []), im["width"], im["height"], idx))
    chosen = sample(list(entries), limit, dcfg["seed"])
    stems = {p: loco_member_and_stem(p) for p in chosen}
    print(f"[{name}] {len(entries)} gambar beranotasi, dipilih {len(chosen)}")
    dp.reset_dir(root, keep={st for _, st in stems.values()})
    zs = ZipSource(_local(s.get("local_zip")), s["images_zip_url"], dcfg)
    members = set(zs.zf().namelist())
    by_base = {}
    for m in members:
        by_base.setdefault(Path(m).name, []).append(m)

    def job(p: str) -> int:
        member, stem = stems[p]
        W, H, lines = entries[p]
        img_dst, lab_dst = root / "images" / f"{stem}.jpg", root / "labels" / f"{stem}.txt"
        if not dp._have(img_dst, lab_dst):
            if member not in members:
                cand = [m for m in by_base.get(Path(member).name, []) if member.split("/")[1] in m]
                if len(cand) != 1:
                    print(f"  lewati (tidak ada di zip): {member}")
                    return 0
                member = cand[0]
            try:
                img_dst.write_bytes(zs.zf().read(member))
            except Exception as e:  # noqa: BLE001
                print(f"  lewati {member}: {e}")
                img_dst.unlink(missing_ok=True)
                return 0
        lab_dst.write_text("".join(ln + "\n" for ln in lines), encoding="utf-8")
        return 1

    n = _parallel(job, chosen, dcfg["workers"])
    if not keep_archives:
        shutil.rmtree(cache, ignore_errors=True)
    dp.write_meta(root, {"source": name, "type": "loco", "images_zip_url": s["images_zip_url"],
                         "available": len(entries), "downloaded": n, "limit": limit}, names or [])
    return n


# ------------------------------------------------------------------ handler: CrowdHuman
def dl_crowdhuman(name: str, s: dict, root: Path, limit: int | None, dcfg: dict, keep_archives: bool) -> int:
    cache = root / "_archives"
    ldir = _local(s.get("local_dir"))
    recs: dict[str, dict] = {}
    for af in s["annotation_files"]:
        jp = (ldir / af) if (ldir and (ldir / af).exists()) else _fetch_cached(s["url_template"].format(file=af), cache, dcfg)
        for ln in Path(jp).read_text(encoding="utf-8").splitlines():
            if ln.strip():
                r = json.loads(ln)
                recs[r["ID"]] = r
    chosen = sample(list(recs), limit, dcfg["seed"])
    print(f"[{name}] {len(recs)} gambar beranotasi, dipilih {len(chosen)}")
    dp.reset_dir(root, keep={crowdhuman_stem(i) for i in chosen})

    if ldir and (ldir / s.get("local_images_dir", "Images")).exists():
        imgdir = ldir / s.get("local_images_dir", "Images")
        reader = lambda iid: (imgdir / f"{iid}.jpg").read_bytes()  # noqa: E731
    else:
        zsrc = [ZipSource(None, s["url_template"].format(file=z), dcfg) for z in s["image_zips"]]
        where: dict[str, tuple[ZipSource, str]] = {}
        for zs in zsrc:
            for m in zs.zf().namelist():
                if Path(m).suffix.lower() in IMG_EXTS:
                    where[Path(m).stem] = (zs, m)
        missing = [i for i in chosen if i not in where]
        if missing:
            print(f"  {len(missing)} ID tidak ada di zip {s['image_zips']} (mis. anotasi train tapi zip train0x tidak dipakai)")

        def reader(iid):
            zs, m = where[iid]
            return zs.zf().read(m)

    def job(iid: str) -> int:
        stem = crowdhuman_stem(iid)
        img_dst, lab_dst = root / "images" / f"{stem}.jpg", root / "labels" / f"{stem}.txt"
        try:
            if not dp._have(img_dst, lab_dst):
                img_dst.write_bytes(reader(iid))
            W, H = image_size(path=img_dst)
        except Exception as e:  # noqa: BLE001
            print(f"  lewati {iid}: {e}")
            img_dst.unlink(missing_ok=True)
            return 0
        lines = crowdhuman_to_yolo(recs[iid], W, H, s.get("box", "vbox"))
        lab_dst.write_text("".join(ln + "\n" for ln in lines), encoding="utf-8")
        return 1

    n = _parallel(job, chosen, dcfg["workers"])
    if not keep_archives:
        shutil.rmtree(cache, ignore_errors=True)
    dp.write_meta(root, {"source": name, "type": "crowdhuman", "box": s.get("box", "vbox"),
                         "available": len(recs), "downloaded": n, "limit": limit}, ["person"])
    return n


# ------------------------------------------------------------------ handler: Apron (manual)
def _manual_missing(name: str, s: dict, d: Path | None) -> int:
    print(f"[{name}] folder manual tidak ditemukan ({s.get('local_dir')}) -> dilewati. "
          f"Lihat petunjuk di sources_front.yaml ({name}.manual).")
    return 0


def _link(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    if dst.exists():
        dst.unlink()
    try:
        os.link(src, dst)
    except OSError:
        shutil.copy2(src, dst)


def rel_stem(rel: Path) -> str:
    """'seq_012/frame_0001.jpg' -> 'seq_012__frame_0001' (folder = sekuens tetap ada di nama)."""
    return "__".join(sanitize(p) for p in [*rel.parent.parts, rel.stem])


def dl_apron_local(name: str, s: dict, root: Path, limit: int | None, dcfg: dict, keep_archives: bool) -> int:
    d = _local(s.get("local_dir"))
    if d is None:
        return _manual_missing(name, s, d)
    imgs_dir, csv_dir = d / s.get("images_dir", "images"), d / s.get("bboxes_dir", "bboxes")
    pool = sorted(p.relative_to(imgs_dir) for p in imgs_dir.rglob("*") if p.suffix.lower() in IMG_EXTS)
    chosen = sample(pool, limit, dcfg["seed"])
    dp.reset_dir(root)
    n = no_csv = 0
    for rel in chosen:
        c = (csv_dir / rel).with_suffix(".csv")
        if not c.exists():
            no_csv += 1
            continue
        src = imgs_dir / rel
        W, H = image_size(path=src)
        stem = rel_stem(rel)
        _link(src, root / "images" / f"{stem}{src.suffix.lower()}")
        lines = apron_csv_to_yolo(c.read_text(encoding="utf-8-sig"), W, H)
        (root / "labels" / f"{stem}.txt").write_text("".join(ln + "\n" for ln in lines), encoding="utf-8")
        n += 1
    print(f"[{name}] {len(pool)} gambar, diambil {n}" + (f", {no_csv} tanpa CSV dilewati" if no_csv else ""))
    dp.write_meta(root, {"source": name, "type": "apron_local", "local_dir": str(d),
                         "available": len(pool), "downloaded": n, "limit": limit}, list(s["names"]))
    return n


# ------------------------------------------------------------------ handler: folder YOLO lokal (sintetis)
def dl_yolo_local(name: str, s: dict, root: Path, limit: int | None, dcfg: dict, keep_archives: bool) -> int:
    d = _local(s.get("local_dir"))
    if d is None:
        return _manual_missing(name, s, d)
    names = [x.strip() for x in (d / "classes.txt").read_text(encoding="utf-8").splitlines() if x.strip()]
    pool = sorted(p.relative_to(d / "images") for p in (d / "images").rglob("*") if p.suffix.lower() in IMG_EXTS)
    chosen = sample(pool, limit, dcfg["seed"])
    dp.reset_dir(root)
    for rel in chosen:
        stem = rel_stem(rel)
        _link(d / "images" / rel, root / "images" / f"{stem}{rel.suffix.lower()}")
        lab = (d / "labels" / rel).with_suffix(".txt")
        (root / "labels" / f"{stem}.txt").write_text(lab.read_text(encoding="utf-8") if lab.exists() else "", encoding="utf-8")
    print(f"[{name}] {len(pool)} gambar, diambil {len(chosen)}")
    dp.write_meta(root, {"source": name, "type": "yolo_local", "local_dir": str(d),
                         "available": len(pool), "downloaded": len(chosen), "limit": limit}, names)
    return len(chosen)


HANDLERS = {
    "coco_ultralytics": dp.dl_coco_ultralytics,
    "roboflow": dp.dl_roboflow,
    "loco": dl_loco,
    "crowdhuman": dl_crowdhuman,
    "apron_local": dl_apron_local,
    "yolo_local": dl_yolo_local,
}


def run(out: Path, limit: int | None, only: list[str] | None, cfg: dict, keep_archives: bool = False) -> dict:
    import time

    sources = load_yaml(rpath(cfg["paths"]["sources"]))["sources"]
    dcfg = cfg["download"]
    counts = {}
    for name, s in dp.enabled_sources(sources, only).items():
        lim = limit if limit is not None else s.get("max_images")
        t0 = time.time()
        counts[name] = HANDLERS[s["type"]](name, s, out / name, lim, dcfg, keep_archives)
        print(f"[{name}] selesai: {counts[name]} gambar ({time.time() - t0:.0f} s)")
    return counts
