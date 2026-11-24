"""Unduh dataset publik (tanpa token) untuk model YOLO kabin.

Hasil per sumber (label masih memakai ID kelas upstream):
    <out>/<source>/images/*.{jpg,png}
    <out>/<source>/labels/*.txt      (YOLO, ID upstream)
    <out>/<source>/classes.txt       (nama kelas upstream, satu per baris)
    <out>/<source>/source.yaml       (metadata: repo/URL, jumlah, limit)

Contoh:
    python training/scripts/download_public.py --out data/raw --limit 80          # smoke
    python training/scripts/download_public.py --out data/raw                     # penuh (max_images di sources.yaml)
    python training/scripts/download_public.py --only coco_phone ppe_vincent

Sumber Roboflow hanya diproses jika ROBOFLOW_API_KEY ada di environment.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
import time
import zipfile
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path

import requests
import yaml

sys.path.insert(0, str(Path(__file__).resolve().parent))
from common import IMG_EXTS, load_build_cfg, load_yaml, rpath, sample  # noqa: E402

UA = {"User-Agent": "cabin-dataset-builder/1.0"}


# ------------------------------------------------------------------ HTTP helpers
def http_get(url: str, dst: Path | None, retries: int, timeout: float, params: dict | None = None) -> bytes | None:
    """GET dengan retry. Jika dst diberikan, tulis streaming ke file dan kembalikan None."""
    last = None
    headers = dict(UA)
    tok = os.environ.get("HF_TOKEN")
    if tok and "huggingface.co" in url:  # opsional: menaikkan rate limit HF, tidak wajib
        headers["Authorization"] = f"Bearer {tok}"
    for i in range(retries):
        try:
            with requests.get(url, stream=dst is not None, timeout=timeout, headers=headers, params=params) as r:
                if r.status_code == 429:  # rate limit: tunggu sesuai Retry-After
                    wait = float(r.headers.get("Retry-After", 30))
                    time.sleep(min(wait, 300))
                    raise RuntimeError("HTTP 429")
                if r.status_code == 404:
                    raise FileNotFoundError(url)
                r.raise_for_status()
                if dst is None:
                    return r.content
                dst.parent.mkdir(parents=True, exist_ok=True)
                tmp = dst.with_suffix(dst.suffix + ".part")
                with open(tmp, "wb") as f:
                    for chunk in r.iter_content(1 << 20):
                        f.write(chunk)
                tmp.replace(dst)
                return None
        except FileNotFoundError:
            raise
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(min(60, 2 ** i))
    raise RuntimeError(f"gagal unduh {url}: {last}")


def write_meta(root: Path, meta: dict, names: list[str]) -> None:
    root.mkdir(parents=True, exist_ok=True)
    (root / "classes.txt").write_text("\n".join(names) + "\n", encoding="utf-8")
    with open(root / "source.yaml", "w", encoding="utf-8") as f:
        yaml.safe_dump(meta, f, sort_keys=False, allow_unicode=True)


def reset_dir(root: Path, keep: set[str] | None = None) -> None:
    """Siapkan images/ dan labels/. Dengan `keep`, file yang stem-nya ada di keep dipertahankan
    (unduhan bisa dilanjutkan setelah gagal); sisanya dihapus."""
    for sub in ("images", "labels"):
        d = root / sub
        if d.exists():
            if keep is None:
                shutil.rmtree(d)
            else:
                for f in d.iterdir():
                    if f.stem not in keep or f.suffix == ".part":
                        f.unlink()
        d.mkdir(parents=True, exist_ok=True)


def _have(img_dst: Path, lab_dst: Path) -> bool:
    return img_dst.exists() and img_dst.stat().st_size > 0 and lab_dst.exists()


# ------------------------------------------------------------------ handlers
def dl_hf_yolo_files(name: str, s: dict, root: Path, limit: int | None, dcfg: dict) -> int:
    """Repo HF berisi <split>/images/* dan <split>/labels/*.txt (format YOLO)."""
    from huggingface_hub import HfApi, hf_hub_url

    api = HfApi()
    rev = s.get("revision", "main")
    pool: list[str] = []
    for sp in s["splits"]:
        folder = f"{sp}/{s.get('images_dir', 'images')}"
        for f in api.list_repo_tree(s["repo"], path_in_repo=folder, repo_type="dataset", revision=rev):
            if hasattr(f, "size") and Path(f.path).suffix.lower() in IMG_EXTS:
                pool.append(f.path)
    chosen = sample(pool, limit, dcfg["seed"])
    # nama output unik (stem yang sama di dua split upstream diberi prefix split), ditentukan sebelum thread
    out_stems: dict[str, str] = {}
    used: set[str] = set()
    for p in chosen:
        stem = Path(p).stem
        o = stem if stem not in used else f"{p.split('/')[0]}_{stem}"
        used.add(o)
        out_stems[p] = o
    print(f"[{name}] {len(pool)} gambar di repo, dipilih {len(chosen)}")
    reset_dir(root, keep=set(out_stems.values()))

    def job(p: str) -> int:
        sp = p.split("/")[0]
        out_stem = out_stems[p]
        lab = f"{sp}/{s.get('labels_dir', 'labels')}/{Path(p).stem}.txt"
        img_dst = root / "images" / f"{out_stem}{Path(p).suffix.lower()}"
        lab_dst = root / "labels" / f"{out_stem}.txt"
        if _have(img_dst, lab_dst):
            return 1  # sudah ada dari run sebelumnya
        try:
            http_get(hf_hub_url(s["repo"], p, repo_type="dataset", revision=rev), img_dst, dcfg["retries"], dcfg["timeout_s"])
            try:
                http_get(hf_hub_url(s["repo"], lab, repo_type="dataset", revision=rev), lab_dst, dcfg["retries"], dcfg["timeout_s"])
            except FileNotFoundError:
                lab_dst.write_text("", encoding="utf-8")  # gambar tanpa objek
        except Exception as e:  # noqa: BLE001  - jangan hentikan seluruh unduhan karena satu file
            print(f"  lewati {p}: {e}")
            img_dst.unlink(missing_ok=True)
            lab_dst.unlink(missing_ok=True)
            return 0
        return 1

    with ThreadPoolExecutor(dcfg["workers"]) as ex:
        n = sum(ex.map(job, chosen))
    write_meta(root, {"source": name, "type": s["type"], "repo": s["repo"], "revision": rev,
                      "available": len(pool), "downloaded": n, "limit": limit}, list(s["names"]))
    return n


def _ingest_coco_json(zf: zipfile.ZipFile, ann_name: str, names_by_id: dict, prefix: str) -> list[tuple]:
    j = json.loads(zf.read(ann_name))
    base = ann_name.rsplit("/", 1)[0] + "/" if "/" in ann_name else ""
    anns: dict[int, list] = {}
    for a in j["annotations"]:
        anns.setdefault(a["image_id"], []).append(a)
    for c in j["categories"]:
        names_by_id[int(c["id"])] = c["name"]
    return [(prefix, base + im["file_name"], im, anns.get(im["id"], [])) for im in j["images"]]


def dl_hf_coco_zip(name: str, s: dict, root: Path, limit: int | None, dcfg: dict, keep_archives: bool) -> int:
    """Repo HF (roboflow2huggingface) berisi zip dengan gambar + _annotations.coco.json."""
    from huggingface_hub import hf_hub_url

    rev = s.get("revision", "main")
    cache = root / "_archives"
    zips = []
    for z in s["zips"]:
        dst = cache / Path(z).name
        if not dst.exists():
            http_get(hf_hub_url(s["repo"], z, repo_type="dataset", revision=rev), dst, dcfg["retries"], dcfg["timeout_s"])
        zips.append(dst)
    names_by_id: dict[int, str] = {}
    entries = []
    handles = {}
    for zp in zips:
        zf = zipfile.ZipFile(zp)
        handles[zp.stem] = zf
        for ann in [n for n in zf.namelist() if n.endswith(".json")]:
            entries += _ingest_coco_json(zf, ann, names_by_id, zp.stem)
    ids = sorted(names_by_id)
    idx = {cid: i for i, cid in enumerate(ids)}
    keyed = {f"{e[0]}/{e[1]}": e for e in entries}
    chosen = sample(list(keyed), limit, dcfg["seed"])
    print(f"[{name}] {len(keyed)} gambar di zip, diambil {len(chosen)}")
    reset_dir(root)
    n = 0
    for k in chosen:
        prefix, member, im, anns = keyed[k]
        W, H = float(im["width"]), float(im["height"])
        stem = Path(im["file_name"]).stem
        data = handles[prefix].read(member)
        (root / "images" / f"{stem}{Path(member).suffix.lower()}").write_bytes(data)
        lines = []
        for a in anns:
            x, y, w, h = a["bbox"]
            x1, y1 = max(0.0, x), max(0.0, y)
            x2, y2 = min(W, x + w), min(H, y + h)
            if x2 <= x1 or y2 <= y1:
                continue
            lines.append(f"{idx[int(a['category_id'])]} {(x1 + x2) / 2 / W:.6f} {(y1 + y2) / 2 / H:.6f} "
                         f"{(x2 - x1) / W:.6f} {(y2 - y1) / H:.6f}")
        (root / "labels" / f"{stem}.txt").write_text("".join(l + "\n" for l in lines), encoding="utf-8")
        n += 1
    for zf in handles.values():
        zf.close()
    if not keep_archives:
        shutil.rmtree(cache, ignore_errors=True)
    write_meta(root, {"source": name, "type": s["type"], "repo": s["repo"], "revision": rev,
                      "available": len(keyed), "downloaded": n, "limit": limit},
               [names_by_id[i] for i in ids])
    return n


def _ingest_yolo_zip(zf: zipfile.ZipFile, root: Path, names: list[str], limit, seed) -> tuple[int, int]:
    members = [m for m in zf.namelist() if not m.startswith("__MACOSX") and "/._" not in m]
    imgs = [m for m in members if Path(m).suffix.lower() in IMG_EXTS and "/images/" in f"/{m}"]
    labels = {Path(m).stem: m for m in members if m.endswith(".txt") and "/labels/" in f"/{m}"}
    chosen = sample(imgs, limit, seed)
    reset_dir(root)
    for m in chosen:
        stem = Path(m).stem
        (root / "images" / f"{stem}{Path(m).suffix.lower()}").write_bytes(zf.read(m))
        lab = zf.read(labels[stem]).decode("utf-8", "ignore") if stem in labels else ""
        (root / "labels" / f"{stem}.txt").write_text(lab, encoding="utf-8")
    return len(imgs), len(chosen)


def dl_hf_yolo_zip(name: str, s: dict, root: Path, limit: int | None, dcfg: dict, keep_archives: bool) -> int:
    from huggingface_hub import hf_hub_url

    rev = s.get("revision", "main")
    dst = root / "_archives" / Path(s["zip"]).name
    if not dst.exists():
        http_get(hf_hub_url(s["repo"], s["zip"], repo_type="dataset", revision=rev), dst, dcfg["retries"], dcfg["timeout_s"])
    with zipfile.ZipFile(dst) as zf:
        names = [x.strip() for x in zf.read(s["names_file"]).decode("utf-8").splitlines() if x.strip()]
        avail, n = _ingest_yolo_zip(zf, root, names, limit, dcfg["seed"])
    if not keep_archives:
        shutil.rmtree(root / "_archives", ignore_errors=True)
    print(f"[{name}] {avail} gambar di zip, diambil {n}")
    write_meta(root, {"source": name, "type": s["type"], "repo": s["repo"], "revision": rev,
                      "available": avail, "downloaded": n, "limit": limit}, names)
    return n


def _coco_names() -> list[str]:
    import ultralytics
    y = load_yaml(Path(ultralytics.__file__).parent / "cfg" / "datasets" / "coco.yaml")
    return [y["names"][i] for i in sorted(y["names"])]


def dl_coco_ultralytics(name: str, s: dict, root: Path, limit: int | None, dcfg: dict, keep_archives: bool) -> int:
    """COCO 2017: label YOLO dari zip rilis Ultralytics, gambar per file dari images.cocodataset.org."""
    zdst = root / "_archives" / "coco2017labels.zip"
    if not zdst.exists():
        http_get(s["labels_zip_url"], zdst, dcfg["retries"], max(dcfg["timeout_s"], 300))
    # require_class: satu kelas (int) ATAU daftar (salah satu cukup), dengan kotak >= min_box_area_frac.
    # require_also: kelas lain yang harus ada (mis. person -> objek dipegang orang).
    # exclude_classes: gambar dibuang jika berisi kelas ini (mis. cell phone untuk negatif keras;
    #   COCO melabel 80 kelasnya secara lengkap, jadi "tidak ada label cell phone" = tidak ada ponsel).
    rc = s["require_class"]
    want = {int(x) for x in rc} if isinstance(rc, (list, tuple)) else {int(rc)}
    min_area = float(s.get("min_box_area_frac", 0.0))
    also = {int(x) for x in s.get("require_also", []) or []}
    excl = {int(x) for x in s.get("exclude_classes", []) or []}
    pool: dict[str, tuple[str, str]] = {}
    with zipfile.ZipFile(zdst) as zf:
        for m in zf.namelist():
            parts = m.split("/")
            if len(parts) < 2 or not m.endswith(".txt") or parts[-2] not in s["splits"]:
                continue
            txt = zf.read(m).decode("utf-8", "ignore")
            ok = False
            present = set()
            for ln in txt.splitlines():
                v = ln.split()
                if len(v) < 5:
                    continue
                c = int(v[0])
                present.add(c)
                if c in want and float(v[3]) * float(v[4]) >= min_area:
                    ok = True
            if ok and also <= present and not (excl & present):
                pool[f"{parts[-2]}/{Path(m).stem}"] = (parts[-2], txt)
    chosen = sample(list(pool), limit, dcfg["seed"])
    print(f"[{name}] {len(pool)} gambar COCO berisi kelas {want}, dipilih {len(chosen)}")
    reset_dir(root, keep={k.split("/")[1] for k in chosen})

    def job(k: str) -> int:
        split, txt = pool[k]
        stem = k.split("/")[1]
        img_dst, lab_dst = root / "images" / f"{stem}.jpg", root / "labels" / f"{stem}.txt"
        if not _have(img_dst, lab_dst):
            url = s["image_url_template"].format(split=split, name=f"{stem}.jpg")
            try:
                http_get(url, img_dst, dcfg["retries"], dcfg["timeout_s"])
            except Exception as e:  # noqa: BLE001
                print(f"  lewati {stem}: {e}")
                img_dst.unlink(missing_ok=True)
                return 0
        lab_dst.write_text(txt, encoding="utf-8")
        return 1

    with ThreadPoolExecutor(dcfg["workers"]) as ex:
        n = sum(ex.map(job, chosen))
    if not keep_archives:
        shutil.rmtree(root / "_archives", ignore_errors=True)
    write_meta(root, {"source": name, "type": s["type"], "labels_zip_url": s["labels_zip_url"],
                      "available": len(pool), "downloaded": n, "limit": limit}, _coco_names())
    return n


def _oi_stream_rows(url: str, keep, timeout: float):
    """Baca CSV Open Images (GB) secara streaming; yield baris (dict) yang lolos keep(row)."""
    import csv as _csv
    with requests.get(url, stream=True, timeout=timeout, headers=UA) as r:
        r.raise_for_status()
        lines = (ln.decode("utf-8", "ignore") for ln in r.iter_lines() if ln)
        for row in _csv.DictReader(lines):
            if keep(row):
                yield row


def dl_openimages(name: str, s: dict, root: Path, limit: int | None, dcfg: dict, keep_archives: bool) -> int:
    """Open Images V7: kotak `Mobile phone` (anotasi CC BY 4.0, gambar terdaftar CC BY 2.0).

    Pilih gambar yang ponselnya kemungkinan dipegang/dipakai (kotak ponsel menyentuh kotak tangan atau
    dekat wajah), buang gambar dengan kotak ponsel group-of / depiction (instance tidak lengkap / gambar
    kartun), gambar berisi helm (helm proyek tidak dilabel di sini), dan gambar yang perlu dirotasi.
    Gambar diunduh dari bucket S3 publik Open Images dan diperkecil (sisi terpanjang max_side_px).
    """
    import cv2
    import numpy as np

    phone, hand, face = s["phone_label"], s["hand_label"], s["face_label"]
    skip_labels = set(s.get("skip_if_labels", []) or [])
    wanted = {phone, hand, face} | skip_labels
    min_area = float(s.get("min_box_area_frac", 0.0))
    near_face = float(s.get("near_face_widths", 1.5))
    t_out = max(dcfg["timeout_s"], 600)
    boxes: dict[str, dict[str, list]] = {}
    print(f"[{name}] stream {s['bbox_csv_url']}")
    for row in _oi_stream_rows(s["bbox_csv_url"], lambda r: r["LabelName"] in wanted, t_out):
        b = (float(row["XMin"]), float(row["YMin"]), float(row["XMax"]), float(row["YMax"]),
             row["IsGroupOf"] == "1", row["IsDepiction"] == "1")
        boxes.setdefault(row["ImageID"], {}).setdefault(row["LabelName"], []).append(b)

    def touches(a, b) -> bool:
        return min(a[2], b[2]) > max(a[0], b[0]) and min(a[3], b[3]) > max(a[1], b[1])

    pool: dict[str, list] = {}
    stats = {"with_phone": 0, "group_or_depiction": 0, "skip_label": 0, "small": 0, "not_in_use": 0}
    for iid, by in boxes.items():
        ph = by.get(phone)
        if not ph:
            continue
        stats["with_phone"] += 1
        if any(b[4] or b[5] for b in ph):
            stats["group_or_depiction"] += 1
            continue
        if any(lbl in by for lbl in skip_labels):
            stats["skip_label"] += 1
            continue
        big = [b for b in ph if (b[2] - b[0]) * (b[3] - b[1]) >= min_area]
        if not big:
            stats["small"] += 1
            continue
        in_use = False
        for b in big:
            if any(touches(b, h) for h in by.get(hand, [])):
                in_use = True
            for f in by.get(face, []):
                fw = max(f[2] - f[0], 1e-6)
                if (abs((b[0] + b[2]) / 2 - (f[0] + f[2]) / 2) <= near_face * fw and
                        abs((b[1] + b[3]) / 2 - (f[1] + f[3]) / 2) <= near_face * fw):
                    in_use = True
        if not in_use and s.get("require_in_use", True):
            stats["not_in_use"] += 1
            continue
        pool[iid] = ph
    del boxes
    print(f"[{name}] kandidat {len(pool)} gambar; {stats}")

    # rotasi + atribusi (Author/License/OriginalURL) dari daftar gambar resmi; buang yang perlu dirotasi
    meta: dict[str, dict] = {}
    for row in _oi_stream_rows(s["images_csv_url"], lambda r: r["ImageID"] in pool, t_out):
        meta[row["ImageID"]] = row
    rot_bad = [i for i in pool if i not in meta or meta[i].get("Rotation", "") not in ("", "0", "0.0")]
    for i in rot_bad:
        pool.pop(i, None)
    print(f"[{name}] dibuang karena rotasi/tanpa metadata: {len(rot_bad)}")

    chosen = sample(list(pool), limit, dcfg["seed"])
    reset_dir(root, keep=set(chosen))
    max_side = int(s.get("max_side_px", 1024))

    def job(iid: str) -> int:
        img_dst, lab_dst = root / "images" / f"{iid}.jpg", root / "labels" / f"{iid}.txt"
        if _have(img_dst, lab_dst):
            return 1
        try:
            data = http_get(s["image_url_template"].format(id=iid), None, dcfg["retries"], dcfg["timeout_s"])
            img = cv2.imdecode(np.frombuffer(data, np.uint8), cv2.IMREAD_COLOR)
            if img is None:
                return 0
            h, w = img.shape[:2]
            k = max_side / max(h, w)
            if k < 1:
                img = cv2.resize(img, (round(w * k), round(h * k)), interpolation=cv2.INTER_AREA)
            ok, buf = cv2.imencode(".jpg", img, [cv2.IMWRITE_JPEG_QUALITY, 95])
            if not ok:
                return 0
            buf.tofile(str(img_dst))
        except Exception as e:  # noqa: BLE001
            print(f"  lewati {iid}: {e}")
            img_dst.unlink(missing_ok=True)
            return 0
        lab_dst.write_text("".join(f"0 {(b[0] + b[2]) / 2:.6f} {(b[1] + b[3]) / 2:.6f} {b[2] - b[0]:.6f} "
                                   f"{b[3] - b[1]:.6f}\n" for b in pool[iid]), encoding="utf-8")
        return 1

    with ThreadPoolExecutor(dcfg["workers"]) as ex:
        n = sum(ex.map(job, chosen))
    import csv as _csv
    with open(root / "attribution.csv", "w", newline="", encoding="utf-8") as f:
        w = _csv.writer(f)
        w.writerow(["ImageID", "OriginalURL", "OriginalLandingURL", "License", "Author", "Title"])
        for iid in chosen:
            m = meta.get(iid, {})
            w.writerow([iid] + [m.get(k, "") for k in ("OriginalURL", "OriginalLandingURL", "License", "Author", "Title")])
    write_meta(root, {"source": name, "type": s["type"], "bbox_csv_url": s["bbox_csv_url"],
                      "available": len(pool), "downloaded": n, "limit": limit, "filter_stats": stats}, ["Mobile phone"])
    return n


def dl_roboflow(name: str, s: dict, root: Path, limit: int | None, dcfg: dict, keep_archives: bool) -> int:
    """Roboflow Universe (opsional, butuh ROBOFLOW_API_KEY). Belum diuji tanpa key."""
    key = os.environ.get("ROBOFLOW_API_KEY")
    if not key:
        print(f"[{name}] ROBOFLOW_API_KEY tidak ada -> dilewati")
        return 0
    url = f"https://api.roboflow.com/{s['workspace']}/{s['project']}/{s['version']}/{s.get('export_format', 'yolov8')}"
    info = json.loads(http_get(url, None, dcfg["retries"], dcfg["timeout_s"], params={"api_key": key}))
    link = info["export"]["link"]
    zdst = root / "_archives" / f"{s['project']}_{s['version']}.zip"
    http_get(link, zdst, dcfg["retries"], max(dcfg["timeout_s"], 300))
    with zipfile.ZipFile(zdst) as zf:
        dy = [m for m in zf.namelist() if m.endswith("data.yaml")]
        y = yaml.safe_load(zf.read(dy[0])) if dy else {}
        names = y.get("names", [])
        names = [names[i] for i in sorted(names)] if isinstance(names, dict) else list(names)
        avail, n = _ingest_yolo_zip(zf, root, names, limit, dcfg["seed"])
    if not keep_archives:
        shutil.rmtree(root / "_archives", ignore_errors=True)
    print(f"[{name}] {avail} gambar, diambil {n}")
    write_meta(root, {"source": name, "type": "roboflow", "workspace": s["workspace"], "project": s["project"],
                      "version": s["version"], "available": avail, "downloaded": n, "limit": limit}, names)
    return n


def dl_local_yolo(name: str, s: dict, root: Path, limit: int | None, dcfg: dict, keep_archives: bool) -> int:
    """Folder YOLO lokal (mis. frame DMD hasil training/scripts/dmd_to_yolo.py): <dir>/images, <dir>/labels,
    <dir>/classes.txt. Folder tidak ada -> sumber dilewati (bukan error)."""
    d = rpath(s["local_dir"])
    if not (d / "classes.txt").exists():
        print(f"[{name}] {d} belum ada -> dilewati")
        return 0
    names = [x.strip() for x in (d / "classes.txt").read_text(encoding="utf-8").splitlines() if x.strip()]
    imgs = sorted(p for p in (d / "images").iterdir() if p.suffix.lower() in IMG_EXTS)
    chosen = sample(imgs, limit, dcfg["seed"])
    reset_dir(root)
    for p in chosen:
        dst = root / "images" / p.name
        try:
            os.link(p, dst)          # hardlink: hemat disk (folder sama-sama di /workspace)
        except OSError:
            shutil.copy2(p, dst)
        lab = d / "labels" / f"{p.stem}.txt"
        (root / "labels" / f"{p.stem}.txt").write_text(lab.read_text(encoding="utf-8") if lab.exists() else "",
                                                       encoding="utf-8")
    print(f"[{name}] {len(imgs)} gambar, diambil {len(chosen)}")
    write_meta(root, {"source": name, "type": "local_yolo", "local_dir": str(d), "available": len(imgs),
                      "downloaded": len(chosen), "limit": limit}, names)
    return len(chosen)


HANDLERS = {
    "local_yolo": dl_local_yolo,
    "hf_yolo_files": lambda n, s, r, l, d, k: dl_hf_yolo_files(n, s, r, l, d),
    "hf_coco_zip": dl_hf_coco_zip,
    "hf_yolo_zip": dl_hf_yolo_zip,
    "coco_ultralytics": dl_coco_ultralytics,
    "openimages": dl_openimages,
    "roboflow": dl_roboflow,
}


def enabled_sources(sources: dict, only: list[str] | None) -> dict:
    if only:
        return {k: v for k, v in sources.items() if k in only}
    out = {k: v for k, v in sources.items() if v.get("enabled", False)}
    for k, v in sources.items():  # sumber Roboflow otomatis aktif jika key tersedia dan sudah diisi
        if v.get("type") == "roboflow" and os.environ.get("ROBOFLOW_API_KEY") and "CHANGE_ME" not in str(v.get("project")):
            out[k] = v
    return out


def run(out: Path, limit: int | None, only: list[str] | None, cfg: dict, keep_archives: bool = False) -> dict:
    sources = load_yaml(rpath(cfg["paths"]["sources"]))["sources"]
    dcfg = cfg["download"]
    counts = {}
    for name, s in enabled_sources(sources, only).items():
        lim = limit if limit is not None else s.get("max_images")
        t0 = time.time()
        counts[name] = HANDLERS[s["type"]](name, s, out / name, lim, dcfg, keep_archives)
        print(f"[{name}] selesai: {counts[name]} gambar ({time.time() - t0:.0f} s)")
    return counts


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--out", default=None, help="root data mentah (default: paths.raw di cabin_build.yaml)")
    ap.add_argument("--limit", type=int, default=None, help="maks gambar per sumber (smoke test)")
    ap.add_argument("--only", nargs="*", default=None, help="hanya sumber ini (nama di sources.yaml)")
    ap.add_argument("--config", default=None, help="cabin_build.yaml")
    ap.add_argument("--keep-archives", action="store_true", help="jangan hapus zip setelah ekstraksi")
    a = ap.parse_args()
    cfg = load_build_cfg(a.config)
    out = rpath(a.out or cfg["paths"]["raw"])
    counts = run(out, a.limit, a.only, cfg, a.keep_archives)
    print(json.dumps(counts, indent=2))


if __name__ == "__main__":
    main()
