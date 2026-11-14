"""Uji konverter dataset kamera depan (L9) dengan fixture sintetis kecil dalam format asli tiap sumber:
COCO json, LOCO (COCO-style + path sesi), Apron (CSV per gambar), CrowdHuman (.odgt).
Tidak ada akses jaringan: handler LOCO/CrowdHuman/Apron diuji lewat zip/folder lokal."""
from __future__ import annotations

import csv
import json
import random
import sys
import zipfile
from pathlib import Path

import numpy as np
import pytest
import yaml

REPO = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO / "training" / "scripts"))

import build_front_dataset as bfd  # noqa: E402
import front_sources as fs  # noqa: E402
from common import Box, imwrite, read_yolo  # noqa: E402
from convert_to_yolo import ClassMap  # noqa: E402

CFG = yaml.safe_load((REPO / "training/configs/front_build.yaml").read_text(encoding="utf-8"))
SOURCES = yaml.safe_load((REPO / "training/datasets/sources_front.yaml").read_text(encoding="utf-8"))["sources"]
CMAPS = REPO / "training/datasets/class_maps"
OURS = CFG["classes"]
C = {n: i for i, n in enumerate(OURS)}
DCFG = {"workers": 2, "retries": 1, "timeout_s": 5, "seed": 0}


def cmap(name: str) -> dict:
    return yaml.safe_load((CMAPS / SOURCES[name]["class_map"]).read_text(encoding="utf-8"))


def parse(lines):
    return [(int(v[0]), *map(float, v[1:])) for v in (ln.split() for ln in lines)]


def approx(t):
    return pytest.approx(t, abs=1e-5)   # label ditulis 6 desimal


def img(path: Path, w: int, h: int, seed: int) -> Path:
    rng = np.random.default_rng(seed)
    imwrite(path, rng.integers(0, 255, (h, w, 3), dtype=np.uint8))
    return path


# ------------------------------------------------------------------ konfigurasi
def test_class_order_matches_spec():
    assert OURS == ["person", "forklift", "pallet_jack", "gse_vehicle", "aircraft_part", "ladder_stand", "obstacle"]


def test_apron_class_map_covers_all_43_names():
    names = SOURCES["apron"]["names"]
    assert len(names) == 43 and names[12] == "Forklift" and names[22] == "Person Wearing Reflective Vests"
    m = cmap("apron")["map"]
    assert set(m) == set(names), f"tidak dipetakan/dibuang eksplisit: {set(names) ^ set(m)}"
    for targets in m.values():
        assert all(t in OURS for t in targets)
    cm = ClassMap(cmap("apron"), names, OURS)
    assert cm.map[12] == [C["forklift"]]
    assert cm.map[22] == [C["person"]] and cm.map[42] == [C["person"]]
    assert cm.map[names.index("Passenger Jet")] == [C["aircraft_part"]]
    assert cm.map[names.index("Tow Vehicle")] == [C["gse_vehicle"]]
    assert cm.map[names.index("Traffic Cone - On")] == [C["obstacle"]]
    assert cm.map[names.index("Traffic Sign")] == []


def test_all_front_class_maps_target_known_classes():
    for name, s in SOURCES.items():
        cm = cmap(name)
        for targets in (cm.get("map") or {}).values():
            assert all(t in OURS for t in targets), name
        assert all(t in OURS for t in cm.get("labelled", [])), name
        assert all(t in CFG["teacher_fill"]["map"] for t in (cm.get("derive") or {}).get("teacher_fill") or []), name


# ------------------------------------------------------------------ COCO json
def test_coco_json_to_yolo():
    j = {"images": [{"id": 1, "width": 200, "height": 100, "file_name": "a.jpg"}],
         "categories": [{"id": 9, "name": "truck"}, {"id": 1, "name": "person"}],
         "annotations": [
             {"image_id": 1, "category_id": 1, "bbox": [10, 20, 40, 60], "iscrowd": 0},
             {"image_id": 1, "category_id": 9, "bbox": [180, 50, 40, 80], "iscrowd": 0},   # keluar gambar -> clip
             {"image_id": 1, "category_id": 1, "bbox": [0, 0, 50, 50], "iscrowd": 1},      # crowd -> buang
         ]}
    names, idx, anns = fs.coco_index(j)
    assert names == ["person", "truck"] and idx == {1: 0, 9: 1}
    got = parse(fs.coco_anns_to_yolo(anns[1], 200, 100, idx))
    assert len(got) == 2
    assert got[0] == approx((0, 30 / 200, 50 / 100, 40 / 200, 60 / 100))
    assert got[1] == approx((1, 190 / 200, 75 / 100, 20 / 200, 50 / 100))   # x 180..200, y 50..100


# ------------------------------------------------------------------ LOCO
LOCO_CATS = [{"id": 3, "name": "small_load_carrier"}, {"id": 5, "name": "forklift"}, {"id": 7, "name": "pallet"},
             {"id": 10, "name": "stillage"}, {"id": 11, "name": "pallet_truck"}]


def test_loco_member_and_stem():
    m, s = fs.loco_member_and_stem("/dataset/subset-5/2020-03-05_1/RealSense/color/0/1583416214257,48.jpg")
    assert m == "dataset/subset-5/2020-03-05_1/RealSense/color/0/1583416214257,48.jpg"
    assert s == "subset-5__2020-03-05_1__RealSense__color__0__1583416214257_48"
    assert fs.loco_member_and_stem("/dataset/subset-1/1564562430367,07.jpg")[1] == "subset-1__1564562430367_07"


def test_loco_class_map_and_groups():
    names = ["small_load_carrier", "forklift", "pallet", "stillage", "pallet_truck"]
    cmd = cmap("loco")
    m = ClassMap(cmd, names, OURS)
    assert m.map[1] == [C["forklift"]] and m.map[4] == [C["pallet_jack"]]
    assert m.map[0] == [] and m.map[2] == [] and m.map[3] == []          # palet/SLC/stillage dibuang
    g = lambda st: bfd.group_of(m, cmd, "loco", st)  # noqa: E731
    assert g("subset-5__2020-03-05_1__RealSense__color__0__158_48") == "loco:subset-5__2020-03-05_1"
    assert g("subset-4__2020-01-16_17__cam2__1579163807.9130263") == "loco:subset-4__2020-01-16_17"
    assert g("subset-1__1564562430367_07") == "loco:subset-1"
    assert g("aneh") == "loco:__nomatch__"


def _loco_fixture(tmp: Path) -> tuple[Path, Path]:
    """JSON LOCO-style + dataset.zip lokal dengan 3 sesi."""
    images, anns, members = [], [], {}
    paths = [f"/dataset/subset-{s}/{sess}/cam/{t},5.jpg" for s, sess in ((2, "2019-11-25_0"), (5, "2020-03-05_1"))
             for t in range(1000, 1004)] + [f"/dataset/subset-1/{t}.25.jpg" for t in range(10, 13)]
    aid = 0
    for i, p in enumerate(paths):
        images.append({"id": i, "path": p, "width": 64, "height": 48, "file_name": p.rsplit("/", 1)[1]})
        for cid, bb in ((5, [4, 4, 20, 20]), (11, [30, 10, 16, 20]), (7, [40, 30, 10, 10])):
            anns.append({"id": aid, "image_id": i, "category_id": cid, "bbox": bb, "iscrowd": False, "segmentation": []})
            aid += 1
        members[p.lstrip("/")] = img(tmp / "src" / f"{i}.jpg", 64, 48, i).read_bytes()
    jd = tmp / "ann"
    jd.mkdir(parents=True)
    (jd / "loco-test.json").write_text(json.dumps({"images": images, "categories": LOCO_CATS, "annotations": anns}))
    zp = tmp / "dataset.zip"
    with zipfile.ZipFile(zp, "w") as z:
        for k, v in members.items():
            z.writestr(k, v)
    return jd, zp


def test_dl_loco_from_local_zip(tmp_path):
    jd, zp = _loco_fixture(tmp_path)
    s = {**SOURCES["loco"], "annotation_files": ["loco-test.json"], "local_dir": str(jd), "local_zip": str(zp)}
    root = tmp_path / "raw" / "loco"
    n = fs.dl_loco("loco", s, root, None, DCFG, False)
    assert n == 11
    assert (root / "classes.txt").read_text().split() == ["small_load_carrier", "forklift", "pallet", "stillage", "pallet_truck"]
    lab = root / "labels" / "subset-5__2020-03-05_1__cam__1000_5.txt"
    got = parse(lab.read_text().splitlines())
    assert [g[0] for g in got] == [1, 4, 2]                 # indeks upstream urut id kategori (5->1, 11->4, 7->2)
    assert got[0] == approx((1, 14 / 64, 14 / 48, 20 / 64, 20 / 48))
    assert (root / "images" / "subset-1__10.25.jpg").exists()


# ------------------------------------------------------------------ Apron
def test_apron_csv_to_yolo_column_order_independent():
    text = "occluded,bry,label,tlx,brx,tly,atmosphere\n0,80,12,10,50,20,0\n1,120,42,90,110,40,0\nx,1,nan,1,1,1,0\n"
    got = parse(fs.apron_csv_to_yolo(text, 100, 100))
    assert got[0] == approx((12, 0.30, 0.50, 0.40, 0.60))
    assert got[1] == approx((42, 0.95, 0.70, 0.10, 0.60))   # brx 110 -> clip 100
    assert len(got) == 2
    # BOM + spasi di header tetap terbaca; header tanpa kolom wajib -> error (bukan gambar tanpa label)
    assert len(fs.apron_csv_to_yolo("﻿label, tlx, tly, brx, bry\n12,10,20,50,80\n", 100, 100)) == 1
    with pytest.raises(ValueError):
        fs.apron_csv_to_yolo("cls,x1,y1,x2,y2\n12,10,20,50,80\n", 100, 100)
    assert fs.apron_csv_to_yolo("", 100, 100) == []


def test_dl_apron_local(tmp_path):
    d = tmp_path / "apron"
    for seq in ("seqA", "seqB"):
        for k in range(2):
            img(d / "images" / seq / f"f{k}.png", 80, 40, k)
            (d / "bboxes" / seq).mkdir(parents=True, exist_ok=True)
            (d / "bboxes" / seq / f"f{k}.csv").write_text("label,tlx,tly,brx,bry,occluded\n31,0,0,40,20,0\n16,40,20,80,40,0\n")
    img(d / "images" / "seqA" / "nolabel.png", 80, 40, 9)       # tanpa CSV -> dilewati
    root = tmp_path / "raw" / "apron"
    n = fs.dl_apron_local("apron", {**SOURCES["apron"], "local_dir": str(d)}, root, None, DCFG, False)
    assert n == 4
    assert len((root / "classes.txt").read_text(encoding="utf-8").splitlines()) == 43
    got = parse((root / "labels" / "seqB__f1.txt").read_text().splitlines())
    assert got[0] == approx((31, 0.25, 0.25, 0.5, 0.5))
    cmd = cmap("apron")
    m = ClassMap(cmd, SOURCES["apron"]["names"], OURS)
    assert bfd.group_of(m, cmd, "apron", "seqB__f1") == "apron:seqB"
    assert bfd.group_of(m, cmd, "apron", "rec07_000123") == "apron:rec07"


def test_manual_sources_missing_are_skipped(tmp_path):
    s = {**SOURCES["apron"], "local_dir": str(tmp_path / "tidak_ada")}
    assert fs.dl_apron_local("apron", s, tmp_path / "raw", None, DCFG, False) == 0
    s = {**SOURCES["synthetic"], "local_dir": str(tmp_path / "tidak_ada")}
    assert fs.dl_yolo_local("synthetic", s, tmp_path / "raw", None, DCFG, False) == 0


# ------------------------------------------------------------------ CrowdHuman
def test_crowdhuman_to_yolo():
    rec = {"ID": "273271,c9db000d5146c15", "gtboxes": [
        {"tag": "person", "fbox": [-10, 10, 40, 100], "vbox": [0, 10, 30, 60], "hbox": [5, 10, 10, 10],
         "extra": {"box_id": 0, "occ": 1}},
        {"tag": "person", "vbox": [50, 0, 20, 20], "extra": {"ignore": 1}},       # ignore -> buang
        {"tag": "mask", "vbox": [70, 0, 20, 20], "extra": {"ignore": 1}},         # kerumunan -> buang
    ]}
    assert parse(fs.crowdhuman_to_yolo(rec, 100, 100)) == [approx((0, 0.15, 0.40, 0.30, 0.60))]
    got = parse(fs.crowdhuman_to_yolo(rec, 100, 100, box="fbox"))
    assert got == [approx((0, 0.15, 0.55, 0.30, 0.90))]                    # fbox x -10 -> clip 0
    assert fs.crowdhuman_stem(rec["ID"]) == "273271_c9db000d5146c15"


def test_dl_crowdhuman_local(tmp_path):
    d = tmp_path / "ch"
    ids = ["273271,aaa", "273271,bbb"]
    d.mkdir()
    with open(d / "annotation_train.odgt", "w") as f:
        for i in ids:
            img(d / "Images" / f"{i}.jpg", 50, 100, 1)
            f.write(json.dumps({"ID": i, "gtboxes": [{"tag": "person", "vbox": [5, 10, 20, 50], "extra": {}}]}) + "\n")
    root = tmp_path / "raw" / "crowdhuman"
    n = fs.dl_crowdhuman("crowdhuman", {**SOURCES["crowdhuman"], "local_dir": str(d)}, root, None, DCFG, False)
    assert n == 2
    assert parse((root / "labels" / "273271_bbb.txt").read_text().splitlines()) == [approx((0, 0.3, 0.35, 0.4, 0.5))]


# ------------------------------------------------------------------ teacher fill + mast strip
def test_teacher_boxes_only_fill_classes_and_dedupe():
    existing = [Box.from_xyxy(C["person"], 0.1, 0.1, 0.3, 0.5)]
    preds = [("person", (0.1, 0.1, 0.3, 0.5), 0.9),       # duplikat label -> buang
             ("person", (0.6, 0.1, 0.8, 0.5), 0.8),       # baru -> pseudo person
             ("car", (0.0, 0.6, 0.4, 1.0), 0.9),          # -> gse_vehicle
             ("airplane", (0.5, 0.5, 1.0, 1.0), 0.9),     # aircraft_part tidak di fill -> buang
             ("dog", (0.0, 0.0, 0.1, 0.1), 0.9)]
    out = bfd.teacher_boxes(preds, ["person", "gse_vehicle"], existing, OURS, CFG["teacher_fill"]["map"], 0.5)
    assert sorted(b.cls for b in out) == [C["person"], C["gse_vehicle"]]
    assert all(b.pseudo for b in out)
    # forklift yang sudah dilabel tidak boleh mendapat pseudo gse_vehicle dari prediksi COCO "truck"
    fk = [Box.from_xyxy(C["forklift"], 0.2, 0.2, 0.6, 0.8)]
    assert bfd.teacher_boxes([("truck", (0.21, 0.2, 0.6, 0.79), 0.9)], ["gse_vehicle"], fk, OURS,
                             CFG["teacher_fill"]["map"], 0.5) == []


def test_mast_strip_drops_hidden_boxes():
    mc = {**CFG["augment"]["mast_strip"], "p": 1.0, "width_frac": [0.10, 0.10]}
    rnd = random.Random(3)
    im = np.full((100, 200, 3), 200, np.uint8)
    # strip lebar 20 px; posisinya acak -> ulangi urutan RNG yang sama untuk tahu x0
    r2 = random.Random(3)
    r2.random()
    r2.uniform(0.10, 0.10)
    x0 = r2.randint(0, 180)
    hidden = Box.from_xyxy(0, (x0 + 2) / 200, 0.1, (x0 + 18) / 200, 0.9)      # tertutup 100%
    fx = 160 if x0 < 90 else 0
    far = Box.from_xyxy(1, fx / 200, 0.1, (fx + 30) / 200, 0.3)                # tidak tersentuh
    out, kept = bfd.mast_strip(im, [hidden, far], mc, rnd)
    assert kept == [far]
    assert (out[:, x0:x0 + 20] <= 60).all() and out[:, x0:x0 + 20].std() == 0
    assert bfd.mast_strip(im.copy(), [hidden], {**mc, "p": 0.0}, random.Random(0))[1] == [hidden]


# ------------------------------------------------------------------ convert + split end-to-end
def _raw_source(root: Path, name: str, classes: list[str], stems_boxes: dict[str, list[str]], seed: int) -> None:
    (root / name / "labels").mkdir(parents=True)
    (root / name / "classes.txt").write_text("\n".join(classes) + "\n")
    for k, (stem, lines) in enumerate(stems_boxes.items()):
        img(root / name / "images" / f"{stem}.jpg", 64, 48, seed * 1000 + k)
        (root / name / "labels" / f"{stem}.txt").write_text("".join(ln + "\n" for ln in lines))


def test_convert_and_split_never_mixes_sequences(tmp_path):
    raw = tmp_path / "raw"
    loco_names = ["small_load_carrier", "forklift", "pallet", "stillage", "pallet_truck"]
    loco = {}
    for sub, sessions in ((1, [None]), (2, ["2019-11-25_0", "2019-11-25_1"]), (3, ["2019-12-17_9"]),
                          (4, ["2020-01-16_15"]), (5, ["2020-03-05_1", "2020-03-05_2"])):
        for sess in sessions:
            for t in range(4):
                stem = f"subset-{sub}__{sess}__cam__{t}" if sess else f"subset-{sub}__{t}"
                loco[stem] = ["1 0.3 0.3 0.4 0.4", "4 0.7 0.7 0.2 0.3", "2 0.5 0.5 0.1 0.1"]
    _raw_source(raw, "loco", loco_names, loco, 1)
    apron = {f"seq{s}__f{t}": ["31 0.5 0.5 0.5 0.5", "22 0.2 0.2 0.1 0.3"] for s in range(12) for t in range(3)}
    _raw_source(raw, "apron", SOURCES["apron"]["names"], apron, 2)

    cfg = {**CFG, "split": {**CFG["split"]}}
    srcs = tmp_path / "sources.yaml"
    srcs.write_text(yaml.safe_dump({"sources": {"loco": SOURCES["loco"], "apron": SOURCES["apron"]}}))
    cfg["paths"] = {**CFG["paths"], "sources": str(srcs), "class_maps": str(CMAPS)}
    man = bfd.convert_all(raw, tmp_path / "merged", cfg, ["loco", "apron"], teacher=None)
    rows = list(csv.DictReader(open(man, encoding="utf-8")))
    assert len(rows) == len(loco) + len(apron)

    # label hasil konversi: LOCO forklift->1, pallet_truck->2, palet dibuang
    lab = read_yolo(tmp_path / "merged" / "labels" / "loco__subset-5__2020-03-05_1__cam__0.txt")
    assert [b.cls for b in lab] == [C["forklift"], C["pallet_jack"]]
    lab = read_yolo(tmp_path / "merged" / "labels" / "apron__seq3__f1.txt")
    assert [b.cls for b in lab] == [C["gse_vehicle"], C["person"]]

    bfd.check_sources(rows, OURS)                       # tidak berhenti: tiap sumber punya label
    seq = bfd.sequence_sources(cfg, ["loco", "apron"])
    assert seq == {"loco", "apron"}
    split_of = bfd.assign(rows, tmp_path / "merged" / "images", cfg["split"], seq)
    seq_splits: dict[str, set] = {}
    for r in rows:
        seq_splits.setdefault(r["group"], set()).add(split_of[r["file"]])
    assert all(len(s) == 1 for s in seq_splits.values()), seq_splits
    assert len([g for g in seq_splits if g.startswith("apron:")]) == 12          # per sekuens, bukan per frame
    # LOCO per lokasi: subset-1 val, subset-4 test, sisanya train
    for g, s in seq_splits.items():
        if g.startswith("loco:subset-1"):
            assert s == {"val"}
        elif g.startswith("loco:subset-4"):
            assert s == {"test"}
        elif g.startswith("loco:"):
            assert s == {"train"}

    out = tmp_path / "front"
    bfd.materialize(tmp_path / "merged", rows, split_of, out, True)
    for sp in bfd.SPLITS:
        imgs = {p.stem for p in (out / "images" / sp).iterdir()}
        assert imgs == {p.stem for p in (out / "labels" / sp).iterdir()}
    n_aug = bfd.augment_train(out / "images" / "train", out / "labels" / "train", {**CFG["augment"], "copies_per_image": 1}, 90)
    assert n_aug == sum(1 for r in rows if split_of[r["file"]] == "train")


def test_split_dhash_merges_duplicates_across_groups(tmp_path):
    d = tmp_path / "images"
    rows = []
    for k in range(6):
        img(d / f"a{k}.jpg", 64, 48, 7 if k < 2 else 100 + k)       # a0, a1 identik
        rows.append({"file": f"a{k}.jpg", "group": f"coco_person:a{k}"})
    split_of = bfd.assign(rows, d, {**CFG["split"], "fixed": []})
    assert split_of["a0.jpg"] == split_of["a1.jpg"]


def test_split_no_dhash_for_sequence_sources(tmp_path):
    """Frame mirip dari dua sesi LOCO berbeda tidak boleh digabung dHash (aturan lokasi tetap berlaku)."""
    d = tmp_path / "images"
    img(d / "x.jpg", 64, 48, 7)
    img(d / "y.jpg", 64, 48, 7)                                     # identik
    rows = [{"file": "x.jpg", "group": "loco:subset-1", "source": "loco"},
            {"file": "y.jpg", "group": "loco:subset-4__2020-01-16_15", "source": "loco"}]
    split_of = bfd.assign(rows, d, CFG["split"], {"loco"})
    assert split_of == {"x.jpg": "val", "y.jpg": "test"}


def test_check_sources_stops_on_silent_conversion_failure():
    row = {"file": "a.jpg", "source": "apron", "group": "apron:s", "labelled": "person|gse_vehicle",
           **{f"n_{c}": 0 for c in OURS}, **{f"pseudo_{c}": 0 for c in OURS}}
    with pytest.raises(SystemExit):
        bfd.check_sources([row], OURS)
    bfd.check_sources([{**row, "n_person": 1}], OURS)


def test_apron_official_split_folders_are_honoured():
    cmd = cmap("apron")
    m = ClassMap(cmd, SOURCES["apron"]["names"], OURS)
    rows = [{"file": f"{sp}{k}.jpg", "group": bfd.group_of(m, cmd, "apron", f"{sp}__seq{k}__f0"), "source": "apron"}
            for sp in ("train", "val", "test") for k in range(3)]
    split_of = bfd.assign(rows, None, CFG["split"], {"apron"})
    assert all(split_of[r["file"]] == r["file"][:-5] for r in rows)
