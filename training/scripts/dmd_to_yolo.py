"""DMD (sesi distraksi s1/s2) -> sumber YOLO kabin `dmd_phone`: frame + kotak ponsel auto-label (cabin_v3).

Anotasi DMD = interval frame per aksi (OpenLABEL), tanpa kotak. Maka:
  positif : frame di dalam interval driver_actions/texting_* | phonecall_*; kotak ponsel dari YOLO-World.
            Frame positif TANPA kotak dibuang (ponsel tertutup tangan tidak boleh diajarkan sebagai latar).
  negatif : frame aksi lain (menyetir, radio, minum, rambut, bicara, ...) tanpa aksi/objek ponsel; frame yang
            oleh YOLO-World tetap terlihat berponsel (conf >= neg_drop_conf) dibuang (label tidak lengkap).
Kamera: wajah (frontal) dan badan (samping), 1280x720. Frame aliran = frame anotasi - sync.frame_shift;
margin di tepi interval menutup ketidakpastian sinkron. Stem: <grup>_<subjek>_<sesi>_<kamera>_f<frame>
(class map dmd_phone.yaml mengelompokkan per subjek -> split per orang).

Lisensi DMD: CC BY-NC-ND 4.0 (lihat training/datasets/LICENSES.md).

  python training/scripts/dmd_to_yolo.py --dmd /workspace/dmd/files --out data/manual/dmd_phone \
      --world data/weights/yolov8x-worldv2.pt --device 0
Keluar dengan kode 3 jika tingkat kotak pada frame positif < --min-hit-rate (auto-label tidak bisa dipercaya).
"""
from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import cv2
import numpy as np

POS = re.compile(r"driver_actions/(texting|phonecall)", re.I)
PHONE_ANY = re.compile(r"(texting|phonecall|cellphone|phone)", re.I)
SKIP = re.compile(r"driver_actions/(unclassified)", re.I)
WORLD_URL = "https://github.com/ultralytics/assets/releases/download/v8.3.0/yolov8x-worldv2.pt"


def load_ann(p: Path) -> dict:
    d = json.loads(p.read_text(encoding="utf-8"))
    return d.get("openlabel") or d.get("vcd") or d


def intervals(o: dict, rx: re.Pattern) -> list[tuple[int, int, str]]:
    out = []
    for a in (o.get("actions") or {}).values():
        t = a.get("type", "")
        if rx.search(t):
            out += [(int(i["frame_start"]), int(i["frame_end"]), t) for i in a.get("frame_intervals", [])]
    return out


def in_any(f: int, iv: list) -> bool:
    return any(a <= f <= b for a, b, _ in iv)


def plan_session(o: dict, a) -> dict[str, list[tuple[int, str]]]:
    """-> {'pos': [(ann_frame, type)], 'neg': [...]} (frame anotasi)."""
    pos_iv = intervals(o, POS)
    phone_iv = intervals(o, PHONE_ANY)
    skip_iv = intervals(o, SKIP)
    plan = {"pos": [], "neg": []}
    for s, e, t in pos_iv:
        plan["pos"] += [(f, t) for f in range(s + a.margin, e - a.margin + 1, a.pos_step)]
    for s, e, t in intervals(o, re.compile(r"driver_actions/", re.I)):
        if POS.search(t) or SKIP.search(t):
            continue
        for f in range(s + a.margin, e - a.margin + 1, a.neg_step):
            if not in_any(f, phone_iv) and not in_any(f, skip_iv):
                plan["neg"].append((f, t))
    return plan


def grab_frames(video: Path, wanted: dict[int, object], total: int | None) -> dict[int, np.ndarray]:
    cap = cv2.VideoCapture(str(video))
    out, i, last = {}, 0, max(wanted) if wanted else -1
    while i <= last:
        if not cap.grab():
            break
        if i in wanted:
            ok, fr = cap.retrieve()
            if ok:
                out[i] = fr
        i += 1
    cap.release()
    return out


class World:
    def __init__(self, weights: str, device, imgsz: int):
        from ultralytics import YOLOWorld

        p = Path(weights)
        if not p.exists():
            import urllib.request

            p.parent.mkdir(parents=True, exist_ok=True)
            urllib.request.urlretrieve(WORLD_URL, p)
        self.m = YOLOWorld(str(p))
        self.m.set_classes(["cell phone", "smartphone", "mobile phone"])
        self.device, self.imgsz = device, imgsz

    def phones(self, imgs: list[np.ndarray], conf: float) -> list[list[tuple[float, tuple]]]:
        res = self.m.predict(imgs, conf=conf, iou=0.5, imgsz=self.imgsz, device=self.device, verbose=False)
        out = []
        for r in res:
            out.append([(float(s), tuple(float(v) for v in b))
                        for s, b in zip(r.boxes.conf.cpu().numpy(), r.boxes.xyxyn.cpu().numpy())])
        return out


def good_box(b, a) -> bool:
    x1, y1, x2, y2 = b
    area = (x2 - x1) * (y2 - y1)
    return a.min_area <= area <= a.max_area


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--dmd", required=True, help="folder hasil ekstraksi (berisi dmd/g*/<subjek>/s*/...)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--world", default="data/weights/yolov8x-worldv2.pt")
    ap.add_argument("--device", default=None)
    ap.add_argument("--imgsz", type=int, default=960)
    ap.add_argument("--sessions", default="s1,s2")
    ap.add_argument("--cameras", default="face,body")
    ap.add_argument("--pos-step", type=int, default=15, help="frame (~0,5 s @30 fps)")
    ap.add_argument("--neg-step", type=int, default=90, help="frame (~3 s)")
    ap.add_argument("--margin", type=int, default=12, help="frame dibuang di tepi interval (sinkron kamera)")
    ap.add_argument("--max-pos-per-session-cam", type=int, default=160)
    ap.add_argument("--max-neg-per-session-cam", type=int, default=40)
    ap.add_argument("--pos-conf", type=float, default=0.15)
    ap.add_argument("--neg-drop-conf", type=float, default=0.30)
    ap.add_argument("--min-area", type=float, default=0.0004)
    ap.add_argument("--max-area", type=float, default=0.12)
    ap.add_argument("--max-side", type=int, default=960)
    ap.add_argument("--min-hit-rate", type=float, default=0.25)
    ap.add_argument("--limit-sessions", type=int, default=None, help="smoke")
    ap.add_argument("--seed", type=int, default=0)
    a = ap.parse_args()

    rnd = random.Random(a.seed)
    out = Path(a.out)
    (out / "images").mkdir(parents=True, exist_ok=True)
    (out / "labels").mkdir(parents=True, exist_ok=True)
    (out / "classes.txt").write_text("phone\n", encoding="utf-8")
    sessions = set(a.sessions.split(","))
    cams = a.cameras.split(",")
    anns = sorted(p for p in Path(a.dmd).rglob("*_ann_distraction.json")
                  if p.parent.name in sessions)
    if a.limit_sessions:
        anns = anns[: a.limit_sessions]
    print(f"{len(anns)} sesi anotasi ({sorted(sessions)})", flush=True)
    world = World(a.world, a.device, a.imgsz)
    st = defaultdict(Counter)
    sheet = []
    for k, jp in enumerate(anns):
        o = load_ann(jp)
        plan = plan_session(o, a)
        g, subj, sess = jp.parent.parent.parent.name, jp.parent.parent.name, jp.parent.name
        streams = o.get("streams") or {}
        for cam in cams:
            sp = streams.get(f"{cam}_camera") or {}
            vid = jp.parent / Path(sp.get("uri", "")).name
            if not vid.exists():
                cand = list(jp.parent.glob(f"*_rgb_{cam}.*"))
                if not cand:
                    st["_"]["missing_video"] += 1
                    continue
                vid = cand[0]
            shift = int(((sp.get("stream_properties") or {}).get("sync") or {}).get("frame_shift", 0))
            total = (sp.get("stream_properties") or {}).get("total_frames")
            pos = plan["pos"][:]
            neg = plan["neg"][:]
            rnd.shuffle(pos)
            rnd.shuffle(neg)
            pos = pos[: a.max_pos_per_session_cam]
            neg = neg[: a.max_neg_per_session_cam]
            want = {}
            for kind, lst in (("pos", pos), ("neg", neg)):
                for f, t in lst:
                    sf = f - shift
                    if sf >= 0 and (total is None or sf < int(total)):
                        want[sf] = (kind, t, f)
            frames = grab_frames(vid, want, total)
            keys = sorted(frames)
            for i in range(0, len(keys), 32):
                chunk = keys[i:i + 32]
                dets = world.phones([frames[f] for f in chunk], a.pos_conf)
                for f, d in zip(chunk, dets):
                    kind, t, af = want[f]
                    img = frames[f]
                    d = [(s, b) for s, b in d if good_box(b, a)]
                    if kind == "pos":
                        st[cam]["pos_frames"] += 1
                        if not d:
                            st[cam]["pos_dropped_no_box"] += 1
                            continue
                        s, b = max(d, key=lambda x: x[0])       # satu ponsel per pengemudi
                        lab = [b]
                        st[cam]["pos_kept"] += 1
                    else:
                        st[cam]["neg_frames"] += 1
                        if any(s >= a.neg_drop_conf for s, _ in d):
                            st[cam]["neg_dropped_phone_seen"] += 1
                            continue
                        lab = []
                        st[cam]["neg_kept"] += 1
                    stem = f"{g}_{subj}_{sess}_{cam}_f{af:06d}"
                    h, w = img.shape[:2]
                    sc = min(1.0, a.max_side / max(h, w))
                    if sc < 1.0:
                        img = cv2.resize(img, (int(w * sc), int(h * sc)), interpolation=cv2.INTER_AREA)
                    cv2.imwrite(str(out / "images" / f"{stem}.jpg"), img, [cv2.IMWRITE_JPEG_QUALITY, 92])
                    (out / "labels" / f"{stem}.txt").write_text(
                        "".join(f"0 {(x1 + x2) / 2:.6f} {(y1 + y2) / 2:.6f} {x2 - x1:.6f} {y2 - y1:.6f}\n"
                                for x1, y1, x2, y2 in lab), encoding="utf-8")
                    if lab and len(sheet) < 400:
                        sheet.append((out / "images" / f"{stem}.jpg", lab[0]))
        print(f"[{k + 1}/{len(anns)}] {jp.parent.parent.parent.name}/{subj}/{sess}: "
              f"{dict(sum((st[c] for c in cams), Counter()))}", flush=True)

    tot = sum((st[c] for c in cams), Counter())
    hit = tot["pos_kept"] / max(1, tot["pos_frames"])
    info = {"per_camera": {c: dict(st[c]) for c in cams}, "total": dict(tot), "pos_hit_rate": hit,
            "missing_video": st["_"]["missing_video"], "args": vars(a)}
    (out / "stats.json").write_text(json.dumps(info, indent=1), encoding="utf-8")
    # lembar kontrol visual: 24 positif acak dengan kotaknya
    tiles = []
    for p, (x1, y1, x2, y2) in rnd.sample(sheet, min(24, len(sheet))):
        im = cv2.imread(str(p))
        h, w = im.shape[:2]
        cv2.rectangle(im, (int(x1 * w), int(y1 * h)), (int(x2 * w), int(y2 * h)), (0, 255, 0), 3)
        tiles.append(cv2.resize(im, (480, 270)))
    while tiles and len(tiles) % 4:
        tiles.append(np.zeros_like(tiles[0]))
    if tiles:
        cv2.imwrite(str(out / "sheet_pos.jpg"), np.vstack([np.hstack(tiles[i:i + 4]) for i in range(0, len(tiles), 4)]))
    print(json.dumps(info["total"]), f"hit_rate={hit:.2f}", flush=True)
    if hit < a.min_hit_rate:
        print(f"ABORT: hit rate {hit:.2f} < {a.min_hit_rate}", flush=True)
        sys.exit(3)


if __name__ == "__main__":
    main()
