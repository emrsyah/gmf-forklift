"""Kalibrasi intrinsik kamera depan dengan papan catur (Tech Spec bag. 6, langkah 1).

Hasil (K, dist, image_size, fisheye, rms, setelan capture) ditulis ke blok `camera:` di configs/front_camera.yaml;
blok lain (mount, distance, motion, classes, zones) DIPERTAHANKAN.

Langkah:
    # 1) buat papan untuk dicetak (A4 landscape, cetak "Actual size / 100%", lalu UKUR sisi kotak dengan penggaris)
    python tools/calibrate_camera.py --make-board data/calib/board_9x6_25mm.png --pattern 9x6 --square-mm 25

    # 2) live. Setelan kamera (resolusi, MJPG, fokus dikunci) diambil dari runtime.yaml front.source,
    #    jadi K berlaku untuk setelan yang sama dengan tools/run_front.py.
    #    SPASI/c = ambil view, a = ambil otomatis on/off, u = batalkan view terakhir, q/ESC = selesai & kalibrasi
    python tools/calibrate_camera.py --source webcam:1 --pattern 9x6 --square-mm 25 --save-views data/calib/front

    # ulang dari gambar tersimpan (mis. dengan --fisheye atau setelah membuang view jelek)
    python tools/calibrate_camera.py --images "data/calib/front/*.png" --pattern 9x6 --square-mm 25
    # video rekaman, ambil otomatis tiap 1 s tanpa jendela
    python tools/calibrate_camera.py --source calib.mp4 --auto-every-s 1.0 --no-show

--pattern = jumlah sudut DALAM (kolom x baris), bukan jumlah kotak (papan 10x7 kotak = 9x6).
"""
from __future__ import annotations

import argparse
import datetime as _dt
import glob
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import cv2  # noqa: E402
import numpy as np  # noqa: E402
import yaml  # noqa: E402

from src.config import CONFIG_DIR, load_yaml, resolve_path  # noqa: E402

_SUBPIX_CRIT = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)
_HEADER = ("# Kalibrasi + parameter kamera depan (milestone L9). Lihat docs/FRONT_CAMERA.md.\n"
           "# Blok `camera` ditulis oleh tools/calibrate_camera.py; blok lain diisi/diukur manual.\n")
_COVER_GRID = (64, 36)          # resolusi peta cakupan (w, h)
_CAPTURE_KEYS = ("type", "index", "width", "height", "fps", "api", "fourcc", "autofocus", "focus")


# --------------------------------------------------------------------------------------------- papan

def make_board(path: Path, cols: int, rows: int, square_mm: float, dpi: int = 300,
               paper_mm: tuple[float, float] = (297.0, 210.0)) -> Path:
    """PNG papan catur (cols+1 x rows+1 kotak) di kertas A4 landscape, dengan metadata DPI agar ukuran cetak benar."""
    px_per_mm = dpi / 25.4
    board_w, board_h = (cols + 1) * square_mm, (rows + 1) * square_mm
    if board_w > paper_mm[0] - 10 or board_h > paper_mm[1] - 20:
        raise ValueError(f"Papan {board_w:.0f}x{board_h:.0f} mm tidak muat di kertas {paper_mm[0]:.0f}x{paper_mm[1]:.0f} mm; "
                         f"kecilkan --square-mm")
    W, H = round(paper_mm[0] * px_per_mm), round(paper_mm[1] * px_per_mm)
    img = np.full((H, W), 255, np.uint8)
    sq = square_mm * px_per_mm
    x0 = (W - (cols + 1) * sq) / 2
    y0 = (H - (rows + 1) * sq) / 2 - 4 * px_per_mm
    for r in range(rows + 1):
        for c in range(cols + 1):
            if (r + c) % 2 == 0:
                cv2.rectangle(img, (round(x0 + c * sq), round(y0 + r * sq)),
                              (round(x0 + (c + 1) * sq) - 1, round(y0 + (r + 1) * sq) - 1), 0, -1)
    # garis 50 mm untuk cek skala hasil cetak
    ly = round(y0 + (rows + 1) * sq + 7 * px_per_mm)
    lx = round(x0)
    cv2.line(img, (lx, ly), (lx + round(50 * px_per_mm), ly), 0, max(2, round(0.4 * px_per_mm)))
    label = (f"pattern {cols}x{rows} (sudut dalam), kotak {square_mm:g} mm. Garis = 50 mm: ukur setelah dicetak. "
             f"Cetak 100% / actual size.")
    cv2.putText(img, label, (lx + round(55 * px_per_mm), ly + round(1.5 * px_per_mm)), cv2.FONT_HERSHEY_SIMPLEX,
                0.9 * dpi / 300, 0, max(1, dpi // 150), cv2.LINE_AA)
    path.parent.mkdir(parents=True, exist_ok=True)
    try:
        from PIL import Image
        Image.fromarray(img).save(path, dpi=(dpi, dpi))
    except ImportError:          # tanpa PIL: DPI tidak tersimpan, atur skala cetak manual
        cv2.imwrite(str(path), img)
    return path


def board_points(cols: int, rows: int, square_m: float) -> np.ndarray:
    objp = np.zeros((rows * cols, 3), np.float64)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square_m
    return objp


# --------------------------------------------------------------------------------------------- deteksi

def find_corners(gray: np.ndarray, cols: int, rows: int, detect_max_side: int | None = None) -> np.ndarray | None:
    """Sudut dalam (N,2) di resolusi penuh, atau None.

    detect_max_side: cari papan di gambar yang diperkecil (live tetap lancar di 720p/1080p), lalu
    sudut dihaluskan (cornerSubPix) di resolusi penuh, jadi akurasi tidak turun.
    """
    h, w = gray.shape[:2]
    scale = 1.0
    small = gray
    if detect_max_side and max(h, w) > detect_max_side:
        scale = detect_max_side / max(h, w)
        small = cv2.resize(gray, (round(w * scale), round(h * scale)), interpolation=cv2.INTER_AREA)
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_FAST_CHECK
    ok, corners = cv2.findChessboardCorners(small, (cols, rows), flags)
    if not ok:
        return None
    corners = corners.reshape(-1, 2).astype(np.float32) / scale
    # jendela subpix ~ setengah jarak antar-sudut terdekat, dibatasi 3..11 px
    d = np.linalg.norm(np.diff(corners.reshape(rows, cols, 2), axis=1), axis=2).min()
    win = int(np.clip(d / 2 - 1, 3, 11))
    corners = cv2.cornerSubPix(gray, corners.reshape(-1, 1, 2), (win, win), (-1, -1), _SUBPIX_CRIT)
    return corners.reshape(-1, 2).astype(np.float64)


class Coverage:
    """Peta area frame yang sudah tertutup papan. Kalibrasi butuh sudut sampai tepi frame (distorsi)."""

    def __init__(self, size: tuple[int, int]):
        self.size = size
        self.mask = np.zeros((_COVER_GRID[1], _COVER_GRID[0]), np.uint8)

    def add(self, corners: np.ndarray) -> None:
        sx, sy = _COVER_GRID[0] / self.size[0], _COVER_GRID[1] / self.size[1]
        hull = cv2.convexHull((corners * [sx, sy]).astype(np.int32))
        cv2.fillConvexPoly(self.mask, hull, 1)

    def rebuild(self, views: list[np.ndarray]) -> None:
        self.mask[:] = 0
        for c in views:
            self.add(c)

    @property
    def percent(self) -> float:
        return 100.0 * float(self.mask.mean())

    def cells_3x3(self) -> list[bool]:
        """Apakah tiap sel 3x3 (kiri-atas ... kanan-bawah) sudah >= 50% tertutup."""
        h, w = self.mask.shape
        return [bool(self.mask[r * h // 3:(r + 1) * h // 3, c * w // 3:(c + 1) * w // 3].mean() >= 0.5)
                for r in range(3) for c in range(3)]

    def draw(self, vis: np.ndarray) -> None:
        big = cv2.resize(self.mask, (vis.shape[1], vis.shape[0]), interpolation=cv2.INTER_NEAREST).astype(bool)
        vis[big] = (vis[big] * 0.7 + np.array([0, 90, 0])).astype(np.uint8)
        h, w = vis.shape[:2]
        for i, ok in enumerate(self.cells_3x3()):
            if not ok:
                r, c = divmod(i, 3)
                cv2.rectangle(vis, (c * w // 3 + 4, r * h // 3 + 4), ((c + 1) * w // 3 - 4, (r + 1) * h // 3 - 4),
                              (0, 0, 255), 1)


# --------------------------------------------------------------------------------------------- kalibrasi

def calibrate_views(obj_list: list[np.ndarray], img_list: list[np.ndarray], image_size: tuple[int, int],
                    fisheye: bool = False) -> tuple[float, np.ndarray, np.ndarray, list[float]]:
    """(rms_px, K 3x3, dist 1-D, error RMS per view px). image_size = (w, h)."""
    if fisheye:
        # bentuk (1,N,3) / (1,N,2): imgPoints (N,1,2) ditolak cv2.fisheye.calibrate di OpenCV 5.0
        objp = [o.reshape(1, -1, 3).astype(np.float64) for o in obj_list]
        imgp = [i.reshape(1, -1, 2).astype(np.float64) for i in img_list]
        K = np.zeros((3, 3))
        D = np.zeros((4, 1))
        # OpenCV 4.x: cv2.fisheye.CALIB_*; OpenCV 5.x: cv2.CALIB_* (Jetson biasanya masih 4.x)
        flag = lambda n: getattr(cv2.fisheye, n, None) or getattr(cv2, n)  # noqa: E731
        flags = flag("CALIB_RECOMPUTE_EXTRINSIC") | flag("CALIB_FIX_SKEW")
        crit = (cv2.TERM_CRITERIA_COUNT + cv2.TERM_CRITERIA_EPS, 100, 1e-6)
        rms, K, D, rvecs, tvecs = cv2.fisheye.calibrate(objp, imgp, image_size, K, D, flags=flags, criteria=crit)
        errs = []
        for o, i, rv, tv in zip(objp, imgp, rvecs, tvecs):
            proj, _ = cv2.fisheye.projectPoints(o, rv, tv, K, D)
            errs.append(float(np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - i.reshape(-1, 2)) ** 2, axis=1)))))
        return float(rms), K, D.ravel(), errs
    objp = [o.reshape(-1, 3).astype(np.float32) for o in obj_list]
    imgp = [i.reshape(-1, 1, 2).astype(np.float32) for i in img_list]
    rms, K, D, rvecs, tvecs = cv2.calibrateCamera(objp, imgp, image_size, None, None)
    errs = []
    for o, i, rv, tv in zip(objp, imgp, rvecs, tvecs):
        proj, _ = cv2.projectPoints(o, rv, tv, K, D)
        errs.append(float(np.sqrt(np.mean(np.sum((proj.reshape(-1, 2) - i.reshape(-1, 2)) ** 2, axis=1)))))
    return float(rms), K, D.ravel(), errs


def calibrate(obj_list: list[np.ndarray], img_list: list[np.ndarray], image_size: tuple[int, int],
              fisheye: bool = False) -> tuple[float, np.ndarray, np.ndarray]:
    """Mengembalikan (rms_px, K 3x3, dist 1-D). image_size = (w, h)."""
    rms, K, D, _ = calibrate_views(obj_list, img_list, image_size, fisheye)
    return rms, K, D


def calibrate_robust(objp: np.ndarray, img_pts: list[np.ndarray], image_size: tuple[int, int], fisheye: bool,
                     reject_factor: float, reject_floor_px: float, min_views: int):
    """Kalibrasi, buang view dengan error > max(reject_factor x median, reject_floor_px), kalibrasi ulang sekali.

    Mengembalikan (rms, K, dist, view yang dipakai, indeks view yang dibuang).
    """
    rms, K, D, errs = calibrate_views([objp] * len(img_pts), img_pts, image_size, fisheye)
    limit = max(reject_factor * float(np.median(errs)), reject_floor_px)
    bad = [i for i, e in enumerate(errs) if e > limit]
    if bad and len(img_pts) - len(bad) >= min_views:
        keep = [p for i, p in enumerate(img_pts) if i not in bad]
        rms, K, D, _ = calibrate_views([objp] * len(keep), keep, image_size, fisheye)
        return rms, K, D, keep, bad, errs
    return rms, K, D, img_pts, [], errs


def write_camera_block(out_path: Path, K: np.ndarray, dist: np.ndarray, image_size: tuple[int, int],
                       fisheye: bool, rms: float, n_views: int, capture: dict | None = None) -> None:
    data = {}
    if out_path.exists():
        with open(out_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    cam = {
        "calibrated": True,
        "image_size": [int(image_size[0]), int(image_size[1])],
        "fisheye": bool(fisheye),
        "K": [[round(float(v), 4) for v in row] for row in np.asarray(K)],
        "dist": [round(float(v), 8) for v in np.asarray(dist).ravel()],
        "rms_px": round(float(rms), 4),
        "n_views": int(n_views),
        "calibrated_at": _dt.datetime.now().isoformat(timespec="seconds"),
    }
    if capture:
        # setelan kamera saat kalibrasi: K hanya berlaku jika run_front memakai setelan (terutama fokus) yang sama
        cam["capture"] = {k: capture[k] for k in _CAPTURE_KEYS if capture.get(k) is not None}
    data["camera"] = cam
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(_HEADER)
        yaml.safe_dump(data, f, sort_keys=False, default_flow_style=None, allow_unicode=True)


# --------------------------------------------------------------------------------------------- pengumpulan view

def collect_from_images(patterns: list[str], cols: int, rows: int):
    paths: list[str] = []
    for p in patterns:
        paths.extend(sorted(glob.glob(p)) or ([p] if Path(p).is_file() else []))
    img_pts, size = [], None
    for p in paths:
        img = cv2.imread(p)
        if img is None:
            print(f"  lewati (tidak terbaca): {p}")
            continue
        s = (img.shape[1], img.shape[0])
        if size is None:
            size = s
        elif s != size:
            print(f"  lewati (ukuran {s} != {size}): {p}")
            continue
        c = find_corners(cv2.cvtColor(img, cv2.COLOR_BGR2GRAY), cols, rows)
        print(f"  {'OK ' if c is not None else '-- '} {p}")
        if c is not None:
            img_pts.append(c)
    return img_pts, size


def source_config(source_arg: str, runtime: dict | None) -> dict:
    """Webcam: setelan front.source di runtime.yaml (resolusi, MJPG, fokus) + index dari --source."""
    from src.capture import parse_source_arg

    parsed = parse_source_arg(source_arg)
    if parsed["type"] == "webcam" and runtime is not None:
        cfg = {k: v for k, v in (runtime.get("front", {}).get("source") or {}).items()
               if k in _CAPTURE_KEYS}
        cfg.update(parsed)
        return cfg
    parsed.setdefault("realtime", False)
    return parsed


def collect_live(src_cfg: dict, cols: int, rows: int, show: bool, auto_every_s: float,
                 save_dir: Path | None, detect_max_side: int, max_motion_px: float, min_diff_px: float):
    from src.capture import make_source

    img_pts: list[np.ndarray] = []
    size = None
    last_t = -1e9
    prev = None                      # sudut frame sebelumnya (untuk cek papan diam)
    auto = auto_every_s > 0
    auto_s = auto_every_s if auto_every_s > 0 else 1.0
    cover = None
    win = "Kalibrasi kamera depan (SPASI ambil, a auto, u undo, q selesai)"
    status = ""
    with make_source(src_cfg) as source:
        cap = getattr(source, "cap", None)
        if cap is not None and src_cfg.get("type") == "webcam":
            v = int(cap.get(cv2.CAP_PROP_FOURCC))
            fcc = "".join(chr((v >> 8 * k) & 0xFF) for k in range(4)) if v > 0 else "?"
            print(f"[kamera] {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))} "
                  f"{fcc}, autofocus={cap.get(cv2.CAP_PROP_AUTOFOCUS)}, focus={cap.get(cv2.CAP_PROP_FOCUS)}")
        for frame in source:
            img = frame.image
            if size is None:
                size = (img.shape[1], img.shape[0])
                cover = Coverage(size)
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            corners = find_corners(gray, cols, rows, detect_max_side)
            motion = None
            if corners is not None and prev is not None:
                motion = float(np.mean(np.linalg.norm(corners - prev, axis=1)))
            prev = corners
            want = False
            key = -1
            if show:
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key == ord("a"):
                    auto = not auto
                if key == ord("u") and img_pts:
                    img_pts.pop()
                    cover.rebuild(img_pts)
                    status = f"view {len(img_pts) + 1} dibatalkan"
                    print(f"  {status}")
                want = key in (ord(" "), ord("c"))
            if auto and frame.video_t_s - last_t >= auto_s:
                want = want or corners is not None
            if want:
                if corners is None:
                    status = "papan tidak terdeteksi"
                elif motion is not None and motion > max_motion_px:
                    status = f"papan bergerak ({motion:.1f} px), tahan diam"
                elif img_pts and min(float(np.mean(np.linalg.norm(corners - p, axis=1))) for p in img_pts) < min_diff_px:
                    status = "terlalu mirip view sebelumnya, ubah posisi/kemiringan"
                else:
                    img_pts.append(corners)
                    cover.add(corners)
                    last_t = frame.video_t_s
                    if save_dir is not None:
                        save_dir.mkdir(parents=True, exist_ok=True)
                        cv2.imwrite(str(save_dir / f"view_{len(img_pts):03d}.png"), img)
                    status = f"view {len(img_pts)} diambil"
                    print(f"  {status} (t={frame.video_t_s:.2f}s, cakupan {cover.percent:.0f}%)")
            if show:
                vis = img.copy()
                cover.draw(vis)
                if corners is not None:
                    cv2.drawChessboardCorners(vis, (cols, rows), corners.astype(np.float32).reshape(-1, 1, 2), True)
                missing = 9 - sum(cover.cells_3x3())
                lines = [f"views: {len(img_pts)}   cakupan: {cover.percent:.0f}%   sel kosong (kotak merah): {missing}",
                         f"auto: {'ON' if auto else 'off'}   {status}"]
                for i, t in enumerate(lines):
                    y = 30 + 28 * i
                    cv2.putText(vis, t, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 0), 4, cv2.LINE_AA)
                    cv2.putText(vis, t, (10, y), cv2.FONT_HERSHEY_SIMPLEX, 0.7,
                                (0, 255, 0) if corners is not None else (0, 0, 255), 2, cv2.LINE_AA)
                cv2.imshow(win, vis)
    if show:
        cv2.destroyAllWindows()
    return img_pts, size


# --------------------------------------------------------------------------------------------- main

def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--source", help="webcam | webcam:N | path video | URL")
    g.add_argument("--images", nargs="+", help="file/glob gambar papan catur")
    g.add_argument("--make-board", default=None, help="tulis PNG papan untuk dicetak (A4), lalu keluar")
    ap.add_argument("--pattern", default="9x6", help="sudut dalam COLSxROWS (default 9x6)")
    ap.add_argument("--square-mm", type=float, default=25.0, help="sisi kotak papan HASIL CETAK (mm)")
    ap.add_argument("--fisheye", action="store_true", help="pakai cv2.fisheye.calibrate (lensa sangat lebar)")
    ap.add_argument("--min-views", type=int, default=12)
    ap.add_argument("--warn-rms", type=float, default=1.0, help="peringatkan jika RMS reprojeksi > ini (px)")
    ap.add_argument("--auto-every-s", type=float, default=0.0, help="live/video: ambil view otomatis tiap N s")
    ap.add_argument("--no-show", action="store_true")
    ap.add_argument("--save-views", default=None, help="simpan view yang diambil ke folder ini")
    ap.add_argument("--runtime", default=str(CONFIG_DIR / "runtime.yaml"),
                    help="webcam: setelan front.source (resolusi, MJPG, fokus) diambil dari sini")
    ap.add_argument("--no-runtime-source", action="store_true", help="webcam: abaikan setelan front.source")
    ap.add_argument("--detect-max-side", type=int, default=960, help="live: cari papan di gambar diperkecil")
    ap.add_argument("--max-motion-px", type=float, default=3.0, help="live: tolak view jika papan bergerak > ini")
    ap.add_argument("--min-diff-px", type=float, default=25.0, help="live: tolak view yang terlalu mirip")
    ap.add_argument("--reject-factor", type=float, default=2.5, help="buang view dengan error > faktor x median")
    ap.add_argument("--reject-floor-px", type=float, default=0.5, help="...tetapi hanya jika error > ini (px)")
    ap.add_argument("--out", default=str(CONFIG_DIR / "front_camera.yaml"))
    args = ap.parse_args(argv)

    cols, rows = (int(x) for x in args.pattern.lower().split("x"))

    if args.make_board:
        p = make_board(resolve_path(args.make_board), cols, rows, args.square_mm)
        print(f"Papan ditulis ke {p}. Cetak A4 landscape 100% (actual size, tanpa 'fit to page'), "
              f"tempel di papan rata dan kaku, lalu ukur garis 50 mm dan sisi kotak. Pakai ukuran kotak hasil "
              f"ukur di --square-mm.")
        return 0

    objp = board_points(cols, rows, args.square_mm / 1000.0)
    capture = None
    if args.images:
        img_pts, size = collect_from_images(args.images, cols, rows)
    else:
        runtime = None if args.no_runtime_source else load_yaml(resolve_path(args.runtime))
        src_cfg = source_config(args.source, runtime)
        if src_cfg.get("type") == "webcam":
            capture = src_cfg
            if src_cfg.get("autofocus") is not False:
                print("PERINGATAN: autofocus tidak dimatikan; K bisa bergeser saat fokus berubah. "
                      "Set front.source.autofocus: false di runtime.yaml.")
        img_pts, size = collect_live(src_cfg, cols, rows, not args.no_show, args.auto_every_s,
                                     resolve_path(args.save_views) if args.save_views else None,
                                     args.detect_max_side, args.max_motion_px, args.min_diff_px)

    if size is None or len(img_pts) < args.min_views:
        print(f"Gagal: hanya {len(img_pts)} view valid (min {args.min_views}).")
        return 1
    rms, K, dist, used, bad, errs = calibrate_robust(objp, img_pts, size, args.fisheye, args.reject_factor,
                                                     args.reject_floor_px, args.min_views)
    if bad:
        print("View dibuang (error tinggi): " + ", ".join(f"#{i + 1} ({errs[i]:.2f} px)" for i in bad))
    cover = Coverage(size)
    cover.rebuild(used)
    out = resolve_path(args.out)
    write_camera_block(out, K, dist, size, args.fisheye, rms, len(used), capture)
    hfov = np.degrees(2 * np.arctan(size[0] / (2 * K[0, 0])))
    vfov = np.degrees(2 * np.arctan(size[1] / (2 * K[1, 1])))
    print(f"RMS reprojeksi = {rms:.3f} px dari {len(used)} view, ukuran {size[0]}x{size[1]}, "
          f"cakupan frame {cover.percent:.0f}%")
    print(f"K =\n{np.array2string(K, precision=2)}\ndist = {np.array2string(dist, precision=5)}")
    print(f"FOV dari K: horizontal {hfov:.1f} deg, vertikal {vfov:.1f} deg")
    print(f"Ditulis ke {out}. Jangan lupa ukur mount.h_m dan mount.theta_deg (docs/FRONT_CAMERA.md).")
    if rms > args.warn_rms:
        print(f"PERINGATAN: RMS > {args.warn_rms} px; ulangi dengan view lebih banyak/tajam yang menutupi seluruh frame.")
    if cover.percent < 60 or not all(cover.cells_3x3()):
        print("PERINGATAN: papan belum menutupi seluruh frame (terutama tepi/sudut); distorsi bisa kurang akurat.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
