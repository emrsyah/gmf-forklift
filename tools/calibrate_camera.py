"""Kalibrasi intrinsik kamera depan dengan papan catur (Tech Spec bag. 6, langkah 1).

Hasil (K, dist, image_size, fisheye, rms) ditulis ke blok `camera:` di configs/front_camera.yaml;
blok lain (mount, distance, motion, classes, zones) DIPERTAHANKAN.

Contoh:
    # live: SPASI/c = ambil view saat papan terdeteksi, q/ESC = selesai & kalibrasi
    python tools/calibrate_camera.py --source webcam:1 --pattern 9x6 --square-mm 25
    # folder gambar
    python tools/calibrate_camera.py --images data/calib/front/*.jpg --pattern 9x6 --square-mm 25
    # lensa sangat lebar
    python tools/calibrate_camera.py --images "data/calib/front/*.jpg" --fisheye
    # video rekaman, ambil otomatis tiap 1 s tanpa jendela
    python tools/calibrate_camera.py --source calib.mp4 --auto-every-s 1.0 --no-show

--pattern = jumlah sudut DALAM (kolom x baris), bukan jumlah kotak.
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

from src.config import CONFIG_DIR, resolve_path  # noqa: E402

_SUBPIX_WIN = (11, 11)
_SUBPIX_CRIT = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 1e-3)
_HEADER = ("# Kalibrasi + parameter kamera depan (milestone L9). Lihat docs/FRONT_CAMERA.md.\n"
           "# Blok `camera` ditulis oleh tools/calibrate_camera.py; blok lain diisi/diukur manual.\n")


def board_points(cols: int, rows: int, square_m: float) -> np.ndarray:
    objp = np.zeros((rows * cols, 3), np.float64)
    objp[:, :2] = np.mgrid[0:cols, 0:rows].T.reshape(-1, 2) * square_m
    return objp


def find_corners(gray: np.ndarray, cols: int, rows: int) -> np.ndarray | None:
    flags = cv2.CALIB_CB_ADAPTIVE_THRESH | cv2.CALIB_CB_NORMALIZE_IMAGE | cv2.CALIB_CB_FAST_CHECK
    ok, corners = cv2.findChessboardCorners(gray, (cols, rows), flags)
    if not ok:
        return None
    corners = cv2.cornerSubPix(gray, corners, _SUBPIX_WIN, (-1, -1), _SUBPIX_CRIT)
    return corners.reshape(-1, 2).astype(np.float64)


def calibrate(obj_list: list[np.ndarray], img_list: list[np.ndarray], image_size: tuple[int, int],
              fisheye: bool = False) -> tuple[float, np.ndarray, np.ndarray]:
    """Mengembalikan (rms_px, K 3x3, dist 1-D). image_size = (w, h)."""
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
        rms, K, D, _, _ = cv2.fisheye.calibrate(objp, imgp, image_size, K, D, flags=flags, criteria=crit)
        return float(rms), K, D.ravel()
    objp = [o.reshape(-1, 3).astype(np.float32) for o in obj_list]
    imgp = [i.reshape(-1, 1, 2).astype(np.float32) for i in img_list]
    rms, K, D, _, _ = cv2.calibrateCamera(objp, imgp, image_size, None, None)
    return float(rms), K, D.ravel()


def write_camera_block(out_path: Path, K: np.ndarray, dist: np.ndarray, image_size: tuple[int, int],
                       fisheye: bool, rms: float, n_views: int) -> None:
    data = {}
    if out_path.exists():
        with open(out_path, "r", encoding="utf-8") as f:
            data = yaml.safe_load(f) or {}
    data["camera"] = {
        "calibrated": True,
        "image_size": [int(image_size[0]), int(image_size[1])],
        "fisheye": bool(fisheye),
        "K": [[round(float(v), 4) for v in row] for row in np.asarray(K)],
        "dist": [round(float(v), 8) for v in np.asarray(dist).ravel()],
        "rms_px": round(float(rms), 4),
        "n_views": int(n_views),
        "calibrated_at": _dt.datetime.now().isoformat(timespec="seconds"),
    }
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w", encoding="utf-8") as f:
        f.write(_HEADER)
        yaml.safe_dump(data, f, sort_keys=False, default_flow_style=None, allow_unicode=True)


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


def collect_live(source_arg: str, cols: int, rows: int, show: bool, auto_every_s: float,
                 save_dir: Path | None):
    from src.capture import make_source, parse_source_arg

    src_cfg = parse_source_arg(source_arg)
    src_cfg.setdefault("realtime", False)
    img_pts, size, last_t = [], None, -1e9
    win = "calibrate_camera (SPASI/c = ambil, q = selesai)"
    with make_source(src_cfg) as source:
        for frame in source:
            img = frame.image
            size = (img.shape[1], img.shape[0])
            gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
            corners = find_corners(gray, cols, rows)
            take = False
            if corners is not None and auto_every_s > 0 and frame.video_t_s - last_t >= auto_every_s:
                take = True
            key = -1
            if show:
                vis = img.copy()
                if corners is not None:
                    cv2.drawChessboardCorners(vis, (cols, rows), corners.astype(np.float32).reshape(-1, 1, 2), True)
                cv2.putText(vis, f"views: {len(img_pts)}", (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                            (0, 255, 0) if corners is not None else (0, 0, 255), 2)
                cv2.imshow(win, vis)
                key = cv2.waitKey(1) & 0xFF
                if key in (ord("q"), 27):
                    break
                if key in (ord(" "), ord("c")) and corners is not None:
                    take = True
            if take:
                img_pts.append(corners)
                last_t = frame.video_t_s
                if save_dir is not None:
                    save_dir.mkdir(parents=True, exist_ok=True)
                    cv2.imwrite(str(save_dir / f"view_{len(img_pts):03d}.png"), img)
                print(f"  view {len(img_pts)} diambil (t={frame.video_t_s:.2f}s)")
    if show:
        cv2.destroyAllWindows()
    return img_pts, size


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    g = ap.add_mutually_exclusive_group(required=True)
    g.add_argument("--source", help="webcam | webcam:N | path video | URL")
    g.add_argument("--images", nargs="+", help="file/glob gambar papan catur")
    ap.add_argument("--pattern", default="9x6", help="sudut dalam COLSxROWS (default 9x6)")
    ap.add_argument("--square-mm", type=float, default=25.0, help="sisi kotak papan (mm)")
    ap.add_argument("--fisheye", action="store_true", help="pakai cv2.fisheye.calibrate (lensa sangat lebar)")
    ap.add_argument("--min-views", type=int, default=10)
    ap.add_argument("--warn-rms", type=float, default=1.0, help="peringatkan jika RMS reprojeksi > ini (px)")
    ap.add_argument("--auto-every-s", type=float, default=0.0, help="live/video: ambil view otomatis tiap N s")
    ap.add_argument("--no-show", action="store_true")
    ap.add_argument("--save-views", default=None, help="simpan view yang diambil ke folder ini")
    ap.add_argument("--out", default=str(CONFIG_DIR / "front_camera.yaml"))
    args = ap.parse_args(argv)

    cols, rows = (int(x) for x in args.pattern.lower().split("x"))
    objp = board_points(cols, rows, args.square_mm / 1000.0)

    if args.images:
        img_pts, size = collect_from_images(args.images, cols, rows)
    else:
        img_pts, size = collect_live(args.source, cols, rows, not args.no_show, args.auto_every_s,
                                     resolve_path(args.save_views) if args.save_views else None)

    if size is None or len(img_pts) < args.min_views:
        print(f"Gagal: hanya {len(img_pts)} view valid (min {args.min_views}).")
        return 1
    rms, K, dist = calibrate([objp] * len(img_pts), img_pts, size, args.fisheye)
    out = resolve_path(args.out)
    write_camera_block(out, K, dist, size, args.fisheye, rms, len(img_pts))
    print(f"RMS reprojeksi = {rms:.3f} px dari {len(img_pts)} view, ukuran {size[0]}x{size[1]}")
    print(f"K =\n{np.array2string(K, precision=2)}\ndist = {np.array2string(dist, precision=5)}")
    print(f"Ditulis ke {out}. Jangan lupa ukur mount.h_m dan mount.theta_deg (docs/FRONT_CAMERA.md).")
    if rms > args.warn_rms:
        print(f"PERINGATAN: RMS > {args.warn_rms} px; ulangi dengan view lebih banyak/tajam yang menutupi seluruh frame.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
