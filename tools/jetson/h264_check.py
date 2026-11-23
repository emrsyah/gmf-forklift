"""Cek H.264 kamera depan (C930e) di perangkat: sumber, perekam, file hasil. Satu perintah, hasil ringkas.

    cd ~/gmf-forklift && . .venv/bin/activate
    python tools/jetson/h264_check.py --index 1 --seconds 20

Kamera tidak boleh sedang dipakai proses lain (hentikan detektor dulu). Rekaman uji ditulis ke --out
dan DIHAPUS di akhir (kecuali --keep), karena berisi gambar ruangan/orang.

Yang diperiksa:
- mode H.264 aktif (unit XU ditemukan, kamera menerima mux) dan berapa frame membawa H.264;
- jarak keyframe IDR sebenarnya (frame dan detik) -> menentukan seberapa cepat segmen/stream bisa mulai;
- CPU proses ini (baca kamera + parsing + perekam) dengan / tanpa dekode JPEG;
- tiap segmen MP4: jumlah frame terbaca OpenCV == baris .ts.csv (syarat replay), orientasi, durasi, ukuran;
- laju data (Mbps) dan perkiraan GB/jam.
"""
from __future__ import annotations

import argparse
import json
import os
import shutil
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import cv2  # noqa: E402

from src.capture import make_source  # noqa: E402
from src.io.h264_recorder import H264Recorder  # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--index", type=int, default=1)
    ap.add_argument("--width", type=int, default=1280)
    ap.add_argument("--height", type=int, default=720)
    ap.add_argument("--fps", type=int, default=30)
    ap.add_argument("--bitrate", type=int, default=3_000_000)
    ap.add_argument("--seconds", type=float, default=20)
    ap.add_argument("--segment-s", type=float, default=6)
    ap.add_argument("--rotate", type=int, default=270)
    ap.add_argument("--decode", action="store_true", help="dekode JPEG tiap frame (seperti pipeline deteksi)")
    ap.add_argument("--out", default="/tmp/h264check")
    ap.add_argument("--keep", action="store_true", help="jangan hapus rekaman uji")
    a = ap.parse_args()

    shutil.rmtree(a.out, ignore_errors=True)
    src = make_source({"type": "webcam", "index": a.index, "api": "v4l2", "fourcc": "MJPG", "width": a.width,
                       "height": a.height, "fps": a.fps,
                       "h264_mux": {"enabled": True, "bitrate": a.bitrate, "iframe_period_ms": 1000,
                                    "profile": "constrained_baseline", "decode": a.decode}})
    print(f"sumber {type(src).__name__} | unit XU {getattr(src.ctl, 'unit', None)} | "
          f"H.264 {'AKTIF' if src.ctl else 'TIDAK: ' + str(src.h264_error)}")
    if src.negotiated:
        n = src.negotiated
        print(f"negosiasi: {n['wWidth']}x{n['wHeight']} {n['dwBitRate'] / 1e6:.1f} Mbps "
              f"interval {n['dwFrameInterval'] / 1e4:.1f} ms IDR {n['wIFramePeriod']} ms profil 0x{n['wProfile']:04x}")
    rec = H264Recorder({"dir": a.out, "segment_s": a.segment_s, "min_free_gb": 1, "prefix": "check"}, "dev",
                       rotate=a.rotate, on_segment=src.request_idr)
    frames = with_h264 = 0
    idr = []
    c0, t0 = time.process_time(), time.time()
    try:
        while time.time() - t0 < a.seconds:
            f = src.read()
            if f is None:
                print("kamera berhenti mengirim frame")
                break
            frames += 1
            au = f.meta.get("h264")
            if au:
                with_h264 += 1
                if f.meta.get("h264_idr"):
                    idr.append((frames, time.time() - t0))
                rec.push(au, f.ts_ms, f.meta.get("h264_idr"))
    finally:
        cpu, wall = time.process_time() - c0, time.time() - t0
        src.release()
        rec.close()
    print(f"frame {frames} dalam {wall:.1f} s = {frames / wall:.1f} fps | ber-H.264 {with_h264} | "
          f"dibuang perekam {rec.dropped}")
    print(f"CPU proses ini {100 * cpu / wall:.0f}% satu core ({'dengan' if a.decode else 'tanpa'} dekode JPEG)")
    if len(idr) > 1:
        gf = [b[0] - x[0] for x, b in zip(idr, idr[1:])]
        gs = [b[1] - x[1] for x, b in zip(idr, idr[1:])]
        print(f"jarak IDR: {min(gf)}-{max(gf)} frame, {min(gs):.2f}-{max(gs):.2f} s (diminta 1 s)")
    ok_all = True
    for seg in rec.segments:
        rows = sum(1 for _ in open(str(seg) + ".ts.csv", encoding="utf-8")) - 1
        cap = cv2.VideoCapture(str(seg))
        k, shape = 0, None
        while True:
            ok, img = cap.read()
            if not ok:
                break
            k, shape = k + 1, img.shape
        cap.release()
        pr = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration,size", "-of", "json",
                             str(seg)], capture_output=True, text=True).stdout
        fmt = json.loads(pr or "{}").get("format", {})
        same = k == rows
        ok_all &= same
        print(f"  {seg.name}: baris {rows}, frame OpenCV {k} {'OK' if same else 'BEDA'}, ukuran gambar "
              f"{shape[1] if shape else '?'}x{shape[0] if shape else '?'}, durasi {float(fmt.get('duration', 0)):.1f} s, "
              f"{int(fmt.get('size', 0)) / 1e6:.2f} MB")
    total = sum(os.path.getsize(s) for s in rec.segments if os.path.exists(s))
    if wall > 0:
        print(f"total {total / 1e6:.1f} MB = {8 * total / wall / 1e6:.2f} Mbps -> {total / wall * 3600 / 1e9:.2f} GB/jam")
    if not a.keep:
        shutil.rmtree(a.out, ignore_errors=True)
        print(f"rekaman uji dihapus ({a.out})")
    return 0 if (src.h264_error is None and rec.segments and ok_all) else 1


if __name__ == "__main__":
    raise SystemExit(main())
