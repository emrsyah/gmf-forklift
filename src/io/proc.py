"""Proses pembantu (ffmpeg) berprioritas rendah supaya loop deteksi tidak tersendat saat encode.

Di Linux (Jetson) proses anak dijalankan dengan `nice` lebih tinggi: encoder tetap dapat CPU yang
menganggur, tetapi kalah prioritas dari MediaPipe/YOLO (K5: waktu deteksi). Di Windows diabaikan.
"""
from __future__ import annotations

import os
import subprocess
import sys


def popen_nice(cmd: list[str], nice: int = 0, **kw) -> subprocess.Popen:
    if nice and sys.platform.startswith("linux"):
        n = int(nice)
        kw["preexec_fn"] = lambda: os.nice(n)
    return subprocess.Popen(cmd, **kw)
