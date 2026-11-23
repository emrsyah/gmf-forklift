"""Proses pembantu (ffmpeg) berprioritas rendah supaya loop deteksi tidak tersendat saat encode.

Di Linux (Jetson) proses anak dijalankan dengan `nice` lebih tinggi: encoder tetap dapat CPU yang
menganggur, tetapi kalah prioritas dari MediaPipe/YOLO (K5: waktu deteksi). Di Windows diabaikan.
"""
from __future__ import annotations

import shutil
import subprocess
import sys


def popen_nice(cmd: list[str], nice: int = 0, **kw) -> subprocess.Popen:
    # `nice -n N cmd`, bukan preexec_fn: preexec_fn memaksa fork() penuh proses multi-thread (CUDA, MediaPipe)
    # -> lambat dan tidak aman thread; tanpa preexec_fn Python memakai vfork/posix_spawn.
    if nice and sys.platform.startswith("linux") and shutil.which("nice"):
        cmd = ["nice", "-n", str(int(nice)), *cmd]
    return subprocess.Popen(cmd, **kw)
