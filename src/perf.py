"""Batas thread pustaka numerik (OpenBLAS numpy/scipy, OpenMP/libgomp, OpenCV) dari config.

Di Jetson (6 core) tiap pustaka membuat ~5 thread pekerja yang ikut berputar (spin) saat menunggu kerja.
Untuk gambar kecil per frame, paralelisme itu jarang membantu tetapi memakan CPU yang dibutuhkan
MediaPipe/YOLO/encoder. `runtime.threads` membatasinya; null = biarkan bawaan pustaka.

    threads:
      blas: 1        # OpenBLAS (numpy, scipy)
      openmp: 1      # libgomp (OpenMP)
      opencv: 2      # cv2.setNumThreads

Diterapkan saat runtime lewat threadpoolctl (opsional; kalau tidak terpasang, batas BLAS/OpenMP dilewati
dengan peringatan). Ukur dampaknya di perangkat (CPU proses + FPS) sebelum mengubah config Jetson.
"""
from __future__ import annotations

_KEEP: list = []        # objek threadpoolctl harus tetap hidup agar batasnya berlaku


def apply_thread_limits(cfg: dict | None) -> dict:
    """Terapkan batas; kembalikan ringkasan yang benar-benar diterapkan (untuk log)."""
    cfg = cfg or {}
    applied: dict = {}
    n_cv = cfg.get("opencv")
    if n_cv is not None:
        import cv2
        cv2.setNumThreads(int(n_cv))
        applied["opencv"] = cv2.getNumThreads()
    wanted = {api: cfg.get(key) for key, api in (("blas", "blas"), ("openmp", "openmp"))
              if cfg.get(key) is not None}
    if wanted:
        try:
            from threadpoolctl import threadpool_limits
        except ImportError:
            print(f"[threads] threadpoolctl tidak terpasang; batas {sorted(wanted)} dilewati", flush=True)
            return applied
        for api, n in wanted.items():
            _KEEP.append(threadpool_limits(limits=int(n), user_api=api))
            applied[api] = int(n)
    return applied
