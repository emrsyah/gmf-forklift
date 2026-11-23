"""A9: P(mata tertutup) dari potongan mata, untuk frame tanpa landmark (cadangan pose YuNet, wajah kecil).

Saat landmark MediaPipe gagal, D-01/D-02 tidak punya sinyal mata (eye_valid False). Classifier kecil
(training/train_eye_cls.py, MRL Eye inframerah) membaca dua potongan mata di sekitar titik mata detektor.
Sisi potongan = side_d x jarak antar mata. Angka di configs/thresholds.yaml signals.eye_classifier.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from .ppe_cls import cut, inside_frac_img, to_input


class EyeClassifier:
    def __init__(self, cfg: dict | None):
        c = cfg or {}
        self.enabled = bool(c.get("enabled", False))
        self.size = int(c.get("input_px", 32))
        self.side_d = float(c.get("side_d", 0.7))
        self.min_d_px = float(c.get("min_eye_dist_px", 12))
        self.min_inside = float(c.get("min_inside", 0.8))
        self.sess = None
        if self.enabled:
            import onnxruntime as ort

            from src.config import resolve_path
            p = resolve_path(c["weights"])
            if p is None or not Path(p).exists():
                raise FileNotFoundError(f"Bobot classifier mata tidak ditemukan: {p}")
            self.sess = ort.InferenceSession(str(p), providers=c.get("providers") or ["CPUExecutionProvider"])
            self.input_name = self.sess.get_inputs()[0].name

    def boxes(self, eye_pts) -> list[tuple[float, float, float, float]]:
        e = np.asarray(eye_pts, np.float32).reshape(-1, 2)[:2]
        d = float(np.hypot(*(e[0] - e[1])))
        if d < self.min_d_px:
            return []
        h = self.side_d * d / 2
        return [(float(x - h), float(y - h), float(x + h), float(y + h)) for x, y in e]

    def p_closed(self, image: np.ndarray, eye_pts) -> float | None:
        """Rata-rata P(tertutup) mata yang potongannya cukup di dalam gambar; None jika tidak bisa dinilai."""
        if self.sess is None or eye_pts is None:
            return None
        wh = (image.shape[1], image.shape[0])
        crops = [cut(image, b, self.size) for b in self.boxes(eye_pts) if inside_frac_img(b, wh) >= self.min_inside]
        if not crops:
            return None
        logits = self.sess.run(None, {self.input_name: to_input(crops)})[0]
        e = np.exp(logits - logits.max(1, keepdims=True))
        return float((e[:, 1] / e.sum(1)).mean())
