"""A8 (D-10): klasifikasi helm / rompi pada potongan kepala dan dada pengemudi (dari kotak wajah).

YOLO kabin melihat kepala dan badan pengemudi di frame penuh (diperkecil ke 640) dan harus menemukan kotaknya
dulu. Di sini potongan diambil langsung dari kotak wajah Face Landmarker (selalu pengemudi), lalu classifier
kecil (training/train_ppe_cls.py, data publik ber-anotasi upstream) menjawab "tanpa helm?" / "tanpa rompi?".
Geometri potongan (head_box / torso_box) dipakai SAMA oleh training dan runtime.
Semua angka dari configs/thresholds.yaml D10_ppe.classifier.
"""
from __future__ import annotations

from pathlib import Path

import cv2
import numpy as np

# Geometri bawaan (satuan u = sisi terpanjang kotak wajah landmark). Disalin ke metadata model saat training;
# runtime membaca dari config (harus sama dengan saat training).
HEAD_GEOM = {"dy_u": -0.30, "side_u": 1.8}            # pusat sedikit di atas pusat wajah (helm)
TORSO_GEOM = {"top_u": 0.15, "side_u": 2.4}           # dada: mulai sedikit di bawah dagu


def head_box(face_box, geom=HEAD_GEOM):
    x1, y1, x2, y2 = face_box[:4]
    u = max(x2 - x1, y2 - y1)
    cx, cy = (x1 + x2) / 2, (y1 + y2) / 2 + geom["dy_u"] * u
    h = geom["side_u"] * u / 2
    return cx - h, cy - h, cx + h, cy + h


def torso_box(face_box, geom=TORSO_GEOM):
    x1, y1, x2, y2 = face_box[:4]
    u = max(x2 - x1, y2 - y1)
    cx, top = (x1 + x2) / 2, y2 + geom["top_u"] * u
    s = geom["side_u"] * u
    return cx - s / 2, top, cx + s / 2, top + s


def inside_frac_img(box, wh) -> float:
    x1, y1, x2, y2 = box
    a = max(0.0, (x2 - x1) * (y2 - y1))
    iw = max(0.0, min(x2, wh[0]) - max(x1, 0))
    ih = max(0.0, min(y2, wh[1]) - max(y1, 0))
    return iw * ih / a if a > 0 else 0.0


def cut(img: np.ndarray, box, size: int) -> np.ndarray:
    """Potongan persegi (bagian di luar gambar = abu-abu 114, seperti letterbox) -> size x size, 1 channel."""
    g = img if img.ndim == 2 else cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    x1, y1, x2, y2 = (int(round(v)) for v in box)
    H, W = g.shape[:2]
    out = np.full((max(1, y2 - y1), max(1, x2 - x1)), 114, np.uint8)
    sx1, sy1, sx2, sy2 = max(0, x1), max(0, y1), min(W, x2), min(H, y2)
    if sx2 > sx1 and sy2 > sy1:
        out[sy1 - y1:sy2 - y1, sx1 - x1:sx2 - x1] = g[sy1:sy2, sx1:sx2]
    return cv2.resize(out, (size, size), interpolation=cv2.INTER_AREA if out.shape[0] > size else cv2.INTER_LINEAR)


def to_input(crops: list[np.ndarray]) -> np.ndarray:
    return (np.stack(crops)[:, None].astype(np.float32) / 255.0)


class PpeClassifier:
    """Dua model biner ONNX (kepala: P(tanpa helm), dada: P(tanpa rompi))."""

    def __init__(self, cfg: dict | None):
        c = cfg or {}
        self.enabled = bool(c.get("enabled", False))
        self.size = int(c.get("input_px", 96))
        self.head_geom = dict(HEAD_GEOM, **(c.get("head_geom") or {}))
        self.torso_geom = dict(TORSO_GEOM, **(c.get("torso_geom") or {}))
        self.no_helmet_thr = float(c.get("no_helmet_thr", 0.5))
        self.no_vest_thr = float(c.get("no_vest_thr", 0.5))
        self.min_face_u_px = float(c.get("min_face_u_px", 40))
        self.torso_min_inside = float(c.get("torso_min_inside", 0.6))
        self.mode = c.get("mode", "replace")      # replace | and | or (terhadap aturan YOLO)
        self.head_sess = self.torso_sess = None
        self.last: dict = {}
        if self.enabled:
            from src.config import resolve_path
            import onnxruntime as ort

            prov = c.get("providers") or ["CPUExecutionProvider"]
            for key in ("head_weights", "torso_weights"):
                p = resolve_path(c[key])
                if p is None or not Path(p).exists():
                    raise FileNotFoundError(f"Bobot classifier APD tidak ditemukan: {p}")
            self.head_sess = ort.InferenceSession(str(resolve_path(c["head_weights"])), providers=prov)
            self.torso_sess = ort.InferenceSession(str(resolve_path(c["torso_weights"])), providers=prov)

    @staticmethod
    def _p1(sess, x: np.ndarray) -> float:
        logits = sess.run(None, {sess.get_inputs()[0].name: x})[0][0]
        e = np.exp(logits - logits.max())
        return float(e[1] / e.sum())

    def classify(self, image: np.ndarray, face_box) -> dict:
        """{'p_no_helmet': float|None, 'p_no_vest': float|None}; None = tidak bisa dinilai."""
        out = {"p_no_helmet": None, "p_no_vest": None}
        if face_box is None or self.head_sess is None:
            self.last = out
            return out
        u = max(face_box[2] - face_box[0], face_box[3] - face_box[1])
        if u < self.min_face_u_px:
            self.last = out
            return out
        wh = (image.shape[1], image.shape[0])
        hb = head_box(face_box, self.head_geom)
        out["p_no_helmet"] = self._p1(self.head_sess, to_input([cut(image, hb, self.size)]))
        tb = torso_box(face_box, self.torso_geom)
        if inside_frac_img(tb, wh) >= self.torso_min_inside:     # dada tidak terlihat -> rompi tidak dinilai
            out["p_no_vest"] = self._p1(self.torso_sess, to_input([cut(image, tb, self.size)]))
        out["head_box"], out["torso_box"] = hb, tb
        self.last = out
        return out

    def combine(self, yolo_no_helmet: bool, yolo_no_vest: bool, res: dict) -> tuple[bool, bool]:
        def one(yolo: bool, p, thr) -> bool:
            if p is None:
                return yolo if self.mode != "replace" else False
            c = p >= thr
            return c if self.mode == "replace" else (c and yolo) if self.mode == "and" else (c or yolo)
        return (one(yolo_no_helmet, res.get("p_no_helmet"), self.no_helmet_thr),
                one(yolo_no_vest, res.get("p_no_vest"), self.no_vest_thr))
