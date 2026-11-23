"""Detektor wajah + crop sebelum Face Landmarker (interface sama dengan MediaPipeFace).

Masalah (uji forklift 2026-11-23): wajah kecil, di tepi bawah gambar, gelap karena langit di belakang.
Detektor bawaan Face Landmarker adalah BlazeFace "short range" (jarak selfie) -> "wajah terlihat 0%".

Alur per frame:
  1. kotak wajah = kotak landmark frame sebelumnya (pelacakan), atau detektor (YuNet) jika belum/hilang;
  2. crop persegi `margin` x kotak itu (diberi bingkai hitam jika keluar gambar, mis. dagu terpotong),
     diperbesar ke `out_px` -> wajah mengisi gambar seperti selfie;
  3. Face Landmarker di crop; landmark dipetakan kembali ke koordinat frame.
Crop ditahan selama wajah masih di dalamnya (`hold_*`) supaya pelacakan VIDEO MediaPipe stabil.

Catatan pose: matriks transformasi MediaPipe dihitung relatif ke crop (titik pusat optik = pusat crop),
jadi yaw/pitch absolut sedikit bergeser dibanding full frame. Indikator memakai pose relatif terhadap
baseline kalibrasi (yang juga diukur lewat crop), jadi selama crop tidak berpindah jauh efeknya kecil.
"""
from __future__ import annotations

import cv2
import numpy as np

from src.config import resolve_path

from .face import FaceObs


class YuNetDetector:
    """OpenCV FaceDetectorYN (YuNet, MIT, opencv_zoo). Mengembalikan kotak (x1, y1, x2, y2, skor)."""

    def __init__(self, cfg: dict):
        path = resolve_path(cfg.get("yunet_model", "models/face/face_detection_yunet_2023mar.onnx"))
        if path is None or not path.exists():
            raise FileNotFoundError(f"Model YuNet tidak ditemukan: {path}")
        self.score = float(cfg.get("det_score", 0.6))
        self.max_side = int(cfg.get("det_max_side", 640))
        self._det = cv2.FaceDetectorYN.create(str(path), "", (320, 320), self.score, 0.3, 50)
        self._size = None

    def detect(self, img: np.ndarray) -> list[tuple[float, float, float, float, float]]:
        h, w = img.shape[:2]
        s = min(1.0, self.max_side / max(h, w))
        small = img if s == 1.0 else cv2.resize(img, (int(w * s), int(h * s)), interpolation=cv2.INTER_AREA)
        if small.ndim == 2:
            small = cv2.cvtColor(small, cv2.COLOR_GRAY2BGR)
        size = (small.shape[1], small.shape[0])
        if size != self._size:
            self._det.setInputSize(size)
            self._size = size
        _, faces = self._det.detect(small)
        out = []
        for f in faces if faces is not None else []:
            x, y, bw, bh, score = float(f[0]), float(f[1]), float(f[2]), float(f[3]), float(f[-1])
            out.append((x / s, y / s, (x + bw) / s, (y + bh) / s, score))
        return out


def make_face_detector(cfg: dict):
    kind = cfg.get("detector", "yunet")
    if kind == "yunet":
        return YuNetDetector(cfg)
    raise ValueError(f"Detektor wajah tidak dikenal: {kind}")


def square_crop_box(box, margin: float) -> tuple[float, float, float]:
    """(cx, cy, sisi) persegi di sekitar kotak wajah."""
    x1, y1, x2, y2 = box[:4]
    side = margin * max(x2 - x1, y2 - y1)
    return (x1 + x2) / 2, (y1 + y2) / 2, side


def crop_square(img: np.ndarray, cx: float, cy: float, side: float, out_px: int) -> tuple[np.ndarray, float, float]:
    """Crop persegi (bingkai hitam di luar gambar), diubah ke out_px x out_px. -> (crop, x0, y0)."""
    side_i = max(2, int(round(side)))
    x0, y0 = int(round(cx - side_i / 2)), int(round(cy - side_i / 2))
    h, w = img.shape[:2]
    canvas = np.zeros((side_i, side_i) + img.shape[2:], dtype=img.dtype)
    sx1, sy1, sx2, sy2 = max(0, x0), max(0, y0), min(w, x0 + side_i), min(h, y0 + side_i)
    if sx2 > sx1 and sy2 > sy1:
        canvas[sy1 - y0:sy2 - y0, sx1 - x0:sx2 - x0] = img[sy1:sy2, sx1:sx2]
    interp = cv2.INTER_AREA if side_i > out_px else cv2.INTER_LINEAR
    return cv2.resize(canvas, (out_px, out_px), interpolation=interp), float(x0), float(y0)


class CropFace:
    """Bungkus model wajah (MediaPipeFace) dengan detektor + crop. `process()` sama dengan model dalam."""

    def __init__(self, inner, detector, cfg: dict, inner_full=None):
        self.inner = inner                  # Face Landmarker untuk crop
        self.inner_full = inner_full        # opsional: landmarker terpisah untuk full frame (cadangan)
        self.det = detector
        self.margin = float(cfg.get("margin", 1.8))
        self.out_px = int(cfg.get("out_px", 320))
        self.min_face_px = float(cfg.get("min_face_px", 20))
        self.hold_shift = float(cfg.get("hold_shift", 0.12))   # geser pusat < 12% sisi -> crop ditahan
        self.hold_scale = float(cfg.get("hold_scale", 0.15))   # ukuran berubah < 15% -> crop ditahan
        self.redetect_every = int(cfg.get("redetect_every", 0))  # 0 = detektor hanya saat wajah hilang
        # wajah hilang (menoleh jauh, kamera ditutup, kursi kosong): pencarian (YuNet + full frame) hanya
        # tiap N frame, frame lain langsung "tidak ada wajah". 1 = tiap frame (perilaku lama).
        self.lost_search_every = max(1, int(cfg.get("lost_search_every", 1)))
        # ... tapi hanya setelah wajah hilang >= sekian detik: hilang 1-2 frame (mis. menunduk sambil pejam,
        # skenario 7) tetap dicari tiap frame supaya celah tidak melewati toleransi D-02 (face_gap_tolerance_s).
        self.lost_search_after_ms = 1000 * float(cfg.get("lost_search_after_s", 0.5))
        self._lost_since_ms: int | None = None
        self._crop: tuple[float, float, float] | None = None    # (cx, cy, sisi) crop aktif
        self._track_box = None              # kotak landmark terakhir (koordinat gambar input)
        self._n = 0
        self._lost_n = 0                    # frame berturut-turut tanpa wajah
        self.last_source = "none"           # track | detector | full | none (diagnosis/overlay)
        self.last_crop_box = None           # (x1, y1, x2, y2) crop terakhir, koordinat gambar input

    def _target_box(self, img):
        self._n += 1
        use_track = self._track_box is not None and not (
            self.redetect_every and self._n % self.redetect_every == 0)
        if use_track:
            return self._track_box, "track"
        faces = [f for f in self.det.detect(img) if max(f[2] - f[0], f[3] - f[1]) >= self.min_face_px]
        if not faces:
            return None, "none"
        return max(faces, key=lambda f: f[4] * (f[2] - f[0]) * (f[3] - f[1])), "detector"

    def _update_crop(self, box) -> tuple[float, float, float]:
        cx, cy, side = square_crop_box(box, self.margin)
        if self._crop is not None:
            pcx, pcy, pside = self._crop
            if (abs(cx - pcx) < self.hold_shift * pside and abs(cy - pcy) < self.hold_shift * pside
                    and abs(side - pside) < self.hold_scale * pside):
                return self._crop
        self._crop = (cx, cy, side)
        return self._crop

    def process(self, image_bgr: np.ndarray, ts_ms: int, offset: tuple[int, int] = (0, 0),
                scale_to=None) -> FaceObs:
        if (self._track_box is None and self._lost_since_ms is not None
                and ts_ms - self._lost_since_ms >= self.lost_search_after_ms
                and self._lost_n % self.lost_search_every):
            self._lost_n += 1                 # sudah lama hilang dan bukan giliran mencari
            self.last_source = "none"
            self.last_crop_box = None
            return FaceObs(present=False)
        obs = self._process(image_bgr, ts_ms, offset)
        if obs.present:
            self._lost_n, self._lost_since_ms = 0, None
        else:
            self._lost_n += 1
            if self._lost_since_ms is None:
                self._lost_since_ms = ts_ms
        return obs

    def _process(self, image_bgr: np.ndarray, ts_ms: int, offset: tuple[int, int]) -> FaceObs:
        box, src = self._target_box(image_bgr)
        if box is None:
            self._crop = None
            self.last_crop_box = None
            if self.inner_full is not None:   # cadangan: perilaku lama (full frame)
                obs = self.inner_full.process(image_bgr, ts_ms, offset)
                self.last_source = "full" if obs.present else "none"
                if obs.present:
                    self._track_box = self._lm_box(obs.landmarks_px, offset)
                return obs
            self.last_source = "none"
            return FaceObs(present=False)
        cx, cy, side = self._update_crop(box)
        crop, x0, y0 = crop_square(image_bgr, cx, cy, side, self.out_px)
        self.last_crop_box = (x0, y0, x0 + side, y0 + side)
        obs = self.inner.process(crop, ts_ms, (0, 0))
        if not obs.present:
            self._track_box = None            # hilang -> frame berikut pakai detektor
            self._crop = None if src == "track" else self._crop
            if self.inner_full is not None:   # coba full frame di frame yang sama: tidak pernah lebih buruk
                full = self.inner_full.process(image_bgr, ts_ms, offset)
                if full.present:
                    self._track_box = self._lm_box(full.landmarks_px, offset)
                    self.last_source = "full"
                    return full
            self.last_source = "none"
            return obs
        k = side / self.out_px
        obs.landmarks_px = obs.landmarks_px * k + np.array([x0 + offset[0], y0 + offset[1]], dtype=np.float32)
        self._track_box = self._lm_box(obs.landmarks_px, offset)
        self.last_source = src
        return obs

    @staticmethod
    def _lm_box(xy: np.ndarray, offset) -> tuple[float, float, float, float, float]:
        x1, y1 = xy.min(axis=0)
        x2, y2 = xy.max(axis=0)
        return (float(x1 - offset[0]), float(y1 - offset[1]), float(x2 - offset[0]), float(y2 - offset[1]), 1.0)

    def close(self):
        for m in (self.inner, self.inner_full):
            if m is not None and hasattr(m, "close"):
                m.close()
