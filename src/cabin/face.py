"""Model 1+2: MediaPipe Face Landmarker (VIDEO mode, 1 wajah).

Output modul ini adalah `FaceObs` mentah. Sinyal (openness, yaw/pitch/roll, ...) dihitung di
`signals.py` sehingga model wajah bisa diganti (mis. MtDMS di Fase 2) tanpa mengubah indikator.
"""
from __future__ import annotations

import math
from dataclasses import dataclass, field

import cv2
import numpy as np

from src.config import resolve_path
from src.preprocess import resize_max_side


@dataclass
class FaceObs:
    present: bool
    landmarks_px: np.ndarray | None = None      # (478, 2) piksel pada frame input (setelah offset ROI)
    landmarks_z: np.ndarray | None = None
    blendshapes: dict = field(default_factory=dict)
    matrix: np.ndarray | None = None            # 4x4 facial transformation matrix
    score: float = 0.0
    yaw_raw: float = 0.0                        # derajat, sebelum dikurangi baseline kalibrasi
    pitch_raw: float = 0.0
    roll_raw: float = 0.0
    # True = hanya pose kepala (cadangan PnP dari 5 titik detektor saat landmark gagal); mata/mulut tak terbaca
    pose_only: bool = False
    # pose_only: titik tengah mata kanan & kiri dari detektor (2x2 piksel), untuk classifier mata (A9)
    eye_pts: np.ndarray | None = None
    eye_closed_p: float | None = None           # P(mata tertutup) dari classifier (A9); None = tidak dihitung


def pose_from_matrix(m: np.ndarray) -> tuple[float, float, float]:
    """Yaw/pitch/roll (derajat) dari facial transformation matrix MediaPipe.

    Ruang metrik MediaPipe: kamera di origin menghadap -Z, +Y ke atas, +X ke kanan gambar.
    Sumbu Z wajah kanonik menunjuk keluar dari hidung. Kita pakai arah vektor, bukan Euler,
    agar tidak ambigu urutan rotasi:
      pitch < 0  -> hidung menunjuk ke bawah (menunduk)
      yaw   > 0  -> hidung menunjuk ke +X kamera (kanan gambar = kiri pengemudi jika tidak di-mirror)
      roll       -> kemiringan sumbu atas wajah terhadap +Y kamera
    Tanda akhir bisa dibalik lewat config (yaw_sign/pitch_sign/roll_sign); cek dengan tools/check_pose.py.
    """
    r = np.asarray(m, dtype=float)[:3, :3]
    fwd = r[:, 2]
    up = r[:, 1]
    yaw = math.degrees(math.atan2(fwd[0], fwd[2]))
    pitch = math.degrees(math.atan2(fwd[1], math.hypot(fwd[0], fwd[2])))
    roll = math.degrees(math.atan2(-up[0], up[1]))
    return yaw, pitch, roll


class MediaPipeFace:
    def __init__(self, cfg: dict):
        from mediapipe.tasks.python import BaseOptions, vision

        path = resolve_path(cfg.get("model_path", "models/mediapipe/face_landmarker.task"))
        opts = vision.FaceLandmarkerOptions(
            base_options=BaseOptions(model_asset_path=str(path)),
            running_mode=vision.RunningMode.VIDEO,
            num_faces=1,
            min_face_detection_confidence=cfg.get("min_detection_conf", 0.5),
            min_face_presence_confidence=cfg.get("min_presence_conf", 0.5),
            min_tracking_confidence=cfg.get("min_tracking_conf", 0.5),
            output_face_blendshapes=True,
            output_facial_transformation_matrixes=True,
        )
        self._lm = vision.FaceLandmarker.create_from_options(opts)
        self.max_side = int(cfg.get("max_side_px", 640))
        self.signs = (cfg.get("yaw_sign", 1), cfg.get("pitch_sign", 1), cfg.get("roll_sign", 1))
        self._last_ts = -1

    def process(self, image_bgr: np.ndarray, ts_ms: int, offset: tuple[int, int] = (0, 0),
                scale_to: tuple[int, int] | None = None) -> FaceObs:
        import mediapipe as mp

        # VIDEO mode butuh timestamp naik ketat
        ts = int(ts_ms)
        if ts <= self._last_ts:
            ts = self._last_ts + 1
        self._last_ts = ts

        h0, w0 = image_bgr.shape[:2]
        small = resize_max_side(image_bgr, self.max_side)
        rgb = cv2.cvtColor(small, cv2.COLOR_BGR2RGB)
        res = self._lm.detect_for_video(mp.Image(image_format=mp.ImageFormat.SRGB, data=np.ascontiguousarray(rgb)), ts)
        if not res.face_landmarks:
            return FaceObs(present=False)
        lms = res.face_landmarks[0]
        xy = np.array([[p.x * w0 + offset[0], p.y * h0 + offset[1]] for p in lms], dtype=np.float32)
        z = np.array([p.z for p in lms], dtype=np.float32)
        bs = {}
        if res.face_blendshapes:
            bs = {c.category_name: float(c.score) for c in res.face_blendshapes[0]}
        mat = None
        yaw = pitch = roll = 0.0
        if res.facial_transformation_matrixes:
            mat = np.array(res.facial_transformation_matrixes[0], dtype=float).reshape(4, 4)
            yaw, pitch, roll = pose_from_matrix(mat)
            yaw *= self.signs[0]; pitch *= self.signs[1]; roll *= self.signs[2]
        # Face Landmarker tidak memberi skor eksplisit; deteksi lolos min_presence_conf -> 1.0
        return FaceObs(True, xy, z, bs, mat, 1.0, yaw, pitch, roll)

    def close(self):
        self._lm.close()


def make_face_model(cfg: dict):
    kind = cfg.get("model", "mediapipe")
    if kind == "mediapipe":
        crop = cfg.get("crop") or {}
        if not crop.get("enabled", False):
            return MediaPipeFace(cfg)
        from .face_crop import CropFace, make_face_detector   # detektor + crop dulu (face_crop.py)
        full = MediaPipeFace(cfg) if crop.get("fallback_full", True) else None
        return CropFace(MediaPipeFace(cfg), make_face_detector(crop), crop, inner_full=full)
    raise ValueError(f"Model wajah belum tersedia: {kind} (MtDMS direncanakan untuk Fase 2)")
