"""Estimasi jarak lantai dari titik kaki (Tech Spec bag. 6, "Estimasi jarak (tanpa model)").

Geometri (kamera di tinggi h, menunduk theta di bawah horizontal):
    (u, v) titik kaki -> hilangkan distorsi -> koordinat ternormalisasi (x_n, y_n)
    angle      = theta + atan(y_n)            # sudut sinar ke titik kaki di bawah horizon
    distance_m = h / tan(angle)               # jarak maju di lantai dari titik di bawah kamera
    lateral_m  = h * x_n / (y_n*cos(theta) + sin(theta))   # offset samping (informasi)

Cadangan jika titik kaki tidak andal (kotak menyentuh tepi bawah / sinar di atas horizon):
    d ~= fy * H_kelas / tinggi_kotak_px       # H_kelas dari config (person 1,65 m)

Untuk kotak yang terpotong tepi bawah, kedua estimasi adalah batas ATAS jarak sebenarnya
(kaki sebenarnya di bawah frame; tinggi kotak terpotong), sehingga diambil minimumnya.
Semua angka geometri dari configs/front_camera.yaml.
"""
from __future__ import annotations

import math
from dataclasses import dataclass

import cv2
import numpy as np

METHOD_FOOT = "foot"
METHOD_HEIGHT = "height"
METHOD_CLIPPED = "clipped"   # kotak terpotong tepi bawah: min(foot@tepi, height)


@dataclass
class DistanceEstimate:
    distance_m: float
    lateral_m: float | None
    method: str


class CameraModel:
    """K + distorsi (normal atau fisheye), diskalakan otomatis ke resolusi frame."""

    def __init__(self, K, dist, fisheye: bool = False, image_size: tuple[int, int] | None = None):
        self.K = np.asarray(K, dtype=np.float64).reshape(3, 3)
        self.dist = np.asarray(dist if dist is not None else [], dtype=np.float64).ravel()
        self.fisheye = bool(fisheye)
        if self.fisheye and self.dist.size != 4:
            raise ValueError("Distorsi fisheye harus tepat 4 koefisien (k1..k4)")
        self.image_size = tuple(int(x) for x in image_size) if image_size else None
        self._scaled: dict[tuple[int, int], np.ndarray] = {}

    def K_for(self, w: int, h: int) -> np.ndarray:
        if self.image_size is None or (w, h) == self.image_size:
            return self.K
        key = (int(w), int(h))
        if key not in self._scaled:
            sx, sy = w / self.image_size[0], h / self.image_size[1]
            K = self.K.copy()
            K[0, 0] *= sx
            K[0, 2] *= sx
            K[0, 1] *= sx
            K[1, 1] *= sy
            K[1, 2] *= sy
            self._scaled[key] = K
        return self._scaled[key]

    def normalize(self, pts_uv: np.ndarray, w: int, h: int) -> np.ndarray:
        """Piksel terdistorsi (N,2) -> koordinat ternormalisasi tanpa distorsi (N,2). P tidak diberikan."""
        K = self.K_for(w, h)
        src = np.asarray(pts_uv, dtype=np.float64).reshape(-1, 1, 2)
        if self.fisheye:
            out = cv2.fisheye.undistortPoints(src, K, self.dist)
        else:
            out = cv2.undistortPoints(src, K, self.dist)
        return out.reshape(-1, 2)


class DistanceEstimator:
    def __init__(self, camera: CameraModel, h_m: float, theta_deg: float,
                 class_height_m: dict[str, float] | None = None, bottom_margin_px: float = 0.0,
                 min_ray_angle_deg: float = 0.0, max_distance_m: float = math.inf):
        self.camera = camera
        self.h_m = float(h_m)
        self.theta = math.radians(float(theta_deg))
        self.class_height_m = dict(class_height_m or {})
        self.bottom_margin_px = float(bottom_margin_px)
        self.min_ray_angle = math.radians(float(min_ray_angle_deg))
        self.max_distance_m = float(max_distance_m)

    @classmethod
    def from_config(cls, cam_cfg: dict) -> "DistanceEstimator":
        c = cam_cfg["camera"]
        m = cam_cfg["mount"]
        d = cam_cfg.get("distance", {})
        camera = CameraModel(c["K"], c.get("dist"), c.get("fisheye", False), c.get("image_size"))
        return cls(camera, m["h_m"], m["theta_deg"], d.get("class_height_m"),
                   d.get("bottom_margin_px", 0.0), d.get("min_ray_angle_deg", 0.0),
                   d.get("max_distance_m", math.inf))

    # --- metode dasar -------------------------------------------------------------------------
    def ground_point(self, u: float, v: float, w: int, h: int) -> tuple[float, float] | None:
        """Titik piksel di lantai -> (maju_m, samping_m), atau None jika di atas horizon / tidak valid."""
        xn, yn = self.camera.normalize(np.array([[u, v]]), w, h)[0]
        if not (math.isfinite(xn) and math.isfinite(yn)):
            return None
        angle = self.theta + math.atan(yn)
        if angle <= self.min_ray_angle:
            return None
        if angle >= math.pi / 2:
            return 0.0, 0.0   # tepat di bawah kamera
        fwd = self.h_m / math.tan(angle)
        denom = yn * math.cos(self.theta) + math.sin(self.theta)
        lat = self.h_m * xn / denom if denom > 0 else None
        if not math.isfinite(fwd) or fwd < 0:
            return None
        return fwd, lat

    def height_distance(self, box_h_px: float, cls_name: str, w: int, h: int) -> float | None:
        H = self.class_height_m.get(cls_name)
        if H is None or box_h_px <= 0:
            return None
        fy = self.camera.K_for(w, h)[1, 1]
        return fy * H / box_h_px

    def row_for_distance(self, distance_m: float, w: int, h: int) -> float | None:
        """Baris piksel (tanpa distorsi, di sumbu tengah) untuk jarak lantai tertentu. Untuk overlay."""
        if distance_m <= 0:
            return None
        yn = math.tan(math.atan(self.h_m / distance_m) - self.theta)
        K = self.camera.K_for(w, h)
        return K[1, 2] + K[1, 1] * yn

    # --- API utama ------------------------------------------------------------------------------
    def estimate(self, xyxy, cls_name: str, w: int, h: int) -> DistanceEstimate | None:
        x1, y1, x2, y2 = (float(t) for t in xyxy)
        u = 0.5 * (x1 + x2)
        clipped = y2 >= h - 1 - self.bottom_margin_px
        foot = self.ground_point(u, min(y2, h - 1), w, h)
        by_h = self.height_distance(y2 - y1, cls_name, w, h)

        if clipped:
            cands = [c for c in (foot[0] if foot else None, by_h) if c is not None]
            if not cands:
                return None
            est = DistanceEstimate(min(cands), foot[1] if foot else None, METHOD_CLIPPED)
        elif foot is not None:
            est = DistanceEstimate(foot[0], foot[1], METHOD_FOOT)
        elif by_h is not None:
            est = DistanceEstimate(by_h, None, METHOD_HEIGHT)
        else:
            return None
        if not math.isfinite(est.distance_m) or est.distance_m > self.max_distance_m:
            return None
        return est
