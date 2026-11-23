"""YOLO pose (export Ultralytics, 17 keypoint COCO) tanpa torch: ONNX Runtime atau TensorRT native.

Output yang diharapkan: (1, 5 + 17*3, N) = cx, cy, w, h, conf orang, lalu (x, y, conf) per keypoint dalam
piksel input model. Dipakai A7 (D-06): pergelangan tangan pengemudi -> potongan ponsel di sekitar tangan.
Ganti backend lewat configs/runtime.yaml cabin.pose (sama dengan cabin.yolo).
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from .ort_backend import preprocess

# indeks keypoint COCO
NOSE, L_SHOULDER, R_SHOULDER, L_ELBOW, R_ELBOW, L_WRIST, R_WRIST, L_HIP, R_HIP = 0, 5, 6, 7, 8, 9, 10, 11, 12
N_KPTS = 17


@dataclass
class Person:
    conf: float
    xyxy: tuple[float, float, float, float]
    kpts: np.ndarray                     # (17, 3): x, y (piksel gambar asli), conf


def decode_pose(out: np.ndarray, image_shape, r: float, pad, conf: float, iou: float = 0.7,
                max_det: int = 10) -> list[Person]:
    pl, pt = pad
    pred = out[0].T if out.shape[1] == 5 + 3 * N_KPTS else out[0]      # (N, 56)
    pred = pred.astype(np.float32, copy=False)
    keep = pred[:, 4] >= conf
    if not keep.any():
        return []
    pred = pred[keep]
    cx, cy, w, h = pred[:, 0], pred[:, 1], pred[:, 2], pred[:, 3]
    xyxy = np.stack([cx - w / 2, cy - h / 2, cx + w / 2, cy + h / 2], 1)
    idx = cv2.dnn.NMSBoxes([[float(a), float(b), float(c - a), float(d - b)] for a, b, c, d in xyxy],
                           pred[:, 4].astype(float).tolist(), conf, iou)
    idx = np.array(idx).reshape(-1)[:max_det]
    H, W = image_shape[:2]
    people = []
    for i in idx:
        k = pred[i, 5:].reshape(-1, 3).copy()
        k[:, 0] = np.clip((k[:, 0] - pl) / r, 0, W)
        k[:, 1] = np.clip((k[:, 1] - pt) / r, 0, H)
        x1, y1, x2, y2 = xyxy[i]
        box = (float(np.clip((x1 - pl) / r, 0, W)), float(np.clip((y1 - pt) / r, 0, H)),
               float(np.clip((x2 - pl) / r, 0, W)), float(np.clip((y2 - pt) / r, 0, H)))
        people.append(Person(float(pred[i, 4]), box, k))
    people.sort(key=lambda p: -p.conf)
    return people


class PoseEstimator:
    def predict(self, image_bgr: np.ndarray, conf: float = 0.3) -> list[Person]:
        raise NotImplementedError


class OrtPose(PoseEstimator):
    def __init__(self, weights, imgsz: int = 640, providers=None):
        import onnxruntime as ort

        self.sess = ort.InferenceSession(str(weights), providers=providers or ["CPUExecutionProvider"])
        inp = self.sess.get_inputs()[0]
        self.input_name = inp.name
        self.imgsz = int(inp.shape[2]) if isinstance(inp.shape[2], int) else imgsz

    def predict(self, image_bgr, conf=0.3):
        x, r, pad = preprocess(image_bgr, self.imgsz)
        return decode_pose(self.sess.run(None, {self.input_name: x})[0], image_bgr.shape, r, pad, conf)


class TrtPose(PoseEstimator):
    """Engine TensorRT (build di Jetson, lalu tools.wrap_trt_engine). Memakai mesin I/O TrtYoloDetector."""

    def __init__(self, weights, imgsz: int = 640):
        from .trt_backend import TrtYoloDetector

        self._trt = TrtYoloDetector(weights, imgsz=imgsz, names=["person"])
        self.imgsz = self._trt.imgsz

    def predict(self, image_bgr, conf=0.3):
        x, r, pad = preprocess(image_bgr, self.imgsz)
        return decode_pose(self._trt._infer(x), image_bgr.shape, r, pad, conf)


def make_pose(cfg: dict) -> PoseEstimator:
    from src.config import resolve_path

    weights = resolve_path(cfg["weights"])
    if weights is None or not Path(weights).exists():
        raise FileNotFoundError(f"Bobot pose tidak ditemukan: {weights}")
    backend = cfg.get("backend", "onnxruntime")
    imgsz = int(cfg.get("imgsz", 640))
    if backend == "onnxruntime":
        return OrtPose(weights, imgsz, cfg.get("providers"))
    if backend == "tensorrt_native":
        return TrtPose(weights, imgsz)
    raise ValueError(f"Backend pose tidak dikenal: {backend} (onnxruntime | tensorrt_native)")
