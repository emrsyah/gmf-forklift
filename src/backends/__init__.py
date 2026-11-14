"""Backend inferensi deteksi objek dengan interface yang sama.

- `ultralytics`  : .pt / .onnx / .engine lewat Ultralytics (laptop, juga bisa di Jetson)
- `onnxruntime`  : .onnx murni lewat ONNX Runtime + NMS numpy (tanpa torch; ringan)
- `tensorrt`     : .engine (Fase 2, di-build di Jetson) — lewat Ultralytics
Ganti backend cukup di configs/runtime.yaml.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np


@dataclass
class Detection:
    cls_id: int
    cls_name: str
    conf: float
    xyxy: tuple[float, float, float, float]   # piksel pada gambar input asli
    track_id: int | None = None


class Detector:
    names: list[str]

    def predict(self, image_bgr: np.ndarray, conf: float = 0.25) -> list[Detection]:
        raise NotImplementedError


def make_detector(cfg: dict) -> Detector:
    from src.config import resolve_path

    backend = cfg.get("backend", "ultralytics")
    weights = resolve_path(cfg["weights"])
    if weights is None or not Path(weights).exists():
        raise FileNotFoundError(f"Bobot model tidak ditemukan: {weights}")
    imgsz = int(cfg.get("imgsz", 640))
    names = cfg.get("class_names")
    if backend == "onnxruntime":
        from .ort_backend import OrtYoloDetector
        return OrtYoloDetector(weights, imgsz=imgsz, names=names,
                               providers=cfg.get("providers") or ["CPUExecutionProvider"])
    if backend in ("ultralytics", "tensorrt"):
        from .ultralytics_backend import UltralyticsDetector
        return UltralyticsDetector(weights, imgsz=imgsz, names=names)
    raise ValueError(f"Backend tidak dikenal: {backend}")
