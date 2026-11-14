from __future__ import annotations

from pathlib import Path

import numpy as np

from . import Detection, Detector


class UltralyticsDetector(Detector):
    """Membungkus ultralytics.YOLO untuk .pt, .onnx, dan .engine (TensorRT di Jetson)."""

    def __init__(self, weights: str | Path, imgsz: int = 640, names: list[str] | None = None):
        from ultralytics import YOLO

        self.model = YOLO(str(weights), task="detect")
        self.imgsz = imgsz
        model_names = self.model.names if isinstance(self.model.names, dict) else dict(enumerate(self.model.names))
        self.names = list(names) if names else [model_names[i] for i in sorted(model_names)]

    def _to_dets(self, r) -> list[Detection]:
        out: list[Detection] = []
        if r.boxes is None or len(r.boxes) == 0:
            return out
        xyxy = r.boxes.xyxy.cpu().numpy()
        confs = r.boxes.conf.cpu().numpy()
        clss = r.boxes.cls.cpu().numpy().astype(int)
        ids = r.boxes.id.cpu().numpy().astype(int) if r.boxes.id is not None else [None] * len(clss)
        for b, c, k, tid in zip(xyxy, confs, clss, ids):
            name = self.names[k] if k < len(self.names) else str(k)
            out.append(Detection(int(k), name, float(c), tuple(float(v) for v in b),
                                 None if tid is None else int(tid)))
        return out

    def predict(self, image_bgr: np.ndarray, conf: float = 0.25) -> list[Detection]:
        r = self.model.predict(image_bgr, imgsz=self.imgsz, conf=conf, verbose=False)[0]
        return self._to_dets(r)

    def track(self, image_bgr: np.ndarray, conf: float = 0.25, tracker: str = "bytetrack.yaml",
              classes: list[int] | None = None) -> list[Detection]:
        r = self.model.track(image_bgr, imgsz=self.imgsz, conf=conf, persist=True,
                             tracker=tracker, classes=classes, verbose=False)[0]
        return self._to_dets(r)
