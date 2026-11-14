"""YOLO (v8/11, export Ultralytics) di ONNX Runtime murni: letterbox + decode + NMS numpy.

Tidak butuh torch/ultralytics sehingga cocok di perangkat edge. Output ONNX yang diharapkan:
(1, 4 + nc, N) dengan kotak cx, cy, w, h dalam piksel input model (format export Ultralytics, nms=False).
"""
from __future__ import annotations

import ast
from pathlib import Path

import cv2
import numpy as np

from . import Detection, Detector


def letterbox(img: np.ndarray, size: int, color=(114, 114, 114)):
    h, w = img.shape[:2]
    r = min(size / h, size / w)
    nw, nh = int(round(w * r)), int(round(h * r))
    dw, dh = (size - nw) / 2, (size - nh) / 2
    if (w, h) != (nw, nh):
        img = cv2.resize(img, (nw, nh), interpolation=cv2.INTER_LINEAR)
    top, bottom = int(round(dh - 0.1)), int(round(dh + 0.1))
    left, right = int(round(dw - 0.1)), int(round(dw + 0.1))
    img = cv2.copyMakeBorder(img, top, bottom, left, right, cv2.BORDER_CONSTANT, value=color)
    return img, r, (left, top)


class OrtYoloDetector(Detector):
    def __init__(self, weights: str | Path, imgsz: int = 640, names: list[str] | None = None,
                 providers: list[str] | None = None, iou: float = 0.7, max_det: int = 300):
        import onnxruntime as ort

        self.sess = ort.InferenceSession(str(weights), providers=providers or ["CPUExecutionProvider"])
        inp = self.sess.get_inputs()[0]
        self.input_name = inp.name
        shp = inp.shape
        self.imgsz = int(shp[2]) if isinstance(shp[2], int) else imgsz
        self.iou = iou
        self.max_det = max_det
        meta = self.sess.get_modelmeta().custom_metadata_map or {}
        model_names = None
        if "names" in meta:
            try:
                d = ast.literal_eval(meta["names"])
                model_names = [d[i] for i in sorted(d)]
            except Exception:
                model_names = None
        self.names = list(names) if names else (model_names or [])

    def predict(self, image_bgr: np.ndarray, conf: float = 0.25) -> list[Detection]:
        img, r, (pl, pt) = letterbox(image_bgr, self.imgsz)
        x = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).transpose(2, 0, 1)[None].astype(np.float32) / 255.0
        out = self.sess.run(None, {self.input_name: np.ascontiguousarray(x)})[0]
        pred = out[0].T if out.shape[1] < out.shape[2] else out[0]   # (N, 4+nc)
        boxes, scores = pred[:, :4], pred[:, 4:]
        cls = scores.argmax(1)
        cf = scores[np.arange(len(cls)), cls]
        keep = cf >= conf
        if not keep.any():
            return []
        boxes, cls, cf = boxes[keep], cls[keep], cf[keep]
        xyxy = np.empty_like(boxes)
        xyxy[:, 0] = boxes[:, 0] - boxes[:, 2] / 2
        xyxy[:, 1] = boxes[:, 1] - boxes[:, 3] / 2
        xyxy[:, 2] = boxes[:, 0] + boxes[:, 2] / 2
        xyxy[:, 3] = boxes[:, 1] + boxes[:, 3] / 2
        # NMS per kelas (offset kotak per kelas, sama seperti Ultralytics)
        off = cls[:, None].astype(np.float32) * 7680.0
        b = xyxy + off
        idx = cv2.dnn.NMSBoxes(
            [[float(a), float(c), float(e - a), float(d - c)] for a, c, e, d in b],
            cf.astype(float).tolist(), conf, self.iou)
        idx = np.array(idx).reshape(-1)[: self.max_det]
        h, w = image_bgr.shape[:2]
        dets: list[Detection] = []
        for i in idx:
            x1, y1, x2, y2 = xyxy[i]
            x1 = float(np.clip((x1 - pl) / r, 0, w)); x2 = float(np.clip((x2 - pl) / r, 0, w))
            y1 = float(np.clip((y1 - pt) / r, 0, h)); y2 = float(np.clip((y2 - pt) / r, 0, h))
            k = int(cls[i])
            name = self.names[k] if k < len(self.names) else str(k)
            dets.append(Detection(k, name, float(cf[i]), (x1, y1, x2, y2)))
        dets.sort(key=lambda d: -d.conf)
        return dets
