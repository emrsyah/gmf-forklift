"""Detektor + ByteTrack kamera depan (Model 4 + 5, Tech Spec bag. 6).

Membungkus `make_detector` (src/backends). Tracking memakai `UltralyticsDetector.track`
(ByteTrack, persist=True), jadi backend harus `ultralytics` atau `tensorrt` (.pt/.onnx/.engine).
Nama kelas model dipetakan ke nama kelas kita lewat `classes.aliases` di front_camera.yaml
(sementara untuk COCO); kelas yang dikonfigurasi tapi tidak ada di model diabaikan.
"""
from __future__ import annotations

import logging

import numpy as np

from src.backends import Detection, make_detector

log = logging.getLogger(__name__)


class FrontDetector:
    def __init__(self, yolo_cfg: dict, cam_cfg: dict, person_conf: float, detector=None):
        self.det = detector if detector is not None else make_detector(yolo_cfg)
        if not hasattr(self.det, "track"):
            raise ValueError("Kamera depan butuh tracker: pakai backend 'ultralytics' / 'tensorrt' "
                             "(Ultralytics juga bisa memuat .onnx).")
        self.tracker = yolo_cfg.get("tracker", "bytetrack.yaml")
        self.person_class = yolo_cfg.get("person_class", "person")
        cls_cfg = cam_cfg.get("classes", {})
        self.aliases: dict[str, str] = dict(cls_cfg.get("aliases") or {})
        self.obstacle_classes = set(cls_cfg.get("obstacle_classes") or [])
        self.person_conf = float(person_conf)
        self.obstacle_conf = float(cls_cfg.get("obstacle_conf", person_conf))
        keep = {self.person_class} | self.obstacle_classes
        self.class_ids = [i for i, n in enumerate(self.det.names) if self.alias(n) in keep]
        present = {self.alias(self.det.names[i]) for i in self.class_ids}
        missing = sorted(keep - present)
        if missing:
            log.info("Kelas tidak ada di model depan (diabaikan): %s", ", ".join(missing))
        if self.person_class not in present:
            log.warning("Kelas '%s' tidak ada di model depan", self.person_class)
        self._min_conf = min(self.person_conf, self.obstacle_conf)

    def alias(self, name: str) -> str:
        return self.aliases.get(name, name)

    def track(self, image_bgr: np.ndarray) -> list[Detection]:
        if not self.class_ids:
            return []
        dets = self.det.track(image_bgr, conf=self._min_conf, tracker=self.tracker,
                              classes=self.class_ids)
        out: list[Detection] = []
        for d in dets:
            name = self.alias(d.cls_name)
            thr = self.person_conf if name == self.person_class else self.obstacle_conf
            if d.conf < thr:
                continue
            d.cls_name = name
            out.append(d)
        return out

    def reset(self) -> None:
        """Reset state ByteTrack (mis. saat ganti video)."""
        model = getattr(self.det, "model", None)
        predictor = getattr(model, "predictor", None)
        for t in getattr(predictor, "trackers", None) or []:
            t.reset()
