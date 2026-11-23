"""ByteTrack untuk backend tanpa `.track()` (onnxruntime, tensorrt_native).

Di Jetson, torch sistem crash pada operasi CUDA, jadi `model.track()` Ultralytics tidak bisa dipakai.
Deteksi tetap dari backend apa pun; tracker-nya `BYTETracker` Ultralytics yang hanya memakai numpy
(Kalman + asosiasi IoU), jadi tidak menyentuh CUDA.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

from . import Detection, Detector


class _Boxes:
    """Bentuk minimal yang dibaca BYTETracker.update: .conf, .xywh, .cls, indexing, len."""

    def __init__(self, conf: np.ndarray, xywh: np.ndarray, cls: np.ndarray):
        self.conf, self.xywh, self.cls = conf, xywh, cls

    def __len__(self) -> int:
        return len(self.conf)

    def __getitem__(self, idx) -> "_Boxes":
        return _Boxes(self.conf[idx], self.xywh[idx], self.cls[idx])


def _tracker_args(tracker: str):
    import yaml
    from ultralytics.utils import IterableSimpleNamespace

    p = Path(tracker)
    if not p.exists():
        import ultralytics
        p = Path(ultralytics.__file__).parent / "cfg" / "trackers" / tracker
    with open(p, encoding="utf-8") as f:
        return IterableSimpleNamespace(**yaml.safe_load(f))


class ByteTrackDetector(Detector):
    """Membungkus detektor apa pun dengan ByteTrack; interface `track()` sama dengan UltralyticsDetector."""

    def __init__(self, det: Detector, tracker: str = "bytetrack.yaml", frame_rate: float = 30.0):
        self.det = det
        self.names = det.names
        self.tracker_name = tracker
        self.frame_rate = float(frame_rate)
        self._bt = None

    def _make(self):
        from ultralytics.trackers.byte_tracker import BYTETracker

        args = _tracker_args(self.tracker_name)
        try:   # Ultralytics 8.3: (args, frame_rate); 8.4: frame_rate diambil dari args
            return BYTETracker(args, frame_rate=int(round(self.frame_rate)))
        except TypeError:
            args.frame_rate = int(round(self.frame_rate))
            return BYTETracker(args)

    def predict(self, image_bgr: np.ndarray, conf: float = 0.25) -> list[Detection]:
        return self.det.predict(image_bgr, conf)

    def track(self, image_bgr: np.ndarray, conf: float = 0.25, tracker: str | None = None,
              classes: list[int] | None = None) -> list[Detection]:
        if self._bt is None:
            if tracker:
                self.tracker_name = tracker
            self._bt = self._make()
        dets = self.det.predict(image_bgr, conf)
        if classes is not None:
            keep = set(classes)
            dets = [d for d in dets if d.cls_id in keep]
        if dets:
            xyxy = np.array([d.xyxy for d in dets], dtype=np.float32)
            xywh = np.column_stack([(xyxy[:, 0] + xyxy[:, 2]) / 2, (xyxy[:, 1] + xyxy[:, 3]) / 2,
                                    xyxy[:, 2] - xyxy[:, 0], xyxy[:, 3] - xyxy[:, 1]])
            boxes = _Boxes(np.array([d.conf for d in dets], dtype=np.float32), xywh,
                           np.array([d.cls_id for d in dets], dtype=np.float32))
        else:
            boxes = _Boxes(np.zeros(0, np.float32), np.zeros((0, 4), np.float32), np.zeros(0, np.float32))
        rows = self._bt.update(boxes, image_bgr)
        out: list[Detection] = []
        for r in rows if len(rows) else []:
            # baris: x1, y1, x2, y2, track_id, score, cls, idx
            k = int(r[6])
            name = self.names[k] if k < len(self.names) else str(k)
            out.append(Detection(k, name, float(r[5]), tuple(float(v) for v in r[:4]), int(r[4])))
        return out

    def reset(self) -> None:
        if self._bt is not None:
            self._bt.reset()
