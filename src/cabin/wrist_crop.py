"""A7 (D-06): potongan ponsel di sekitar tangan pengemudi dari keypoint pose (pergelangan tangan).

Ponsel di tangan sering kecil dan sebagian tertutup jari (Lembar A 12-14). Potongan persegi berpusat sedikit
di depan pergelangan (arah siku -> pergelangan) diperbesar ke input YOLO, jadi ponsel tampak beberapa kali
lebih besar. Pose TIDAK pernah memicu D-06 sendiri (minum, garuk, tangan di dagu): hanya menambah kandidat
`phone` dari detektor yang sama, yang tetap melewati PhoneFilter + D06 (conf, rasio, jendela waktu).
Semua angka dari configs/thresholds.yaml D06_phone.wrist_crop.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from src.backends import Detection
from src.backends.pose import L_ELBOW, L_SHOULDER, L_WRIST, NOSE, R_ELBOW, R_SHOULDER, R_WRIST, Person

from .phone_crop import PhoneCrop


def pick_driver(people: list[Person], face_box=None) -> Person | None:
    """Pengemudi = orang yang kotaknya memuat pusat wajah (terdekat hidung); tanpa wajah = kotak terbesar."""
    if not people:
        return None
    if face_box is not None:
        cx, cy = (face_box[0] + face_box[2]) / 2, (face_box[1] + face_box[3]) / 2
        inside = [p for p in people if p.xyxy[0] <= cx <= p.xyxy[2] and p.xyxy[1] <= cy <= p.xyxy[3]]
        if inside:
            return min(inside, key=lambda p: float(np.hypot(p.kpts[NOSE, 0] - cx, p.kpts[NOSE, 1] - cy)))
    return max(people, key=lambda p: (p.xyxy[2] - p.xyxy[0]) * (p.xyxy[3] - p.xyxy[1]))


class WristCrop:
    def __init__(self, cfg: dict | None):
        c = cfg or {}
        self.enabled = bool(c.get("enabled", False))
        self.pose_conf = float(c.get("pose_conf", 0.3))
        self.kp_conf = float(c.get("kp_conf", 0.5))
        self.side_u = float(c.get("side_u", 2.0))            # sisi potongan dalam satuan u (sisi wajah)
        self.ahead = float(c.get("ahead", 0.35))             # pusat digeser sekian x (pergelangan - siku)
        self.shoulder_to_u = float(c.get("shoulder_to_u", 0.5))   # u ~ 0,5 x lebar bahu (tanpa wajah)
        self.box_w_to_u = float(c.get("box_w_to_u", 0.25))   # cadangan terakhir: u ~ 0,25 x lebar kotak orang
        self.min_side_px = int(c.get("min_side_px", 64))
        self.max_crops = int(c.get("max_crops", 2))
        self.merge_iou = float(c.get("merge_iou", 0.5))
        self.classes = set(c.get("classes", ["phone"]))
        self._merge = PhoneCrop({"merge_iou": self.merge_iou}).merge
        self.last_boxes: list[tuple[int, int, int, int]] = []    # overlay/debug
        self.last_wrists: list[tuple[float, float]] = []

    def unit(self, p: Person, face_box) -> float:
        if face_box is not None:
            return float(max(face_box[2] - face_box[0], face_box[3] - face_box[1]))
        k = p.kpts
        if k[L_SHOULDER, 2] >= self.kp_conf and k[R_SHOULDER, 2] >= self.kp_conf:
            sw = float(np.hypot(*(k[L_SHOULDER, :2] - k[R_SHOULDER, :2])))
            if sw > 0:
                return sw * self.shoulder_to_u
        return (p.xyxy[2] - p.xyxy[0]) * self.box_w_to_u

    def boxes(self, p: Person, face_box, frame_wh: tuple[int, int]) -> list[tuple[int, int, int, int]]:
        u = self.unit(p, face_box)
        if u <= 0:
            return []
        k = p.kpts
        W, H = frame_wh
        half = self.side_u * u / 2
        out = []
        self.last_wrists = []
        wrists = sorted(((L_WRIST, L_ELBOW), (R_WRIST, R_ELBOW)), key=lambda we: -k[we[0], 2])
        for wi, ei in wrists[: self.max_crops]:
            if k[wi, 2] < self.kp_conf:
                continue
            c = k[wi, :2].astype(float).copy()
            self.last_wrists.append((float(c[0]), float(c[1])))
            if k[ei, 2] >= self.kp_conf:
                c += self.ahead * (k[wi, :2] - k[ei, :2])
            x1, y1 = int(max(0, c[0] - half)), int(max(0, c[1] - half))
            x2, y2 = int(min(W, c[0] + half)), int(min(H, c[1] + half))
            if min(x2 - x1, y2 - y1) < self.min_side_px:
                continue
            out.append((x1, y1, x2, y2))
        return out

    def run(self, detector, pose, image: np.ndarray, full: list[Detection], face_box, conf: float) -> list[Detection]:
        self.last_boxes = []
        if not self.enabled or pose is None:
            return full
        p = pick_driver(pose.predict(image, conf=self.pose_conf), face_box)
        if p is None:
            self.last_wrists = []
            return full
        extra = []
        for x1, y1, x2, y2 in self.boxes(p, face_box, (image.shape[1], image.shape[0])):
            self.last_boxes.append((x1, y1, x2, y2))
            for d in detector.predict(np.ascontiguousarray(image[y1:y2, x1:x2]), conf=conf):
                if d.cls_name in self.classes:
                    a, b, c, e = d.xyxy
                    extra.append(replace(d, xyxy=(a + x1, b + y1, c + x1, e + y1)))
        return self._merge(full, extra)
