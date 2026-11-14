"""Model 3: YOLO kabin (phone, helmet, head_no_helmet, vest, torso) -> sinyal D-06 / D-10.

`no_vest` bukan kelas: torso yang tidak punya vest tumpang-tindih (IoU < vest_iou_max).
"""
from __future__ import annotations

from dataclasses import dataclass, field

from src.backends import Detection


@dataclass
class CabinDetections:
    phone_conf: float = 0.0
    no_helmet: bool = False
    no_vest: bool = False
    dets: list[Detection] = field(default_factory=list)


def iou(a, b) -> float:
    ax1, ay1, ax2, ay2 = a
    bx1, by1, bx2, by2 = b
    iw = max(0.0, min(ax2, bx2) - max(ax1, bx1))
    ih = max(0.0, min(ay2, by2) - max(ay1, by1))
    inter = iw * ih
    if inter <= 0:
        return 0.0
    union = (ax2 - ax1) * (ay2 - ay1) + (bx2 - bx1) * (by2 - by1) - inter
    return inter / union if union > 0 else 0.0


def interpret(dets: list[Detection], d06: dict, d10: dict) -> CabinDetections:
    phone_conf = max((d.conf for d in dets if d.cls_name == "phone"), default=0.0)
    ppe_conf = d10.get("conf", 0.40)
    no_helmet = any(d.cls_name == "head_no_helmet" and d.conf >= ppe_conf for d in dets)
    vests = [d for d in dets if d.cls_name == "vest" and d.conf >= ppe_conf]
    torsos = [d for d in dets if d.cls_name == "torso" and d.conf >= ppe_conf]
    thr = d10.get("vest_iou_max", 0.3)
    no_vest = any(all(iou(t.xyxy, v.xyxy) < thr for v in vests) for t in torsos)
    return CabinDetections(phone_conf, no_helmet, no_vest, dets)
