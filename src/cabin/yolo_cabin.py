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
    phone_box: tuple | None = None            # ponsel yang lolos filter (None = tidak ada)
    phone_info: dict = field(default_factory=dict)   # referensi/alasan tolak filter ponsel (debug)


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


def interpret(dets: list[Detection], d06: dict, d10: dict, phone_filter=None, ts_ms: int = 0,
              face_box=None, yaw: float | None = None, frame_wh: tuple[int, int] | None = None) -> CabinDetections:
    """phone_filter (src.cabin.phone_filter.PhoneFilter) menyaring `phone` relatif terhadap wajah pengemudi;
    tanpa filter, phone_conf = conf ponsel tertinggi (perilaku lama)."""
    phone_box, phone_info = None, {}
    if phone_filter is not None:
        v = phone_filter.update(dets, ts_ms, face_box, yaw, frame_wh)
        phone_conf, phone_box, phone_info = v.conf, v.box, v.info
    else:
        best = max((d for d in dets if d.cls_name == "phone"), key=lambda d: d.conf, default=None)
        phone_conf = best.conf if best else 0.0
        phone_box = best.xyxy if best else None
    ppe_conf = d10.get("conf", 0.40)
    no_helmet = any(d.cls_name == "head_no_helmet" and d.conf >= ppe_conf for d in dets)
    vests = [d for d in dets if d.cls_name == "vest" and d.conf >= ppe_conf]
    torsos = [d for d in dets if d.cls_name == "torso" and d.conf >= ppe_conf]
    thr = d10.get("vest_iou_max", 0.3)
    no_vest = any(all(iou(t.xyxy, v.xyxy) < thr for v in vests) for t in torsos)
    return CabinDetections(phone_conf, no_helmet, no_vest, dets, phone_box, phone_info)
