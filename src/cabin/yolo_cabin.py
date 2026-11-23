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


def inside_frac(inner, outer) -> float:
    """Bagian kotak `inner` yang berada di dalam `outer` (0..1)."""
    ix1, iy1, ix2, iy2 = inner
    ox1, oy1, ox2, oy2 = outer
    iw = max(0.0, min(ix2, ox2) - max(ix1, ox1))
    ih = max(0.0, min(iy2, oy2) - max(iy1, oy1))
    area = (ix2 - ix1) * (iy2 - iy1)
    return iw * ih / area if area > 0 else 0.0


def torso_judgeable(box, frame_h: int | None, d10: dict) -> bool:
    """Torso cukup terlihat untuk menilai rompi? Kalau hanya bahu yang masuk gambar (kamera terlalu tinggi,
    uji forklift 2026-11-23), "tanpa rompi" tidak bisa dinilai -> tidak dihitung (bukan pelanggaran)."""
    x1, y1, x2, y2 = box
    w, h = x2 - x1, y2 - y1
    if w <= 0 or h / w < d10.get("torso_min_h_over_w", 0.5):
        return False
    return frame_h is None or h >= d10.get("torso_min_h_frac", 0.15) * frame_h


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
    frame_h = frame_wh[1] if frame_wh else None
    torsos = [d for d in dets if d.cls_name == "torso" and d.conf >= ppe_conf
              and torso_judgeable(d.xyxy, frame_h, d10)]
    thr = d10.get("vest_iou_max", 0.3)
    in_min = d10.get("vest_inside_min", 0.6)

    def has_vest(t) -> bool:   # kotak rompi biasanya lebih kecil dari torso: IoU saja bisa < thr
        return any(iou(t.xyxy, v.xyxy) >= thr or inside_frac(v.xyxy, t.xyxy) >= in_min for v in vests)

    no_vest = any(not has_vest(t) for t in torsos)
    return CabinDetections(phone_conf, no_helmet, no_vest, dets, phone_box, phone_info)
