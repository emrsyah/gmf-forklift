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


def helmet_wins(head: Detection, helmets: list[Detection], d10: dict) -> bool:
    """`helmet` dan `head_no_helmet` di kepala yang sama (kebingungan model): pakai yang skornya lebih tinggi."""
    thr = d10.get("helmet_conflict_iou", 0.3)
    return any(iou(head.xyxy, h.xyxy) >= thr and h.conf >= head.conf for h in helmets)


def is_driver_head(box, face_box, d10: dict) -> bool:
    """Kotak kepala milik pengemudi = sebagian besar kotak wajah ada di dalamnya."""
    return inside_frac(face_box[:4], box) >= d10.get("driver_head_face_inside_min", 0.5)


def is_driver_torso(box, face_box, d10: dict) -> bool:
    """Torso pengemudi = pusat torso di bawah pusat wajah dan pusat wajah (x) dalam rentang torso
    (diperlebar `margin` x lebar). Pusat, bukan tepi atas: label 'no vest' sebagian dataset = badan atas + kepala."""
    fx1, fy1, fx2, fy2 = face_box[:4]
    x1, y1, x2, y2 = box
    m = d10.get("driver_torso_x_margin", 0.25) * (x2 - x1)
    cx = (fx1 + fx2) / 2
    return x1 - m <= cx <= x2 + m and (y1 + y2) / 2 > (fy1 + fy2) / 2 and y2 > fy2


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
    helmets = [d for d in dets if d.cls_name == "helmet" and d.conf >= ppe_conf]
    heads = [d for d in dets if d.cls_name == "head_no_helmet" and d.conf >= ppe_conf
             and not helmet_wins(d, helmets, d10)]
    vests = [d for d in dets if d.cls_name == "vest" and d.conf >= ppe_conf]
    frame_h = frame_wh[1] if frame_wh else None
    torsos = [d for d in dets if d.cls_name == "torso" and d.conf >= ppe_conf
              and torso_judgeable(d.xyxy, frame_h, d10)]
    if d10.get("driver_only", False) and face_box is not None:
        # hanya kepala/torso milik pengemudi (wajah dari Face Landmarker); orang lain di gambar diabaikan.
        # Landmark dagu yang terpotong tepi bawah jatuh di luar gambar, kotak YOLO tidak -> clip ke frame dulu.
        fb = face_box[:4]
        if frame_wh:
            fb = (max(0.0, fb[0]), max(0.0, fb[1]), min(float(frame_wh[0]), fb[2]), min(float(frame_wh[1]), fb[3]))
        if fb[2] > fb[0] and fb[3] > fb[1]:
            heads = [d for d in heads if is_driver_head(d.xyxy, fb, d10)]
            torsos = [d for d in torsos if is_driver_torso(d.xyxy, fb, d10)]
    no_helmet = bool(heads)
    thr = d10.get("vest_iou_max", 0.3)
    in_min = d10.get("vest_inside_min", 0.6)

    def has_vest(t) -> bool:   # kotak rompi biasanya lebih kecil dari torso: IoU saja bisa < thr
        return any(iou(t.xyxy, v.xyxy) >= thr or inside_frac(v.xyxy, t.xyxy) >= in_min for v in vests)

    no_vest = any(not has_vest(t) for t in torsos)
    return CabinDetections(phone_conf, no_helmet, no_vest, dets, phone_box, phone_info)
