"""Pass kedua YOLO untuk ponsel di potongan area pengemudi (wajah sampai pangkuan).

Di frame penuh, ponsel hanya beberapa puluh piksel setelah diperkecil ke input YOLO (640) -> recall rendah.
Memotong area tempat ponsel mungkin berada (wilayah yang sama dengan filter ponsel, satuan u = sisi terpanjang
kotak wajah) membuat ponsel 1,5-3x lebih besar bagi model. Hanya kelas `classes` (default phone) yang diambil
dari potongan; helm/rompi/badan tetap dari frame penuh. Semua angka dari configs/thresholds.yaml
D06_phone.crop.
"""
from __future__ import annotations

from dataclasses import replace

import numpy as np

from src.backends import Detection

from .yolo_cabin import iou


class PhoneCrop:
    def __init__(self, cfg: dict | None):
        c = cfg or {}
        self.enabled = bool(c.get("enabled", False))
        self.left_u = float(c.get("left_u", 2.6))
        self.right_u = float(c.get("right_u", 2.6))
        self.up_u = float(c.get("up_u", 1.2))
        self.down_u = float(c.get("down_u", 4.3))
        self.min_gain = float(c.get("min_gain", 1.25))     # sisi frame / sisi potongan minimal agar berguna
        self.min_side_px = int(c.get("min_side_px", 160))
        self.memory_ms = float(c.get("face_memory_s", 5.0)) * 1000
        self.merge_iou = float(c.get("merge_iou", 0.5))
        self.classes = set(c.get("classes", ["phone"]))
        self._face = None
        self._face_ts = None
        self.last_box: tuple[int, int, int, int] | None = None   # untuk overlay debug

    def box(self, face_box, frame_wh: tuple[int, int], ts_ms: int) -> tuple[int, int, int, int] | None:
        """Kotak potongan (x1, y1, x2, y2) piksel, atau None jika tidak ada wajah / tidak menguntungkan."""
        if face_box is not None:
            self._face, self._face_ts = face_box, ts_ms
        elif self._face is None or self._face_ts is None or ts_ms - self._face_ts > self.memory_ms:
            return None
        fx1, fy1, fx2, fy2 = self._face
        u = max(fx2 - fx1, fy2 - fy1)
        if u <= 0:
            return None
        cx, cy = (fx1 + fx2) / 2, (fy1 + fy2) / 2
        w, h = frame_wh
        x1 = int(max(0, cx - self.left_u * u))
        x2 = int(min(w, cx + self.right_u * u))
        y1 = int(max(0, cy - self.up_u * u))
        y2 = int(min(h, cy + self.down_u * u))
        cw, ch = x2 - x1, y2 - y1
        if min(cw, ch) < self.min_side_px:
            return None
        # potongan hanya berguna jika model "melihat" ponsel lebih besar: letterbox memakai sisi terpanjang
        gain = max(w, h) / max(cw, ch)
        if gain < self.min_gain:
            return None
        return x1, y1, x2, y2

    def detect(self, detector, image: np.ndarray, crop: tuple[int, int, int, int], conf: float) -> list[Detection]:
        x1, y1, x2, y2 = crop
        dets = detector.predict(np.ascontiguousarray(image[y1:y2, x1:x2]), conf=conf)
        out = []
        for d in dets:
            if d.cls_name not in self.classes:
                continue
            a, b, c, e = d.xyxy
            out.append(replace(d, xyxy=(a + x1, b + y1, c + x1, e + y1)))
        return out

    def merge(self, full: list[Detection], extra: list[Detection]) -> list[Detection]:
        """Gabung deteksi potongan ke deteksi frame penuh: kotak kelas sama dengan IoU >= merge_iou -> simpan conf
        tertinggi (koordinat dari deteksi ber-conf tertinggi)."""
        out = list(full)
        for e in extra:
            dup = [i for i, f in enumerate(out) if f.cls_name == e.cls_name and iou(f.xyxy, e.xyxy) >= self.merge_iou]
            if not dup:
                out.append(e)
                continue
            i = max(dup, key=lambda k: out[k].conf)
            if e.conf > out[i].conf:
                out[i] = e
        return out

    def run(self, detector, image: np.ndarray, full: list[Detection], face_box, ts_ms: int,
            conf: float) -> list[Detection]:
        self.last_box = None
        if not self.enabled:
            return full
        crop = self.box(face_box, (image.shape[1], image.shape[0]), ts_ms)
        if crop is None:
            return full
        self.last_box = crop
        return self.merge(full, self.detect(detector, image, crop, conf))
