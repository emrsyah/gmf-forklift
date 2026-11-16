"""Filter deteksi `phone` YOLO kabin sebelum D-06 (mengurangi alarm palsu D-06 tanpa kehilangan Lembar A 12-14).

Deteksi `phone` hanya dihitung jika masuk akal untuk PENGEMUDI:

1. Geometri relatif terhadap wajah. Satuan u = sisi terpanjang kotak wajah (landmark MediaPipe), sehingga
   aturan tidak bergantung resolusi/jarak kamera. Pusat ponsel harus di wilayah sekitar pengemudi
   (|dx| <= max_dx_u, min_dy_u <= dy <= max_dy_u; dy positif = di bawah pusat wajah): telinga (skenario 13),
   dada/pangkuan saat mengetik (12), setinggi kemudi (14).
2. Ukuran sisi terpanjang kotak dalam [min_size_u, max_size_u] dan rasio sisi <= max_aspect.
3. Bukan di dalam wajah: kotak yang sebagian besar di dalam kotak wajah DAN pusatnya di elips tengah wajah
   (mulut/dagu/janggut) ditolak saat wajah menghadap kamera (|yaw| kecil). Ponsel di telinga pusatnya di tepi
   wajah, jadi tidak kena aturan ini.
4. Stabil: kandidat harus muncul di >= min_hits run YOLO berturut-turut (jeda <= max_gap_s) di tempat yang
   kurang lebih sama (lompatan pusat <= max_jump_u). Deteksi yang berkedip di tempat acak tidak lolos.

Referensi jika wajah tidak terdeteksi (MediaPipe sering kehilangan wajah saat menunduk ke ponsel di
pangkuan = skenario 12; menolak ponsel di sini akan membuat D-05 terkirim = kode salah K4):
  a. kotak wajah terakhir jika umurnya <= face_memory_s (kamera kabin tetap, kursi pengemudi tidak pindah),
  b. kotak kepala YOLO (helmet / head_no_helmet),
  c. kotak torso YOLO (wajah diperkirakan di atas torso),
  d. tanpa referensi: lolos tanpa aturan geometri, tetapi dengan conf lebih tinggi (no_ref_conf).
Tidak pernah menolak hanya karena wajah tidak ada.

Semua angka dari configs/thresholds.yaml -> D06_phone.filter. Filter di-update HANYA saat YOLO benar-benar
jalan (pipeline memakai ulang hasil YOLO di frame yang dilewati).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


@dataclass
class FaceRef:
    cx: float
    cy: float
    u: float                 # px
    source: str              # face | face_memory | head | torso | none
    box: tuple | None = None  # kotak wajah (hanya untuk aturan "di dalam wajah")
    yaw: float | None = None


@dataclass
class PhoneVerdict:
    conf: float = 0.0                       # conf ponsel yang lolos (0 = tidak ada)
    box: tuple | None = None
    ref: str = "none"
    reason: str = ""                        # alasan kandidat terbaik ditolak (debug/overlay)
    n_candidates: int = 0
    info: dict = field(default_factory=dict)


def face_box_from_landmarks(landmarks_px) -> tuple | None:
    if landmarks_px is None or len(landmarks_px) == 0:
        return None
    xy = np.asarray(landmarks_px, dtype=float)
    return float(xy[:, 0].min()), float(xy[:, 1].min()), float(xy[:, 0].max()), float(xy[:, 1].max())


def _ioa(a, b) -> float:
    """Fraksi area a yang berada di dalam b."""
    iw = max(0.0, min(a[2], b[2]) - max(a[0], b[0]))
    ih = max(0.0, min(a[3], b[3]) - max(a[1], b[1]))
    area = max(1e-9, (a[2] - a[0]) * (a[3] - a[1]))
    return iw * ih / area


class PhoneFilter:
    def __init__(self, cfg: dict | None):
        self.c = dict(cfg or {})
        self.enabled = bool(self.c.get("enabled", False))
        self._face_box: tuple | None = None
        self._face_ts: int | None = None
        self._face_yaw: float | None = None
        self._track: tuple | None = None    # (cx, cy) kandidat terakhir yang diterima geometri
        self._track_ts: int | None = None
        self._hits = 0

    def reset(self) -> None:
        self.__init__(self.c)

    # ------------------------------------------------------------ referensi wajah
    def _reference(self, ts_ms: int, face_box, yaw, dets) -> FaceRef | None:
        c = self.c
        if face_box is not None:
            self._face_box, self._face_ts, self._face_yaw = tuple(face_box), ts_ms, yaw
            x1, y1, x2, y2 = face_box
            return FaceRef((x1 + x2) / 2, (y1 + y2) / 2, max(x2 - x1, y2 - y1), "face", tuple(face_box), yaw)
        if self._face_box is not None and ts_ms - self._face_ts <= c.get("face_memory_s", 5.0) * 1000:
            x1, y1, x2, y2 = self._face_box
            # yaw tidak diketahui saat wajah hilang -> aturan "di dalam wajah" tidak dipakai (box=None)
            return FaceRef((x1 + x2) / 2, (y1 + y2) / 2, max(x2 - x1, y2 - y1), "face_memory", None, None)
        ref_conf = c.get("ref_conf", 0.40)
        heads = [d for d in dets if d.cls_name in ("helmet", "head_no_helmet") and d.conf >= ref_conf]
        if heads:
            h = max(heads, key=lambda d: d.conf)
            x1, y1, x2, y2 = h.xyxy
            u = max(x2 - x1, y2 - y1) * c.get("head_box_scale", 0.85)
            return FaceRef((x1 + x2) / 2, (y1 + y2) / 2 + c.get("head_center_shift_u", 0.1) * u, u, "head")
        torsos = [d for d in dets if d.cls_name == "torso" and d.conf >= ref_conf]
        if torsos:
            t = max(torsos, key=lambda d: d.conf)
            x1, y1, x2, y2 = t.xyxy
            u = (x2 - x1) * c.get("torso_width_to_u", 0.55)
            return FaceRef((x1 + x2) / 2, y1 - c.get("torso_face_above_u", 0.7) * u, u, "torso")
        return None

    # ------------------------------------------------------------ aturan geometri
    def _geometry_reason(self, box, ref: FaceRef) -> str:
        c = self.c
        x1, y1, x2, y2 = box
        w, h = max(1e-6, x2 - x1), max(1e-6, y2 - y1)
        u = max(1e-6, ref.u)
        dx = ((x1 + x2) / 2 - ref.cx) / u
        dy = ((y1 + y2) / 2 - ref.cy) / u
        size = max(w, h) / u
        if abs(dx) > c.get("max_dx_u", 2.2):
            return "region_x"
        if dy < c.get("min_dy_u", -0.9) or dy > c.get("max_dy_u", 4.0):
            return "region_y"
        if size < c.get("min_size_u", 0.3):
            return "too_small"
        if size > c.get("max_size_u", 2.2):
            return "too_big"
        if max(w, h) / min(w, h) > c.get("max_aspect", 4.0):
            return "aspect"
        inner = c.get("inner_face") or {}
        if inner and ref.box is not None and (ref.yaw is None or abs(ref.yaw) <= inner.get("max_abs_yaw_deg", 30.0)):
            ex = dx / max(1e-6, inner.get("rx_u", 0.3))
            ey = dy / max(1e-6, inner.get("ry_u", 0.45))
            if ex * ex + ey * ey <= 1.0 and _ioa(box, ref.box) >= inner.get("min_ioa", 0.5):
                return "inside_face"
        return ""

    # ------------------------------------------------------------ update per run YOLO
    def update(self, dets, ts_ms: int, face_box=None, yaw: float | None = None,
               frame_wh: tuple[int, int] | None = None) -> PhoneVerdict:
        c = self.c
        phones = [d for d in dets if d.cls_name == "phone"]
        if not self.enabled:
            best = max(phones, key=lambda d: d.conf, default=None)
            return PhoneVerdict(best.conf if best else 0.0, best.xyxy if best else None, "off", "", len(phones))

        ref = self._reference(ts_ms, face_box, yaw, dets)
        conf_min = c.get("conf", 0.35) if ref is not None else c.get("no_ref_conf", 0.55)
        cands, reasons = [], []
        for d in sorted(phones, key=lambda d: -d.conf):
            if d.conf < conf_min:
                reasons.append("low_conf")
                continue
            r = self._geometry_reason(d.xyxy, ref) if ref is not None else ""
            if r:
                reasons.append(r)
            else:
                cands.append(d)

        # satuan untuk stabilitas: u referensi, atau fraksi lebar frame jika tanpa referensi
        if ref is not None:
            u = ref.u
        else:
            u = (frame_wh[0] if frame_wh else 640) * c.get("no_ref_unit_frac", 0.25)
        tr = c.get("track", {}) or {}
        gap_ms = tr.get("max_gap_s", 0.5) * 1000
        if self._track_ts is not None and ts_ms - self._track_ts > gap_ms:
            self._track, self._hits = None, 0
        chosen = None
        if cands:
            if self._track is None:
                chosen = cands[0]
                self._hits = 1
            else:
                def dist(d):
                    x1, y1, x2, y2 = d.xyxy
                    return float(np.hypot((x1 + x2) / 2 - self._track[0], (y1 + y2) / 2 - self._track[1]))
                near = [d for d in cands if dist(d) <= tr.get("max_jump_u", 0.8) * u]
                if near:
                    chosen = max(near, key=lambda d: d.conf)
                    self._hits += 1
                else:
                    chosen = cands[0]            # lompat jauh -> mulai jejak baru
                    self._hits = 1
            x1, y1, x2, y2 = chosen.xyxy
            self._track, self._track_ts = ((x1 + x2) / 2, (y1 + y2) / 2), ts_ms

        stable = chosen is not None and self._hits >= int(tr.get("min_hits", 2))
        v = PhoneVerdict(ref=ref.source if ref else "none", n_candidates=len(phones))
        if stable:
            v.conf, v.box = float(chosen.conf), tuple(chosen.xyxy)
        elif chosen is not None:
            v.reason = f"unstable({self._hits})"
        elif reasons:
            v.reason = reasons[0]
        v.info = {"phone_ref": v.ref, "phone_hits": self._hits, "phone_reject": v.reason}
        return v
