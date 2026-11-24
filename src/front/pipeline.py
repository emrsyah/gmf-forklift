"""Pipeline kamera depan: Frame -> deteksi+track -> jarak -> kecepatan -> kandidat F01..F03.

    pipe = FrontPipeline.from_config(Config.load())
    for frame in make_source(...):
        res = pipe.process(frame)          # FrontResult
        vis = pipe.draw(frame.image, res)

Semua waktu memakai `frame.ts_ms` (timestamp sumber) sehingga replay = live (aturan kerja 3).
Kandidat F-xx adalah USULAN; modul keputusan yang memutuskan (events.yaml front_events_enabled).
"""
from __future__ import annotations

from dataclasses import dataclass, field

import cv2
import numpy as np

from src.capture import Frame
from src.config import Config, load_yaml, resolve_path

from .distance import DistanceEstimator
from .motion import MotionTracker
from .zones import FrontCandidate, ZoneEvaluator

# Gaya overlay (bukan threshold/geometri).
_GREEN, _ORANGE, _RED, _GREY, _WHITE = (60, 200, 60), (0, 165, 255), (0, 0, 255), (160, 160, 160), (255, 255, 255)
_FONT = cv2.FONT_HERSHEY_SIMPLEX
_ARROW_PX_PER_MPS, _ARROW_MAX_PX = 30, 60


@dataclass
class FrontTrack:
    track_id: int | None
    cls: str
    conf: float
    box: tuple[float, float, float, float]       # xyxy piksel
    distance_m: float | None
    lateral_m: float | None
    distance_method: str | None
    speed_mps: float | None                      # < 0 = mendekat
    approaching: bool


def track_styles(res: "FrontResult") -> list[tuple["FrontTrack", tuple[int, int, int], str]]:
    """(track, warna BGR, label) per track: dipakai overlay OpenCV dan kotak yang digambar dashboard."""
    cand_by_tid = {c.track_id: c for c in res.candidates}
    out = []
    for t in res.tracks:
        c = cand_by_tid.get(t.track_id) if t.track_id is not None else None
        col = _RED if c and c.level == "A-03" else _ORANGE if c else _GREEN if t.distance_m else _GREY
        tid = "-" if t.track_id is None else str(t.track_id)
        label = f"#{tid} {t.cls}"
        if t.distance_m is not None:
            label += f" {t.distance_m:.1f}m"
            if t.distance_method and t.distance_method != "foot":
                label += f"({t.distance_method[0]})"
        if t.speed_mps is not None:
            label += f" {t.speed_mps:+.1f}m/s"
        if c:
            label += f" {c.code}?"
        out.append((t, col, label))
    return out


def dets_message(res: "FrontResult", w: int, h: int, rot: int = 0) -> dict:
    """Kotak deteksi untuk digambar di atas video kamera asli (dashboard), koordinat 0..1 gambar terputar."""
    boxes = []
    for t, (b, g, r), label in track_styles(res):
        x1, y1, x2, y2 = t.box
        boxes.append({"b": [round(min(max(v, 0.0), 1.0), 4) for v in (x1 / w, y1 / h, x2 / w, y2 / h)],
                      "c": f"#{r:02x}{g:02x}{b:02x}", "l": label})
    return {"t": int(res.ts_ms), "rot": int(rot) % 360, "boxes": boxes}


@dataclass
class FrontResult:
    ts_ms: int
    frame_index: int
    tracks: list[FrontTrack] = field(default_factory=list)
    candidates: list[FrontCandidate] = field(default_factory=list)
    active_codes: list[str] = field(default_factory=list)   # urut prioritas, kandidat saja

    def to_dict(self) -> dict:
        return {
            "ts_ms": self.ts_ms,
            "frame_index": self.frame_index,
            "tracks": [t.__dict__ for t in self.tracks],
            "candidates": [c.__dict__ for c in self.candidates],
            "active_codes": list(self.active_codes),
        }


class FrontPipeline:
    def __init__(self, detector, distance: DistanceEstimator, motion: MotionTracker,
                 zones: ZoneEvaluator, calibrated: bool = True):
        self.detector = detector
        self.distance = distance
        self.motion = motion
        self.zones = zones
        self.calibrated = calibrated

    @classmethod
    def from_config(cls, cfg: Config, cam_cfg: dict | None = None, detector=None) -> "FrontPipeline":
        front_rt = cfg.runtime.get("front", {})
        thr = cfg.thresholds["front"]
        if cam_cfg is None:
            cam_cfg = load_yaml(resolve_path(front_rt.get("calibration", "configs/front_camera.yaml")))
        yolo_cfg = front_rt.get("yolo", {})
        person_class = yolo_cfg.get("person_class", "person")
        if detector is None:
            from .detector import FrontDetector
            detector = FrontDetector(yolo_cfg, cam_cfg, thr["person_conf"])
        return cls(detector,
                   DistanceEstimator.from_config(cam_cfg),
                   MotionTracker.from_config(thr, cam_cfg),
                   ZoneEvaluator.from_config(thr, cam_cfg, cfg.events, person_class),
                   calibrated=bool(cam_cfg.get("camera", {}).get("calibrated", False)))

    def reset(self) -> None:
        self.motion.reset()
        if hasattr(self.detector, "reset"):
            self.detector.reset()

    def set_camera(self, cam_cfg: dict, camera_block: dict) -> None:
        """Ganti intrinsik (blok `camera`) tanpa mengubah mount/distance. Dipakai simulasi profil kamera:
        frame sudah di-crop/resize, jadi K kalibrasi asli tidak berlaku (K_for hanya benar untuk resize penuh)."""
        self.distance = DistanceEstimator.from_config({**cam_cfg, "camera": camera_block})
        self.calibrated = bool(camera_block.get("calibrated", False))

    def process(self, frame: Frame) -> FrontResult:
        img = frame.image
        h, w = img.shape[:2]
        tracks: list[FrontTrack] = []
        for d in self.detector.track(img):
            est = self.distance.estimate(d.xyxy, d.cls_name, w, h)
            dist_m = est.distance_m if est else None
            speed, approaching = None, False
            if d.track_id is not None:
                st = self.motion.update(d.track_id, frame.ts_ms, dist_m, est.method if est else None)
                speed, approaching = st.speed_mps, st.approaching
            tracks.append(FrontTrack(d.track_id, d.cls_name, d.conf, tuple(d.xyxy), dist_m,
                                     est.lateral_m if est else None, est.method if est else None,
                                     speed, approaching))
        self.motion.prune(frame.ts_ms)
        cands = self.zones.evaluate(tracks)
        return FrontResult(frame.ts_ms, frame.index, tracks, cands, self.zones.active_codes(cands))

    # --- overlay debug ----------------------------------------------------------------------------
    def draw(self, image: np.ndarray, res: FrontResult, fps: float | None = None) -> np.ndarray:
        vis = image.copy()
        h, w = vis.shape[:2]
        # garis zona (perkiraan tanpa distorsi, di sumbu tengah)
        for d_m, col in ((self.zones.warning_zone_m, _ORANGE), (self.zones.danger_zone_m, _RED)):
            v = self.distance.row_for_distance(d_m, w, h)
            if v is not None and 0 <= v < h:
                y = int(round(v))
                cv2.line(vis, (0, y), (w - 1, y), col, 1, cv2.LINE_AA)
                cv2.putText(vis, f"{d_m:g} m", (5, y - 4), _FONT, 0.5, col, 1, cv2.LINE_AA)

        for t, col, label in track_styles(res):
            x1, y1, x2, y2 = (int(round(c)) for c in t.box)
            cv2.rectangle(vis, (x1, y1), (x2, y2), col, 2)
            (tw, th), _ = cv2.getTextSize(label, _FONT, 0.5, 1)
            ty = max(th + 4, y1)
            cv2.rectangle(vis, (x1, ty - th - 4), (x1 + tw + 4, ty), col, -1)
            cv2.putText(vis, label, (x1 + 2, ty - 3), _FONT, 0.5, (0, 0, 0), 1, cv2.LINE_AA)
            # panah arah: ke bawah = mendekat, ke atas = menjauh (panjang ~ kecepatan)
            if t.speed_mps is not None:
                fx, fy = (x1 + x2) // 2, y2
                L = int(np.clip(-t.speed_mps * _ARROW_PX_PER_MPS, -_ARROW_MAX_PX, _ARROW_MAX_PX))
                if L != 0:
                    cv2.arrowedLine(vis, (fx, fy), (fx, fy + L), _RED if t.approaching else _WHITE, 2,
                                    tipLength=0.3)

        hud = [f"tracks {len(res.tracks)}"]
        if fps is not None:
            hud.insert(0, f"{fps:.1f} FPS")
        if res.active_codes:
            hud.append("CANDIDATE (usulan): " + ",".join(res.active_codes))
        if not self.calibrated:
            hud.append("KAMERA BELUM DIKALIBRASI - jarak kasar")
        for i, s in enumerate(hud):
            cv2.putText(vis, s, (10, 22 + 22 * i), _FONT, 0.6, (0, 0, 0), 3, cv2.LINE_AA)
            cv2.putText(vis, s, (10, 22 + 22 * i), _FONT, 0.6, _WHITE, 1, cv2.LINE_AA)
        return vis
