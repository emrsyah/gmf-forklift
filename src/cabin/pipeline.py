"""Pipeline kabin: frame -> sinyal -> indikator -> keputusan. Dipakai SAMA oleh live dan replay."""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from src.capture import Frame
from src.decision import Action, DecisionEngine
from src.preprocess import crop_roi, simulate_nir

from .calibration import Calibration
from .camblock import PartialBlock, frame_stats, is_blocked
from .face import FaceObs
from .indicators import IndicatorInput, build_indicators
from .signals import FaceSignals, SignalExtractor
from .phone_crop import PhoneCrop
from .phone_filter import PhoneFilter, face_box_from_landmarks
from .yolo_cabin import CabinDetections, interpret


@dataclass
class CabinState:
    frame: Frame
    image: np.ndarray                  # gambar yang dipakai model (setelah simulate_nir)
    obs: FaceObs
    sig: FaceSignals
    cam_stats: dict
    blocked_frame: bool
    yolo: CabinDetections | None
    active: dict                       # {code: {...}} indikator aktif
    top_code: str | None
    actions: list[Action] = field(default_factory=list)
    timings_ms: dict = field(default_factory=dict)


class CabinPipeline:
    def __init__(self, runtime: dict, thresholds: dict, events: dict,
                 face_model=None, detector=None, use_yolo: bool | None = None):
        cab = runtime["cabin"]
        self.rt = runtime
        self.th = thresholds
        self.cab = cab
        self.simulate_nir = bool(cab.get("simulate_nir", False))
        if face_model is None:
            from .face import make_face_model
            face_model = make_face_model(cab["face"])
        self.face = face_model

        self.yolo_every = int(cab.get("yolo", {}).get("every_n_frames", 2))
        want_yolo = cab.get("yolo", {}).get("enabled", True) if use_yolo is None else use_yolo
        self.detector = detector
        self.yolo_status = "off"
        if want_yolo and self.detector is None:
            try:
                from src.backends import make_detector
                self.detector = make_detector(cab["yolo"])
                self.yolo_status = "on"
            except FileNotFoundError as e:
                self.yolo_status = f"off (bobot tidak ada: {cab['yolo'].get('weights')})"
                print(f"[cabin] YOLO kabin nonaktif -> D-06/D-10 tidak jalan. {e}")
        elif self.detector is not None:
            self.yolo_status = "on"

        self.calibration = Calibration(thresholds.get("calibration", {}))
        self.signals = SignalExtractor(thresholds.get("signals", {}), self.calibration)
        self.indicators = build_indicators(thresholds)
        self.decision = DecisionEngine(events, runtime.get("device_id", "laptop-dev"),
                                       runtime.get("event_id_prefix", "laptop"),
                                       runtime.get("model_versions", {}),
                                       runtime.get("timezone_offset_h", 7.0))
        self.require_moving = bool(thresholds.get("vehicle", {}).get("require_moving", False))
        self._last_yolo: CabinDetections | None = None
        # filter ponsel (D-06) relatif wajah pengemudi; di-update hanya saat YOLO benar-benar jalan
        self.phone_filter = PhoneFilter(thresholds.get("D06_phone", {}).get("filter"))
        self.phone_crop = PhoneCrop(thresholds.get("D06_phone", {}).get("crop"))
        d09 = thresholds.get("D09_camera_block", {})
        self.partial_block = PartialBlock(d09.get("partial"), d09.get("mean_min", 15), d09.get("mean_max", 240))
        self._n = 0
        self._t_first_ms: int | None = None
        self._last_face_ms: int | None = None
        self.auto_recalibrations = 0

    def process(self, frame: Frame, moving: bool = True) -> CabinState:
        t0 = time.perf_counter()
        tm = {}
        raw = frame.image
        # D-09 dihitung dari frame mentah (sebelum preprocessing)
        stats = frame_stats(raw, self.th.get("D09_camera_block", {}).get("downscale_px", 160))
        blocked = is_blocked(stats, self.th.get("D09_camera_block", {}))

        img = simulate_nir(raw, self.cab.get("clahe_clip", 2.0), self.cab.get("clahe_tile", 8)) \
            if self.simulate_nir else raw
        roi_img, off = crop_roi(img, self.cab.get("roi"))

        t1 = time.perf_counter()
        obs = self.face.process(roi_img, frame.ts_ms, off)
        tm["face"] = (time.perf_counter() - t1) * 1000
        self._auto_recalibrate(obs.present, frame.ts_ms)
        sig = self.signals.update(obs, frame.ts_ms)
        # D-09 sebagian: perlu status wajah (baseline sel hanya diperbarui saat wajah terlihat)
        if self.partial_block.update(raw, obs.present):
            blocked = True
        stats["partial_frac"] = round(self.partial_block.frac, 2)     # untuk overlay/log debug

        yolo = None
        if self.detector is not None:
            if self._n % self.yolo_every == 0 or self._last_yolo is None:
                t2 = time.perf_counter()
                yolo_in = roi_img
                if self.cab.get("yolo", {}).get("preprocess", "nir") == "nir" and not self.simulate_nir:
                    # sama dengan training (Tech Spec bag. 5): grayscale 3 channel + CLAHE
                    yolo_in = simulate_nir(roi_img, self.cab.get("clahe_clip", 2.0), self.cab.get("clahe_tile", 8))
                det_conf = min(self.th.get("D06_phone", {}).get("conf", 0.35),
                               self.th.get("D10_ppe", {}).get("conf", 0.40))
                dets = self.detector.predict(yolo_in, conf=det_conf)
                face_box = face_box_from_landmarks(obs.landmarks_px) if obs.present else None
                if self.phone_crop.enabled:
                    # pass kedua: ponsel di potongan area pengemudi (koordinat ROI, sama dengan `dets`)
                    fb_roi = None if face_box is None else (face_box[0] - off[0], face_box[1] - off[1],
                                                            face_box[2] - off[0], face_box[3] - off[1])
                    t3 = time.perf_counter()
                    dets = self.phone_crop.run(self.detector, yolo_in, dets, fb_roi, frame.ts_ms, det_conf)
                    if self.phone_crop.last_box is not None:
                        tm["crop"] = (time.perf_counter() - t3) * 1000
                if off != (0, 0):
                    for d in dets:
                        x1, y1, x2, y2 = d.xyxy
                        d.xyxy = (x1 + off[0], y1 + off[1], x2 + off[0], y2 + off[1])
                self._last_yolo = interpret(
                    dets, self.th.get("D06_phone", {}), self.th.get("D10_ppe", {}),
                    phone_filter=self.phone_filter, ts_ms=frame.ts_ms,
                    face_box=face_box,
                    yaw=obs.yaw_raw if obs.present else None, frame_wh=(img.shape[1], img.shape[0]))
                tm["yolo"] = (time.perf_counter() - t2) * 1000
            yolo = self._last_yolo
        self._n += 1

        moving_ok = moving if self.require_moving else True
        x = IndicatorInput(
            ts_ms=frame.ts_ms, sig=sig,
            phone_conf=yolo.phone_conf if yolo else 0.0,
            no_helmet=yolo.no_helmet if yolo else False,
            no_vest=yolo.no_vest if yolo else False,
            yolo_valid=yolo is not None,
            blocked_frame=blocked, moving=moving_ok,
        )
        if self._t_first_ms is None:
            self._t_first_ms = frame.ts_ms
        # beberapa detik pertama kamera (auto-exposure, frame gelap) tidak boleh memicu alert (mis. D-09)
        warm = frame.ts_ms - self._t_first_ms < self.cab.get("startup_grace_s", 0.0) * 1000
        active = {}
        for code, ind in self.indicators.items():
            if ind.update(x) and not warm:
                active[code] = {"confidence": ind.confidence,
                                "condition_start_ms": ind.condition_start_ms}
        extra = {"t_video_s": round(frame.video_t_s, 3)}
        actions = self.decision.step(frame.ts_ms, active, extra)
        top = min(active, key=self.decision.rank) if active else None
        tm["total"] = (time.perf_counter() - t0) * 1000
        return CabinState(frame, img, obs, sig, stats, blocked, yolo, active, top, actions, tm)

    def _auto_recalibrate(self, face_present: bool, ts_ms: int):
        """Wajah hilang lama lalu muncul lagi (ganti pengemudi / kembali ke kursi) -> kalibrasi ulang."""
        after = self.th.get("calibration", {}).get("auto_recalibrate_after_absent_s")
        if not face_present:
            return
        if (after is not None and self._last_face_ms is not None
                and ts_ms - self._last_face_ms >= float(after) * 1000 and not self.calibration.active):
            self.calibration.start()
            self.auto_recalibrations += 1
        self._last_face_ms = ts_ms

    def finish(self, ts_ms: int, video_t_s: float = 0.0) -> list[Action]:
        return self.decision.flush(ts_ms, {"t_video_s": round(video_t_s, 3)})

    def recalibrate(self):
        self.calibration.start()
