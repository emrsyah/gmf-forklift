"""Pipeline kabin: frame -> sinyal -> indikator -> keputusan. Dipakai SAMA oleh live dan replay."""
from __future__ import annotations

import time
from dataclasses import dataclass, field

import numpy as np

from src.capture import Frame
from src.decision import Action, DecisionEngine
from src.preprocess import crop_roi, gray3, simulate_nir

from .calibration import Calibration
from .camblock import PartialBlock, frame_stats, is_blocked
from .face import FaceObs
from .indicators import IndicatorInput, build_indicators
from .signals import FaceSignals, SignalExtractor
from .phone_crop import PhoneCrop
from .phone_filter import PhoneFilter, face_box_from_landmarks
from .eye_cls import EyeClassifier
from .ppe_cls import PpeClassifier
from .wrist_crop import WristCrop
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
        # ReverseState dari kamera depan (main.py, thread lain); None = tidak ada -> tidak pernah "mundur"
        self.reverse_state = None
        self.reverse_grace_ms = float(thresholds.get("D05_distraction", {}).get("reverse_grace_s", 2.0)) * 1000
        self._last_reverse_wall_ms: int | None = None
        self.reversing = False
        self._last_yolo: CabinDetections | None = None
        # filter ponsel (D-06) relatif wajah pengemudi; di-update hanya saat YOLO benar-benar jalan
        self.phone_filter = PhoneFilter(thresholds.get("D06_phone", {}).get("filter"))
        self.phone_crop = PhoneCrop(thresholds.get("D06_phone", {}).get("crop"))
        # A7: pose -> potongan ponsel di sekitar tangan (model di runtime cabin.pose, ambang di D06_phone.wrist_crop)
        self.wrist_crop = WristCrop(thresholds.get("D06_phone", {}).get("wrist_crop"))
        self.pose = None
        if self.wrist_crop.enabled and self.detector is not None:
            try:
                from src.backends.pose import make_pose
                self.pose = make_pose(cab.get("pose", {"weights": "models/weights/yolo11n-pose.onnx"}))
            except FileNotFoundError as e:
                print(f"[cabin] pose nonaktif -> potongan tangan D-06 tidak jalan. {e}")
        # A8: classifier helm/rompi di potongan kepala/dada dari kotak wajah (D10_ppe.classifier)
        # A9: classifier mata untuk frame cadangan pose (signals.eye_classifier)
        self.ppe_cls = self.eye_cls = None
        for attr, cls_, cfg_, what in (
                ("ppe_cls", PpeClassifier, thresholds.get("D10_ppe", {}).get("classifier"), "APD D-10"),
                ("eye_cls", EyeClassifier, thresholds.get("signals", {}).get("eye_classifier"), "mata A9")):
            try:
                m = cls_(cfg_)
                setattr(self, attr, m if m.enabled else None)
            except FileNotFoundError as e:
                print(f"[cabin] classifier {what} nonaktif. {e}")
        d09 =thresholds.get("D09_camera_block", {})
        self.partial_block = PartialBlock(d09.get("partial"), d09.get("mean_min", 15), d09.get("mean_max", 240))
        # cabin.yolo.parallel: YOLO di thread sendiri, paralel dengan MediaPipe (satu thread -> urutan tetap)
        # cabin.yolo.async: frame TIDAK menunggu YOLO; hasil terbaru dipakai sampai hasil baru selesai.
        #   Hanya untuk live (hasil bergantung waktu); replay `fast` tetap deterministik dengan async: false.
        #   async_max_fps: batas laju YOLO (0 = secepat GPU) supaya CPU (NMS, CLAHE) tidak direbut dari MediaPipe.
        ycfg = cab.get("yolo", {})
        self.yolo_async = bool(ycfg.get("async", False))
        self.yolo_async_min_dt_ms = 1000.0 / ycfg["async_max_fps"] if ycfg.get("async_max_fps") else 0.0
        self._yolo_pending = None           # Future yang sedang jalan (mode async)
        self._yolo_last_submit_ms: int | None = None
        self._yolo_pool = None
        if self.detector is not None and (ycfg.get("parallel", True) or self.yolo_async):
            from concurrent.futures import ThreadPoolExecutor
            self._yolo_pool = ThreadPoolExecutor(max_workers=1, thread_name_prefix="yolo")
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

        if self.simulate_nir:
            img = simulate_nir(raw, self.cab.get("clahe_clip", 2.0), self.cab.get("clahe_tile", 8))
        else:
            img = gray3(raw) if raw.ndim == 2 else raw    # sumber GRAY8: model butuh 3 channel
        roi_img, off = crop_roi(img, self.cab.get("roi"))

        # YOLO (GPU) tidak bergantung pada hasil wajah -> dikirim ke thread pekerja SEBELUM MediaPipe (CPU),
        # hasilnya diambil sesudahnya. Input sama persis dengan mode berurutan -> deteksi identik.
        yolo_job = None
        d10 = self.th.get("D10_ppe", {})
        det_conf = min(self.th.get("D06_phone", {}).get("conf", 0.35), d10.get("conf", 0.40),
                       d10.get("vest_conf", d10.get("conf", 0.40)))
        if (self.detector is not None and not self.yolo_async
                and (self._n % self.yolo_every == 0 or self._last_yolo is None)):
            yolo_in = self._yolo_input(roi_img)
            if self._yolo_pool is not None:
                fut = self._yolo_pool.submit(self._predict_timed, yolo_in, det_conf)
            else:
                fut = self._predict_timed(yolo_in, det_conf)
            yolo_job = (yolo_in, det_conf, fut)

        t1 = time.perf_counter()
        obs = self.face.process(roi_img, frame.ts_ms, off)
        if self.eye_cls is not None and obs.present and obs.pose_only and obs.eye_pts is not None:
            obs.eye_closed_p = self.eye_cls.p_closed(roi_img, obs.eye_pts)   # titik mata dalam koordinat ROI
        tm["face"] = (time.perf_counter() - t1) * 1000
        self._auto_recalibrate(obs.present, frame.ts_ms)
        sig = self.signals.update(obs, frame.ts_ms)
        # D-09 sebagian: perlu status wajah (baseline sel hanya diperbarui saat wajah terlihat)
        if self.partial_block.update(raw, obs.present):
            blocked = True
        stats["partial_frac"] = round(self.partial_block.frac, 2)     # untuk overlay/log debug

        yolo = None
        if self.detector is not None:
            face_box = face_box_from_landmarks(obs.landmarks_px) if obs.present else None
            yaw = obs.yaw_raw if obs.present else None
            wh = (img.shape[1], img.shape[0])
            if self.yolo_async:
                self._yolo_async_step(roi_img, det_conf, off, face_box, yaw, wh, frame.ts_ms, tm)
            elif yolo_job is not None:
                t2 = time.perf_counter()
                yolo_in, det_conf, fut = yolo_job
                dets, tm["yolo_gpu"] = fut.result() if hasattr(fut, "result") else fut
                tm["yolo_wait"] = (time.perf_counter() - t2) * 1000
                self._last_yolo = self._yolo_post(dets, yolo_in, det_conf, off, face_box, yaw, wh,
                                                  frame.ts_ms, tm)
                tm["yolo"] = tm["yolo_gpu"] + (time.perf_counter() - t2) * 1000 - tm["yolo_wait"]
            yolo = self._last_yolo
        self._n += 1

        moving_ok = moving if self.require_moving else True
        self.reversing = False
        if self.reverse_state is not None:
            now = int(time.time() * 1000)
            if self.reverse_state.reversing_now():
                self._last_reverse_wall_ms = now
            self.reversing = (self._last_reverse_wall_ms is not None
                              and now - self._last_reverse_wall_ms <= self.reverse_grace_ms)
        x = IndicatorInput(
            ts_ms=frame.ts_ms, sig=sig,
            phone_conf=yolo.phone_conf if yolo else 0.0,
            no_helmet=yolo.no_helmet if yolo else False,
            no_vest=yolo.no_vest if yolo else False,
            yolo_valid=yolo is not None,
            blocked_frame=blocked, moving=moving_ok, reversing=self.reversing,
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

    def _yolo_input(self, roi_img: np.ndarray) -> np.ndarray:
        if self.cab.get("yolo", {}).get("preprocess", "nir") == "nir" and not self.simulate_nir:
            # sama dengan training (Tech Spec bag. 5): grayscale 3 channel + CLAHE
            return simulate_nir(roi_img, self.cab.get("clahe_clip", 2.0), self.cab.get("clahe_tile", 8))
        return roi_img

    def _yolo_post(self, dets, yolo_in, det_conf, off, face_box, yaw, wh, ts_ms, tm) -> CabinDetections:
        """Setelah prediksi: pass ponsel di crop pengemudi, offset ROI -> frame, interpretasi D-06/D-10."""
        fb_roi = None if face_box is None else (face_box[0] - off[0], face_box[1] - off[1],
                                                face_box[2] - off[0], face_box[3] - off[1])
        if self.pose is not None:
            # A7: potongan ponsel di sekitar pergelangan tangan pengemudi (koordinat ROI)
            t4 = time.perf_counter()
            dets = self.wrist_crop.run(self.detector, self.pose, yolo_in, dets, fb_roi, det_conf)
            tm["wrist"] = (time.perf_counter() - t4) * 1000
        if self.phone_crop.enabled:
            # pass kedua: ponsel di potongan area pengemudi (koordinat ROI, sama dengan `dets`)
            t3 = time.perf_counter()
            dets = self.phone_crop.run(self.detector, yolo_in, dets, fb_roi, ts_ms, det_conf)
            if self.phone_crop.last_box is not None:
                tm["crop"] = (time.perf_counter() - t3) * 1000
        if off != (0, 0):
            for d in dets:
                x1, y1, x2, y2 = d.xyxy
                d.xyxy = (x1 + off[0], y1 + off[1], x2 + off[0], y2 + off[1])
        res = interpret(dets, self.th.get("D06_phone", {}), self.th.get("D10_ppe", {}),
                        phone_filter=self.phone_filter, ts_ms=ts_ms, face_box=face_box, yaw=yaw, frame_wh=wh)
        if self.ppe_cls is not None:
            t5 = time.perf_counter()
            c = self.ppe_cls.classify(yolo_in, fb_roi)          # gambar input YOLO (ROI, praproses sama)
            res.no_helmet, res.no_vest = self.ppe_cls.combine(res.no_helmet, res.no_vest, c)
            res.ppe_info = {k: (round(v, 2) if isinstance(v, float) else v) for k, v in c.items() if k.startswith("p_")}
            tm["ppe_cls"] = (time.perf_counter() - t5) * 1000
        return res

    def _yolo_job(self, roi_img, det_conf, off, face_box, yaw, wh, ts_ms):
        """Satu putaran YOLO lengkap di thread pekerja (mode async). Satu pekerja -> PhoneFilter tetap berurutan."""
        tm = {}
        yolo_in = self._yolo_input(roi_img)
        dets, tm["yolo_gpu"] = self._predict_timed(yolo_in, det_conf)
        t = time.perf_counter()
        res = self._yolo_post(dets, yolo_in, det_conf, off, face_box, yaw, wh, ts_ms, tm)
        tm["yolo"] = tm["yolo_gpu"] + (time.perf_counter() - t) * 1000
        return res, tm

    def _yolo_async_step(self, roi_img, det_conf, off, face_box, yaw, wh, ts_ms, tm):
        """Ambil hasil yang sudah selesai (tanpa menunggu), lalu kirim frame ini jika pekerja kosong."""
        fut = self._yolo_pending
        if fut is not None:
            if not fut.done():
                return
            self._yolo_pending = None
            try:
                res, ytm = fut.result()
            except Exception as e:  # noqa: BLE001 - error YOLO tidak boleh mematikan D-02/D-09 dkk.
                # hasil lama dibuang (yolo_valid=False -> D-06/D-10 tidak aktif dari data basi); job berikut dicoba lagi
                self._last_yolo = None
                self.yolo_errors = getattr(self, "yolo_errors", 0) + 1
                self.yolo_status = f"error ({self.yolo_errors}x): {type(e).__name__}: {e}"
                if self.yolo_errors <= 3 or self.yolo_errors % 100 == 0:
                    print(f"[cabin] YOLO gagal: {self.yolo_status}", flush=True)
                res, ytm = None, {}
            else:
                self._last_yolo = res
                if self.yolo_status.startswith("error"):
                    self.yolo_status = "on"
            tm.update(ytm)
        due = (self._yolo_last_submit_ms is None
               or ts_ms - self._yolo_last_submit_ms >= self.yolo_async_min_dt_ms)
        if due:
            self._yolo_last_submit_ms = ts_ms
            # salinan: frame berikutnya bisa menimpa buffer kamera saat pekerja masih membaca
            self._yolo_pending = self._yolo_pool.submit(self._yolo_job, roi_img.copy(), det_conf, off,
                                                        face_box, yaw, wh, ts_ms)

    def _predict_timed(self, image: np.ndarray, conf: float):
        t = time.perf_counter()
        # helm berskor rendah dipakai untuk veto `head_no_helmet` (D10_ppe.helmet_veto_conf); kelas lain tetap >= conf
        veto = self.th.get("D10_ppe", {}).get("helmet_veto_conf")
        if veto is not None and veto < conf:
            dets = [d for d in self.detector.predict(image, conf=veto)
                    if d.conf >= conf or d.cls_name == "helmet"]
        else:
            dets = self.detector.predict(image, conf=conf)
        return dets, (time.perf_counter() - t) * 1000

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
