"""Sinyal per frame dari observasi wajah (Tech Spec bagian 3).

Interface `FaceSignals` TETAP (aturan kerja 4): indikator hanya membaca field di sini,
apa pun model wajahnya.
"""
from __future__ import annotations

from collections import deque
from dataclasses import dataclass

import numpy as np

from .face import FaceObs

# Indeks landmark mata MediaPipe (Tech Spec bag. 3): p1..p6 untuk EAR
LEFT_EYE = [33, 160, 158, 133, 153, 144]
RIGHT_EYE = [362, 385, 387, 263, 373, 380]
# Mulut (fallback tanpa blendshape): bibir dalam atas/bawah + sudut mulut
MOUTH_INNER_TOP, MOUTH_INNER_BOTTOM = 13, 14
MOUTH_LEFT, MOUTH_RIGHT = 61, 291


@dataclass
class FaceSignals:
    ts_ms: int
    face_present: bool
    ear_l: float = 0.0
    ear_r: float = 0.0
    ear: float = 0.0
    openness: float = 1.0         # mean EAR / ear_baseline (0 = tertutup, ~1 = terbuka normal)
    blink: float = 0.0            # max(eyeBlinkLeft, eyeBlinkRight)
    eye_closed: bool = False
    # mata tertutup TANPA hysteresis (ambang masuk saja). None = sama dengan eye_closed.
    # D-01 (durasi kedip / PERCLOS) memakai ini supaya kedip normal tidak memanjang jadi "kedip lambat".
    eye_closed_strict: bool | None = None
    eye_valid: bool = False
    yaw: float = 0.0              # derajat, relatif ke baseline kalibrasi
    pitch: float = 0.0            # < 0 = menunduk
    roll: float = 0.0
    mouth_open: bool = False
    jaw_open: float = 0.0
    # bukaan mulut 0..1 untuk deteksi menguap (D-01): blendshape jawOpen jika model memberinya,
    # jika tidak -> mouth aspect ratio dari landmark / signals.mouth_mar_full. None = tidak terbaca.
    yawn_open: float | None = None
    calibrated: bool = False
    pose_only: bool = False       # pose dari cadangan PnP (landmark gagal): mata/mulut tidak terbaca


def eye_aspect_ratio(pts: np.ndarray) -> float:
    """EAR standar (Soukupová & Čech 2016): (|p2-p6| + |p3-p5|) / (2 |p1-p4|).

    Catatan: rumus asli di PDF Technical Spec terpotong saat ekspor; ini rumus EAR standar
    dengan urutan indeks yang sama seperti di spec.
    """
    p1, p2, p3, p4, p5, p6 = pts
    horiz = np.linalg.norm(p1 - p4)
    if horiz < 1e-6:
        return 0.0
    return float((np.linalg.norm(p2 - p6) + np.linalg.norm(p3 - p5)) / (2.0 * horiz))


def mouth_aspect_ratio(lm: np.ndarray) -> float:
    """MAR = tinggi celah bibir dalam / lebar mulut (sudut ke sudut). ~0 tertutup, besar saat menguap."""
    width = np.linalg.norm(lm[MOUTH_LEFT] - lm[MOUTH_RIGHT])
    if width < 1e-6:
        return 0.0
    return float(np.linalg.norm(lm[MOUTH_INNER_TOP] - lm[MOUTH_INNER_BOTTOM]) / width)


class _Median:
    def __init__(self, n: int):
        self.buf: deque = deque(maxlen=max(1, n))

    def __call__(self, v: float) -> float:
        self.buf.append(v)
        return float(np.median(self.buf))

    def reset(self):
        self.buf.clear()


class SignalExtractor:
    """FaceObs -> FaceSignals, dengan median filter dan baseline kalibrasi."""

    _KEYS = ("ear_l", "ear_r", "blink", "yaw", "pitch", "roll", "jaw")

    def __init__(self, sig_cfg: dict, calibration):
        self.c = sig_cfg
        self.cal = calibration
        n = int(sig_cfg.get("median_frames", 3))
        self._med = {k: _Median(n) for k in self._KEYS}
        self._closed = False

    def _combine(self, by_open: bool, by_blink: bool) -> bool:
        logic = self.c.get("closed_logic", "and")
        if logic == "and":
            return by_open and by_blink
        if logic == "or":
            return by_open or by_blink
        if logic == "blink_only":
            return by_blink
        return by_open

    def raw_values(self, obs: FaceObs) -> dict:
        lm = obs.landmarks_px
        ear_l = eye_aspect_ratio(lm[LEFT_EYE])
        ear_r = eye_aspect_ratio(lm[RIGHT_EYE])
        bs = obs.blendshapes
        if "jawOpen" in bs:
            jaw = bs["jawOpen"]
        else:
            # model tanpa blendshape (mis. MtDMS nanti): skala MAR ke rentang ~jawOpen
            full = self.c.get("mouth_mar_full", 0.9)
            jaw = float(np.clip(mouth_aspect_ratio(lm) / max(1e-6, full), 0.0, 1.0))
        return {
            "ear_l": ear_l, "ear_r": ear_r,
            "blink": max(bs.get("eyeBlinkLeft", 0.0), bs.get("eyeBlinkRight", 0.0)),
            "yaw": obs.yaw_raw, "pitch": obs.pitch_raw, "roll": obs.roll_raw,
            "jaw": jaw,
        }

    def update(self, obs: FaceObs, ts_ms: int) -> FaceSignals:
        if obs.present and obs.pose_only:
            # hanya pose (D-04/D-05 tetap jalan); mata tak terbaca -> D-01/D-02 memperlakukannya sebagai celah.
            # Tidak masuk kalibrasi (butuh EAR). Median filter pose tetap bersambung.
            # sumbu tak terbaca = NaN: tidak masuk median filter, perbandingan ambang di indikator selalu False
            def axis(k, v, base):
                return float("nan") if v != v else self._med[k](v) - base
            yaw = axis("yaw", obs.yaw_raw, self.cal.yaw0)
            pitch = axis("pitch", obs.pitch_raw, self.cal.pitch0)
            roll = axis("roll", obs.roll_raw, self.cal.roll0)
            p = obs.eye_closed_p
            ec = self.c.get("eye_classifier") or {}
            if p is not None and not (abs(yaw) > ec.get("yaw_max_deg", 30.0)):
                # A9: keadaan mata dari classifier potongan mata (titik mata detektor), dengan hysteresis.
                # Tanpa pitch (PnP 5 titik tak andal) -> menunduk ke pangkuan tidak bisa dikecualikan di sini.
                thr = ec.get("exit_p", 0.4) if self._closed else ec.get("enter_p", 0.6)
                self._closed = p >= thr
                strict = p >= ec.get("enter_p", 0.6)
                return FaceSignals(ts_ms=ts_ms, face_present=True, eye_valid=True, eye_closed=self._closed,
                                   eye_closed_strict=strict, openness=1.0 - p, yaw=yaw, pitch=pitch, roll=roll,
                                   yawn_open=None, calibrated=self.cal.done, pose_only=True)
            self._closed = False
            return FaceSignals(ts_ms=ts_ms, face_present=True, eye_valid=False, eye_closed=False,
                               eye_closed_strict=False, yaw=yaw, pitch=pitch, roll=roll, yawn_open=None,
                               calibrated=self.cal.done, pose_only=True)
        if not obs.present or obs.landmarks_px is None:
            for m in self._med.values():
                m.reset()
            self._closed = False
            return FaceSignals(ts_ms=ts_ms, face_present=False, calibrated=self.cal.done)

        raw = self.raw_values(obs)
        self.cal.observe(ts_ms, raw)
        v = {k: self._med[k](raw[k]) for k in self._KEYS}

        ear = (v["ear_l"] + v["ear_r"]) / 2.0
        base = self.cal.ear_baseline
        openness = ear / base if base > 1e-6 else 1.0
        yaw = v["yaw"] - self.cal.yaw0
        pitch = v["pitch"] - self.cal.pitch0
        roll = v["roll"] - self.cal.roll0

        c = self.c
        open_max = c.get("closed_openness_max", 0.30)
        blink_min = c.get("closed_blink_min", 0.50)
        strict = self._combine(openness < open_max, v["blink"] > blink_min)
        if self._closed:
            # hysteresis: setelah tertutup, mata dianggap tetap tertutup sampai jelas terbuka lagi
            # (pejam santai berayun di sekitar ambang masuk -> run D-02 tidak terputus)
            ex_open = c.get("closed_openness_exit")
            ex_blink = c.get("closed_blink_exit")
            open_max = open_max if ex_open is None else ex_open
            blink_min = blink_min if ex_blink is None else ex_blink
        eye_closed = self._combine(openness < open_max, v["blink"] > blink_min)
        self._closed = bool(eye_closed)

        eye_valid = (pitch >= -c.get("eye_valid_pitch_down_max_deg", 25)
                     and abs(yaw) <= c.get("eye_valid_yaw_max_deg", 45))

        return FaceSignals(
            ts_ms=ts_ms, face_present=True,
            ear_l=v["ear_l"], ear_r=v["ear_r"], ear=ear, openness=openness, blink=v["blink"],
            eye_closed=bool(eye_closed), eye_closed_strict=bool(strict), eye_valid=bool(eye_valid),
            yaw=yaw, pitch=pitch, roll=roll,
            mouth_open=v["jaw"] > c.get("mouth_open_jaw_min", 0.6), jaw_open=v["jaw"],
            yawn_open=float(v["jaw"]),
            calibrated=self.cal.done,
        )
