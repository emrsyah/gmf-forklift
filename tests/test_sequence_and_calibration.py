"""Urutan 20 skenario Lembar A berturut-turut (jeda 30 s) + kalibrasi + preprocessing input YOLO."""
import numpy as np
import pytest

from src.cabin.calibration import Calibration
from tests.synth import Timeline, run_system, starts, thresholds

GAP_S = 30.0


def _neutral(tl, dur):
    t = 0.0
    while t < dur:
        tl.add(2.85, yolo_valid=True).add(0.15, dict(eye_closed=True, openness=0.1, blink=0.9), yolo_valid=True)
        t += 3.0


def _nods(tl, dur, depth=-28):
    t = 0.0
    while t + 3.7 <= dur:
        tl.ramp(0.6, "pitch", 0, depth, ).add(0.5, dict(pitch=depth), yolo_valid=True) \
          .ramp(0.6, "pitch", depth, 0).add(2.0, yolo_valid=True)
        t += 3.7
    tl.add(dur - t, yolo_valid=True)


def _scenario(tl, no):
    Y = dict(yolo_valid=True)
    if no in (1, 2, 3, 4):
        t = 0.0
        while t < 60:
            tl.add(3.1, **Y).add(0.9, dict(eye_closed=True, openness=0.1, blink=0.9), **Y)
            t += 4.0
    elif no in (5, 6):
        tl.add(3, dict(eye_closed=True, openness=0.1, blink=0.9), **Y)
    elif no == 7:
        tl.add(3, dict(eye_closed=True, openness=0.1, blink=0.9, pitch=-18), **Y)
    elif no == 8:
        tl.add(3, dict(yaw=55), **Y)
    elif no == 9:
        tl.add(3, dict(pitch=-40, eye_valid=False, eye_closed=True), **Y)
    elif no == 10:
        tl.add(3, dict(yaw=-55), **Y)
    elif no == 11:
        for i in range(30):
            tl.add(0.7, dict(yaw=45 if i % 2 == 0 else -45), **Y).add(0.3, **Y)
    elif no == 12:
        tl.add(3, dict(pitch=-30), phone_conf=0.8, **Y)
    elif no == 13:
        tl.add(3, dict(yaw=10), phone_conf=0.8, **Y)
    elif no == 14:
        tl.add(3, phone_conf=0.7, **Y)
    elif no == 15:
        tl.add(15, no_helmet=True, **Y)
    elif no == 16:
        tl.add(15, no_vest=True, **Y)
    elif no == 17:
        tl.add(15, no_helmet=True, no_vest=True, **Y)
    elif no == 18:
        _nods(tl, 30)
    elif no == 19:
        tl.ramp(0.5, "roll", 0, 35).add(4.5, dict(roll=35), **Y)
    elif no == 20:
        tl.add(15, dict(face_present=False), blocked_frame=True, **Y)


EXPECT = {1: ("D01", "A-02"), 2: ("D01", "A-02"), 3: ("D01", "A-02"), 4: ("D01", "A-02"),
          5: ("D02", "A-03"), 6: ("D02", "A-03"), 7: ("D02", "A-03"),
          8: ("D05", "A-02"), 9: ("D05", "A-02"), 10: ("D05", "A-02"), 11: ("D05", "A-02"),
          12: ("D06", "A-02"), 13: ("D06", "A-02"), 14: ("D06", "A-02"),
          15: ("D10", "A-02"), 16: ("D10", "A-02"), 17: ("D10", "A-02"),
          18: ("D04", "A-03"), 19: ("D04", "A-03"), 20: ("D09", "A-03")}


def run_sheet(th=None):
    tl = Timeline()
    _neutral(tl, 10)
    windows = {}
    for no in range(1, 21):
        s = (tl.t_ms - tl.t0_ms) / 1000
        _scenario(tl, no)
        e = (tl.t_ms - tl.t0_ms) / 1000
        windows[no] = (s, e)
        _neutral(tl, GAP_S)
    ev, _ = run_system(tl, th)
    st = starts(ev)
    res = {}
    claimed = set()
    for no, (s, e) in windows.items():
        hits = [x for x in st if s <= x["t_video_s"] <= e + 2.0]
        claimed.update(x["event_id"] for x in hits)
        res[no] = (hits[0]["code"], hits[0]["level"], hits[0]["t_video_s"] - s) if hits else None
    stray = [x for x in st if x["event_id"] not in claimed]
    return res, stray, ev


def test_full_sheet_back_to_back_spec_defaults():
    res, stray, _ = run_sheet()
    wrong = {no: (res[no], EXPECT[no]) for no in EXPECT if not res[no] or res[no][:2] != EXPECT[no]}
    assert wrong == {}, wrong
    assert stray == [], [(x["code"], round(x["t_video_s"], 1)) for x in stray]
    for no in (5, 6, 7, 8, 9, 10, 12, 13, 14):   # skenario 3 detik: deteksi <= 2,0 s
        assert res[no][2] <= 2.0, (no, res[no])


def test_d04_buzzes_long_after_nodding_stops_and_hold_option_fixes_it():
    """Spec-literal: D-04 tetap aktif selama >= 2 anggukan masih di jendela 30 s -> event A-03 (buzzer berulang)
    berlanjut lama setelah skenario 18 selesai. Opsi nod_active_hold_s menutupnya cepat."""
    def end_after_18(th):
        _, _, ev = run_sheet(th)
        d04 = [e for e in ev if e["code"] == "D04"]
        first_start = next(e for e in d04 if e["phase"] == "start")
        first_end = next(e for e in d04 if e["phase"] == "end" and e["event_id"] == first_start["event_id"])
        return first_end["t_video_s"] - first_start["t_video_s"]

    literal = end_after_18(thresholds({"D04_head_drop": {"nod_active_hold_s": None, "nod_count": 2}}))
    fixed = end_after_18(thresholds({"D04_head_drop": {"nod_active_hold_s": 5.0}}))
    assert literal > 45          # 30 s skenario + ~20 s ekor
    assert fixed < 36            # selesai ~beberapa detik setelah anggukan terakhir
    res, stray, _ = run_sheet(thresholds({"D04_head_drop": {"nod_active_hold_s": 5.0}}))
    assert res[18][:2] == ("D04", "A-03") and res[19][:2] == ("D04", "A-03") and stray == []


# ------------------------------------------------------------------ kalibrasi
def _feed(cal, n, ts0=0, ear=0.30, blink=0.0, **pose):
    for i in range(n):
        raw = {"ear_l": ear, "ear_r": ear, "blink": blink, "yaw": pose.get("yaw", 0.0),
               "pitch": pose.get("pitch", 0.0), "roll": pose.get("roll", 0.0)}
        cal.observe(ts0 + i * 33, raw)
    return ts0 + n * 33


CAL = thresholds()["calibration"]


def test_calibration_neutral_ok():
    cal = Calibration(CAL)
    _feed(cal, 200, pitch=-12)       # kamera di bawah mata: pitch mentah tidak nol, tetap diterima
    assert cal.done and cal.pitch0 == pytest.approx(-12) and cal.ear_baseline == pytest.approx(0.30)


def test_calibration_rejects_tilted_head_then_falls_back():
    cal = Calibration(CAL)
    ts = 0
    for _ in range(CAL["max_attempts"]):
        ts = _feed(cal, 160, ts, roll=30)   # skenario 19: kepala miring bertahan
    assert not cal.done and not cal.active
    assert "populasi" in cal.status
    assert cal.ear_baseline == pytest.approx(0.30)          # baseline adaptif (EAR peraga), bukan populasi 0.28
    off = Calibration({**CAL, "adaptive_baseline": {"enabled": False}})
    for i in range(CAL["max_attempts"]):
        _feed(off, 160, i * 160 * 33, roll=30)
    assert off.ear_baseline == CAL["population_ear_baseline"]
    _feed(cal, 400, ts, roll=30)            # tidak lagi mencoba -> tidak terkunci di pose miring
    assert not cal.done and cal.roll0 == 0.0
    cal.start()                             # hotkey c
    _feed(cal, 200, ts + 20000)
    assert cal.done


def test_calibration_rejects_steady_closed_eyes():
    # sim_P1c: percobaan ulang jatuh di skenario 5 (pejam 3 s, diam) -> dulu diterima, baseline = mata tertutup
    cal = Calibration(CAL)
    _feed(cal, 153, ear=0.09, blink=0.8)           # tepat satu percobaan (5 s): EAR rendah + blink tinggi
    assert not cal.done and "tertutup" in cal.status
    cal = Calibration(CAL)
    _feed(cal, 160, ear=0.09)                      # model tanpa blendshape: cek EAR vs populasi
    assert not cal.done
    cal = Calibration(CAL)
    _feed(cal, 200, ear=0.19, blink=0.2)           # mata sipit terbuka tetap diterima
    assert cal.done and cal.ear_baseline == pytest.approx(0.19)


def _feed_blinking(cal, n, ts0=0):
    """Mata terbuka (EAR 0.30) dengan kedipan normal: 4 frame terpejam tiap 30 frame (~1 s)."""
    for i in range(n):
        closed = i % 30 < 4
        cal.observe(ts0 + i * 33, {"ear_l": 0.08 if closed else 0.30, "ear_r": 0.08 if closed else 0.30,
                                   "blink": 0.9 if closed else 0.0, "yaw": 0.0, "pitch": 0.0, "roll": 0.0})


def test_calibration_ear_std_open_only_ignores_blinks():
    # Jetson 2026-11-25: kedipan biasa membuat std EAR ~0.1 > 0.04 -> kalibrasi hampir selalu ditolak
    cal = Calibration({**CAL, "ear_std_open_only": False})
    _feed_blinking(cal, 153)                        # tepat satu percobaan (5 s)
    assert not cal.done and "tidak stabil" in cal.status
    cal = Calibration({**CAL, "ear_std_open_only": True})
    _feed_blinking(cal, 153)
    assert cal.done and cal.ear_baseline == pytest.approx(0.30)


def test_adaptive_baseline_learns_open_eye_ear_despite_closures():
    # sim_P1b: peraga bermata sipit (EAR terbuka 0.18) memejam berulang sejak awal -> kalibrasi gagal 3x
    cal = Calibration(CAL)
    ts = 0
    for _ in range(12):                                     # 12 siklus: terbuka 2 s, terpejam 3 s
        ts = _feed(cal, 60, ts, ear=0.18)
        ts = _feed(cal, 90, ts, ear=0.07, blink=0.9)
    assert not cal.done and "populasi" in cal.status
    assert cal.ear_baseline == pytest.approx(0.18)         # bukan 0.28 (populasi) dan bukan EAR terpejam
    # hampir selalu terpejam: lantai populasi x rasio, tidak belajar mata tertutup
    cal = Calibration(CAL)
    _feed(cal, 2000, ear=0.07, blink=0.9)
    assert cal.ear_baseline == pytest.approx(CAL["population_ear_baseline"] * CAL["min_ear_ratio_population"])
    # menoleh (tidak frontal) tidak masuk perkiraan
    cal = Calibration(CAL)
    _feed(cal, 400, ear=0.40, yaw=35)
    assert cal.ear_baseline == CAL["population_ear_baseline"]


# ------------------------------------------------------------------ input YOLO = preprocessing training
def test_yolo_receives_nir_input_when_simulate_nir_off():
    from src.backends import Detector
    from src.cabin.face import FaceObs
    from src.cabin.pipeline import CabinPipeline
    from src.capture import Frame
    from src.config import Config

    class FakeFace:
        def process(self, img, ts, off=(0, 0)):
            return FaceObs(present=False)

    class SpyDetector(Detector):
        names = ["phone", "helmet", "head_no_helmet", "vest", "torso"]

        def __init__(self):
            self.inputs = []

        def predict(self, image_bgr, conf=0.25):
            self.inputs.append(image_bgr.copy())
            return []

    cfg = Config.load()
    cfg.runtime["cabin"]["simulate_nir"] = False
    spy = SpyDetector()
    p = CabinPipeline(cfg.runtime, cfg.thresholds, cfg.events, face_model=FakeFace(), detector=spy)
    rng = np.random.default_rng(0)
    img = np.zeros((120, 160, 3), np.uint8)
    img[..., 2] = rng.integers(100, 255, (120, 160))      # gambar merah berwarna
    p.process(Frame(img, 1000, 0))
    x = spy.inputs[0]
    assert np.array_equal(x[..., 0], x[..., 1]) and np.array_equal(x[..., 1], x[..., 2])   # grayscale 3 channel

    cfg.runtime["cabin"]["yolo"]["preprocess"] = "none"
    spy2 = SpyDetector()
    p2 = CabinPipeline(cfg.runtime, cfg.thresholds, cfg.events, face_model=FakeFace(), detector=spy2)
    p2.process(Frame(img, 1000, 0))
    assert not np.array_equal(spy2.inputs[0][..., 0], spy2.inputs[0][..., 2])
