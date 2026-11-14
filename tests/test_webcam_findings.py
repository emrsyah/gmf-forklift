"""Regresi dari rekaman webcam 2026-11-14 (P0, diag4): pejam santai, tutup lensa, lihat pangkuan dalam."""
from src.cabin.calibration import Calibration
from src.cabin.camblock import is_blocked
from src.cabin.face import FaceObs
from src.cabin.indicators import D05Distraction
from src.cabin.signals import SignalExtractor
from tests.synth import Timeline, run_indicator, thresholds

TH = thresholds()


# ------------------------------------------------------------------ hysteresis mata tertutup
def _extractor(sig_cfg, seq):
    """SignalExtractor dengan raw_values diganti deret (openness, blink); baseline EAR = 1."""
    cal = Calibration(TH["calibration"])
    cal.ear_baseline = 1.0
    ex = SignalExtractor(dict(sig_cfg, median_frames=1), cal)
    it = iter(seq)

    def raw(_obs):
        o, b = next(it)
        return {"ear_l": o, "ear_r": o, "blink": b, "yaw": 0.0, "pitch": 0.0, "roll": 0.0, "jaw": 0.0}

    ex.raw_values = raw
    ex.cal.observe = lambda *a, **k: None
    obs = FaceObs(present=True)
    obs.landmarks_px = object()
    return [ex.update(obs, i * 33).eye_closed for i in range(len(seq))]


# pejam santai nyata: openness berayun 0.21-0.34 sementara blink tetap ~0.6
RELAXED = [(0.27, 0.67), (0.22, 0.68), (0.30, 0.60), (0.32, 0.56), (0.24, 0.61), (0.33, 0.67), (0.34, 0.64),
           (0.28, 0.67), (0.23, 0.67)]


def test_relaxed_closure_flickers_without_hysteresis():
    cfg = dict(TH["signals"], closed_openness_exit=None, closed_blink_exit=None)
    assert not all(_extractor(cfg, RELAXED))


def test_relaxed_closure_stays_closed_with_hysteresis():
    assert all(_extractor(TH["signals"], RELAXED))


def test_hysteresis_still_reopens_and_ignores_normal_blink():
    seq = RELAXED[:3] + [(0.85, 0.25)] + [(0.36, 0.66), (0.38, 0.6)]   # buka mata, lalu kedip biasa
    out = _extractor(TH["signals"], seq)
    assert out[:3] == [True] * 3 and out[3:] == [False] * 3


def test_d01_uses_strict_closure_so_hysteresis_does_not_create_slow_blinks():
    """Kedip normal 0,27 s yang dipanjangkan hysteresis jadi 0,45 s tidak boleh dihitung kedip lambat D-01."""
    from src.cabin.indicators import D01Drowsy
    th = dict(TH["D01_drowsy"])
    tl = Timeline().opened(2)
    for _ in range(8):   # 8 kedip normal dalam ~24 s
        tl.add(0.27, dict(eye_closed=True, eye_closed_strict=True)) \
          .add(0.18, dict(eye_closed=True, eye_closed_strict=False)).opened(2.6)
    ind = D01Drowsy(th)
    run_indicator(ind, tl)
    assert ind.info["slow_blinks"] == 0
    ind = D01Drowsy(dict(th, closed_signal="hysteresis"))
    run_indicator(ind, tl)
    assert ind.info["slow_blinks"] >= 4


def test_long_closure_early_in_session_sends_D02_not_D01():
    """Uji live: 12 s setelah mulai, pejam terus 3 s -> event pertama harus D-02 (bukan D-01 dari PERCLOS)."""
    from tests.synth import run_system, starts
    tl = Timeline().opened(10).closed(3).opened(5)
    codes = [e["code"] for e in starts(run_system(tl)[0])]
    assert codes == ["D02"]
    old = thresholds({"D01_drowsy": {"perclos_full_window": False}})
    assert "D01" in [e["code"] for e in starts(run_system(tl, old)[0])]   # perilaku lama: D-01 palsu


# ------------------------------------------------------------------ D-09 tangan dengan cahaya bocor
def test_leaky_palm_cover_is_blocked_but_normal_scene_is_not():
    cfg = TH["D09_camera_block"]
    assert is_blocked({"mean": 40.0, "std": 18.0, "lap_var": 45.0}, cfg)        # telapak tangan, webcam
    assert not is_blocked({"mean": 112.0, "std": 67.0, "lap_var": 850.0}, cfg)  # kabin normal
    assert not is_blocked({"mean": 80.0, "std": 44.0, "lap_var": 250.0}, cfg)   # tangan baru mendekat


# ------------------------------------------------------------------ D-05 wajah hilang saat menunduk dalam
def _deep_lap(blocked=False, hold=None):
    th = dict(TH["D05_distraction"])
    if hold is not None:
        th["face_lost_offroad_hold_s"] = hold
    tl = Timeline().opened(3).add(0.4, dict(pitch=-35)).add(2.6, dict(face_present=False), blocked_frame=blocked)
    return run_indicator(D05Distraction(th), tl)


def test_d05_deep_lap_face_lost_still_detected():
    assert _deep_lap()[-1]


def test_d05_deep_lap_face_lost_without_hold_misses():
    assert not any(_deep_lap(hold=0.0)[-30:])


def test_d05_face_lost_hold_not_used_when_camera_blocked():
    assert not any(_deep_lap(blocked=True))


def test_d05_face_lost_after_looking_ahead_is_not_offroad():
    tl = Timeline().opened(3).add(3, dict(face_present=False))
    assert not any(run_indicator(D05Distraction(TH["D05_distraction"]), tl))


def test_d05_pitch_wobble_around_threshold_keeps_continuous_run():
    tl = Timeline().opened(3)
    for _ in range(4):   # -27 / -24 bergantian tiap 0,2 s (lihat pangkuan biasa)
        tl.add(0.2, dict(pitch=-27)).add(0.2, dict(pitch=-24))
    th = dict(TH["D05_distraction"], min_window_fill_s=99)   # uji run kontinu saja, bukan akumulasi
    assert run_indicator(D05Distraction(th), tl)[-1]
    assert not run_indicator(D05Distraction(dict(th, hysteresis_deg=0.0)), tl)[-1]
