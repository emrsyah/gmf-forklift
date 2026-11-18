"""Menguap -> skor kantuk D-01 (sinyal sintetis). Kode tetap D01/A-02, Lembar A tidak berubah."""
import numpy as np

from src.cabin.calibration import Calibration
from src.cabin.face import FaceObs
from src.cabin.indicators import D01Drowsy
from src.cabin.signals import (MOUTH_INNER_BOTTOM, MOUTH_INNER_TOP, MOUTH_LEFT, MOUTH_RIGHT,
                               SignalExtractor)
from tests.synth import Timeline, run_indicator, run_system, starts, thresholds

TH = thresholds()
D01 = TH["D01_drowsy"]
WIDE = 0.75      # bukaan menguap
TALK = 0.70      # bicara keras: amplitudo sengaja DI ATAS yawn_open_min -> yang diuji durasi/penggabungan


def _d01(**over):
    return D01Drowsy(dict(D01, **over))


def yawn(tl: Timeline, dur_s: float = 3.0, **face) -> Timeline:
    return tl.add(dur_s, dict(yawn_open=WIDE, **face))


def rest(tl: Timeline, dur_s: float) -> Timeline:
    return tl.add(dur_s, dict(yawn_open=0.05))


def blinky(tl: Timeline, dur_s: float, closed_s: float = 0.3, period_s: float = 2.0) -> Timeline:
    """Mata terpejam singkat berulang (< slow_blink_s min) -> PERCLOS sedang tanpa kedip lambat."""
    n = int(round(dur_s / period_s))
    for _ in range(n):
        tl.closed(closed_s, yawn_open=0.05)
        rest(tl, period_s - closed_s)
    return tl


def _yawns(ind) -> int:
    return ind.info.get("yawns", 0)


# ------------------------------------------------------------------ deteksi menguap
def test_talking_is_not_yawning():
    tl = Timeline()
    rest(tl, 1)
    i = 0
    while len(tl.frames) < 60 * 30:
        tl.add([0.4, 0.6, 0.8, 0.5][i % 4], dict(yawn_open=TALK))
        tl.add([0.15, 0.3, 0.2, 0.25][i % 4], dict(yawn_open=0.1))
        i += 1
    ind = _d01()
    out = run_indicator(ind, tl)
    assert _yawns(ind) == 0 and not any(out)


def test_one_wide_opening_is_one_yawn():
    tl = rest(yawn(rest(Timeline(), 2), 3.0), 2)
    ind = _d01()
    run_indicator(ind, tl)
    assert _yawns(ind) == 1


def test_short_opening_below_min_is_not_a_yawn():
    tl = rest(yawn(rest(Timeline(), 2), 1.5), 2)     # tawa lebar 1,5 s
    ind = _d01()
    run_indicator(ind, tl)
    assert _yawns(ind) == 0


def test_two_frame_dropout_does_not_split_yawn():
    tl = rest(Timeline(), 2)
    yawn(tl, 1.5)
    tl.add(2 / 30, dict(yawn_open=0.1))              # 2 frame bukaan jatuh
    yawn(tl, 1.5)
    rest(tl, 2)
    ind = _d01()
    run_indicator(ind, tl)
    assert _yawns(ind) == 1


def test_face_lost_briefly_does_not_split_yawn():
    tl = rest(Timeline(), 2)
    yawn(tl, 1.2)
    tl.add(0.3, dict(face_present=False))           # tangan menutup mulut -> wajah hilang sesaat
    yawn(tl, 1.2)
    rest(tl, 2)
    ind = _d01()
    run_indicator(ind, tl)
    assert _yawns(ind) == 1


def test_very_long_opening_is_not_a_yawn():
    tl = rest(yawn(rest(Timeline(), 2), D01["yawn_max_s"] + 2), 2)   # makan / mulut menganga
    ind = _d01()
    run_indicator(ind, tl)
    assert _yawns(ind) == 0


def test_yawning_flag_while_in_progress():
    tl = yawn(rest(Timeline(), 2), 2.5)
    ind = _d01()
    run_indicator(ind, tl)
    assert ind.info["yawning"] is True and _yawns(ind) == 0     # belum selesai -> belum dihitung


# ------------------------------------------------------------------ skor kantuk
def test_moderate_perclos_plus_two_yawns_fires_d01_and_releases():
    tl = rest(Timeline(), 1)
    yawn(tl)
    blinky(tl, 30)
    yawn(tl)
    blinky(tl, 20)
    ind = _d01()
    out = run_indicator(ind, tl)
    assert any(out)
    assert ind.info["perclos"] < D01["perclos_min"]          # jalur mata saja tidak cukup
    assert not out[-1]                                        # lepas setelah yawn_active_hold_s
    # kontrol: pola mata sama tanpa menguap -> tidak aktif
    tl2 = rest(Timeline(), 1)
    rest(tl2, 3)
    blinky(tl2, 30)
    rest(tl2, 3)
    blinky(tl2, 20)
    assert not any(run_indicator(_d01(), tl2))


def test_one_yawn_with_own_eye_closure_does_not_fire():
    tl = rest(Timeline(), 25)
    tl.closed(1.2, yawn_open=WIDE)                         # mata ikut terpejam saat menguap
    yawn(tl, 1.0)
    tl.closed(1.0, yawn_open=WIDE)
    rest(tl, 10)
    ind = _d01()
    assert not any(run_indicator(ind, tl))
    assert _yawns(ind) == 1


def test_normal_blinking_plus_max_bonus_does_not_fire():
    tl = rest(Timeline(), 2)
    for _ in range(2):
        yawn(tl)
        blinky(tl, 30, closed_s=0.15, period_s=3.0)          # kedip normal
    ind = _d01()
    assert not any(run_indicator(ind, tl))
    assert _yawns(ind) == 2 and ind.score < 1.0


def test_yawns_alone_below_count_do_not_fire_then_count_fires():
    alone = D01["yawn_alone_count"]
    tl = rest(Timeline(), 2)
    for _ in range(alone - 1):
        yawn(tl)
        rest(tl, 30)
    ind = _d01()
    assert not any(run_indicator(ind, tl))
    tl2 = Timeline(t0_ms=int(tl.t_ms))
    yawn(tl2)
    rest(tl2, 1)
    assert any(run_indicator(ind, tl2))
    assert ind.condition_start_ms >= tl2.t0_ms             # awal = menguap pemicu, bukan menit lalu


def test_perclos_path_unchanged_by_yawn_feature():
    """Tanpa bukaan mulut: aktif + confidence identik dengan fitur menguap ON/OFF."""
    tl = rest(Timeline(), 12)
    for _ in range(8):
        tl.closed(0.8)
        rest(tl, 2.0)
    on, off = _d01(yawn_enabled=True), _d01(yawn_enabled=False)
    seq_on = [(on.update(x), on.confidence) for x in tl.frames]
    seq_off = [(off.update(x), off.confidence) for x in tl.frames]
    assert seq_on == seq_off and any(a for a, _ in seq_on)


# ------------------------------------------------------------------ sistem: prioritas & hold
D02 = TH["D02_long_closure"]


def test_eyes_closed_while_yawning_is_not_d02():
    """Rekaman webcam 2026-11-18: menguap memejamkan mata 1,5-3 s -> dulu D-02 A-03 (kode salah)."""
    tl = rest(Timeline(), 2)
    yawn(tl, 0.5)
    tl.closed(3.0, yawn_open=WIDE)                         # pejam 3 s selama mulut terbuka lebar
    tl.closed(D02["yawn_mask"]["release_s"] * 0.8, yawn_open=0.05)   # mata baru terbuka sesaat setelah mulut menutup
    rest(tl, 5)
    events, first = run_system(tl)
    assert "D02" not in first


def test_microsleep_continuing_after_yawn_is_d02():
    """Mata tetap terpejam setelah mulut menutup -> mikrotidur, tetap D-02 (dihitung setelah mask lepas)."""
    tl = rest(Timeline(), 2)
    yawn(tl, 1.0)
    tl.closed(2.5, yawn_open=WIDE)
    t_mouth_closed = tl.t_ms
    tl.closed(D02["yawn_mask"]["release_s"] + D02["closed_s"] + 1.0, yawn_open=0.05)
    rest(tl, 10)
    events, first = run_system(tl)
    d02 = [e for e in starts(events) if e["code"] == "D02"]
    assert d02 and d02[0]["level"] == "A-03"
    assert d02[0]["ts_event_ms"] >= t_mouth_closed + (D02["yawn_mask"]["release_s"] + D02["closed_s"]) * 1000 - 50


def test_yawn_eye_closure_does_not_inflate_perclos():
    tl = rest(Timeline(), 12)
    for _ in range(2):
        yawn(tl, 0.5)
        tl.closed(3.0, yawn_open=WIDE)
        rest(tl, 8)
    ind = _d01()
    run_indicator(ind, tl)
    assert ind.info["perclos"] < 0.02 and ind.info["slow_blinks"] == 0
    off = _d01(yawn_eye_mask=False)
    run_indicator(off, tl)
    assert off.info["perclos"] > ind.info["perclos"]       # kontrol: tanpa mask, pejaman menguap masuk PERCLOS


def test_d02_microsleep_wins_over_yawn_d01():
    tl = rest(Timeline(), 2)
    for _ in range(D01["yawn_alone_count"] - 1):
        yawn(tl)
        rest(tl, 20)
    yawn(tl, 1.0)
    tl.closed(2.5, yawn_open=WIDE)                         # menguap ke-3 sambil mata terpejam ...
    tl.closed(4.0, yawn_open=0.05)                         # ... dan tetap terpejam setelah mulut menutup
    rest(tl, 10)
    events, first = run_system(tl)
    st = starts(events)
    assert "D01" in first and "D02" in first
    d02 = [e for e in st if e["code"] == "D02"]
    assert d02 and d02[0]["level"] == "A-03"               # mikrotidur tetap D-02 dan menyela D-01
    assert all(e["level"] == "A-02" for e in st if e["code"] == "D01")


def test_yawn_d01_releases_then_d05_can_fire():
    tl = rest(Timeline(), 2)
    for _ in range(D01["yawn_alone_count"]):
        yawn(tl)
        rest(tl, 10)
    tl.add(3.0, dict(yawn_open=0.05, yaw=45.0))             # menoleh (skenario 8)
    rest(tl, 3)
    events, _ = run_system(tl)
    codes = [e["code"] for e in starts(events)]
    assert codes == ["D01", "D05"]


def test_yawn_never_triggers_d02_d04_d05():
    tl = rest(Timeline(), 2)
    for _ in range(4):
        yawn(tl, 5.0)
        rest(tl, 5)
    events, first = run_system(tl)
    assert set(first) <= {"D01"}


# ------------------------------------------------------------------ sinyal
def _sig_from(blendshapes, mar):
    cal = Calibration(TH["calibration"])
    ex = SignalExtractor(dict(TH["signals"], median_frames=1), cal)
    ex.cal.observe = lambda *a, **k: None
    lm = np.zeros((478, 2), dtype=np.float32)
    lm[MOUTH_LEFT] = (100, 200)
    lm[MOUTH_RIGHT] = (160, 200)
    lm[MOUTH_INNER_TOP] = (130, 190)
    lm[MOUTH_INNER_BOTTOM] = (130, 190 + 60 * mar)
    return ex.update(FaceObs(present=True, landmarks_px=lm, blendshapes=blendshapes), 0)


def test_yawn_open_from_blendshape_or_mar_fallback():
    assert abs(_sig_from({"jawOpen": 0.7}, 0.1).yawn_open - 0.7) < 1e-6
    s = _sig_from({}, 0.45)
    assert abs(s.yawn_open - 0.45 / TH["signals"]["mouth_mar_full"]) < 1e-3
    from tests.synth import sig
    assert sig().yawn_open is None                          # sintetis lama: tanpa mulut -> jalur inert
