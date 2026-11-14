"""Unit test indikator dengan sinyal sintetis (aturan kerja 8)."""
import pytest

from src.cabin.indicators import (D01Drowsy, D02LongClosure, D04HeadDrop, D05Distraction,
                                  D06Phone, D09CameraBlock, D10PPE)
from tests.synth import Timeline, run_indicator, thresholds

TH = thresholds()


# ------------------------------------------------------------------ D-02
def test_d02_closed_1_4s_not_active():
    tl = Timeline().opened(1).closed(1.4).opened(1)
    assert not any(run_indicator(D02LongClosure(TH["D02_long_closure"]), tl))


def test_d02_closed_1_6s_active():
    tl = Timeline().opened(1).closed(1.6)
    out = run_indicator(D02LongClosure(TH["D02_long_closure"]), tl)
    assert out[-1]
    # aktif tepat setelah 1,5 s (deteksi < 2 s untuk skenario 3 detik)
    first = out.index(True)
    assert (first - 30) / 30 == pytest.approx(1.5, abs=0.05)


def test_d02_short_face_gap_is_tolerated():
    tl = Timeline().closed(1.0).add(0.2, dict(face_present=False)).closed(0.4)
    assert run_indicator(D02LongClosure(TH["D02_long_closure"]), tl)[-1]


def test_d02_long_face_gap_resets():
    tl = Timeline().closed(1.0).add(0.5, dict(face_present=False)).closed(0.9)
    assert not any(run_indicator(D02LongClosure(TH["D02_long_closure"]), tl))


def test_d02_ignores_closure_when_eye_invalid():
    # skenario 9: menunduk ke pangkuan -> kelopak tampak tertutup tapi eye_valid False
    tl = Timeline().closed(3.0, eye_valid=False, pitch=-35)
    assert not any(run_indicator(D02LongClosure(TH["D02_long_closure"]), tl))


# ------------------------------------------------------------------ D-01
def _slow_blink_session(blink_s=0.8, period_s=5.0, total_s=40.0):
    tl = Timeline()
    t = 0.0
    while t < total_s:
        tl.opened(period_s - blink_s).closed(blink_s)
        t += period_s
    return tl


def test_d01_slow_blinks_trigger():
    out = run_indicator(D01Drowsy(TH["D01_drowsy"]), _slow_blink_session())
    assert any(out)


def test_d01_normal_blinks_do_not_trigger():
    tl = Timeline()
    for _ in range(15):
        tl.opened(3.85).closed(0.15)
    assert not any(run_indicator(D01Drowsy(TH["D01_drowsy"]), tl))


def test_d01_perclos_trigger():
    tl = Timeline()
    for _ in range(12):  # 25% waktu tertutup, pejaman 0,3 s (bukan kedip lambat)
        tl.opened(0.9).closed(0.3)
    assert run_indicator(D01Drowsy(TH["D01_drowsy"]), tl)[-1]


def test_d01_needs_valid_ratio():
    tl = Timeline()
    for _ in range(8):
        tl.opened(4.2, eye_valid=False).closed(0.8)
    assert not any(run_indicator(D01Drowsy(TH["D01_drowsy"]), tl))


# ------------------------------------------------------------------ D-04
def _nod(tl, depth=-25, down_s=0.5, hold_s=0.4, up_s=0.5):
    tl.ramp(down_s, "pitch", 0, depth).add(hold_s, dict(pitch=depth)).ramp(up_s, "pitch", depth, 0)
    return tl


def test_d04_nod_count_reached_active():
    th = TH["D04_head_drop"]
    tl = Timeline().opened(1)
    for _ in range(th["nod_count"]):
        _nod(tl); tl.opened(3)
    assert any(run_indicator(D04HeadDrop(th), tl)[-int(3 * 30):])


def test_d04_two_nods_spec_value_active():
    tl = Timeline().opened(1)
    _nod(tl); tl.opened(3); _nod(tl); tl.opened(0.5)
    assert run_indicator(D04HeadDrop(dict(TH["D04_head_drop"], nod_count=2)), tl)[-1]


def test_d04_two_glances_not_active_with_current_config():
    tl = Timeline().opened(1)
    _nod(tl); tl.opened(3); _nod(tl); tl.opened(0.5)
    assert not any(run_indicator(D04HeadDrop(TH["D04_head_drop"]), tl))


def test_d04_one_nod_not_active():
    tl = Timeline().opened(1)
    _nod(tl); tl.opened(3)
    assert not any(run_indicator(D04HeadDrop(TH["D04_head_drop"]), tl))


def test_d04_sustained_look_down_is_not_a_nod():
    # skenario 9: menunduk diam 3 s lalu kembali -> bukan anggukan (melebihi nod_max_s)
    tl = Timeline().opened(1).ramp(0.5, "pitch", 0, -35).add(3, dict(pitch=-35)).ramp(0.5, "pitch", -35, 0)
    tl.opened(1).ramp(0.5, "pitch", 0, -35).add(3, dict(pitch=-35)).ramp(0.5, "pitch", -35, 0)
    assert not any(run_indicator(D04HeadDrop(TH["D04_head_drop"]), tl))


def test_d04_roll_held():
    th = TH["D04_head_drop"]
    assert not any(run_indicator(D04HeadDrop(th), Timeline().add(1.4, dict(roll=30)).opened(1)))
    assert run_indicator(D04HeadDrop(th), Timeline().add(1.6, dict(roll=-30)))[-1]


# ------------------------------------------------------------------ D-05
def test_d05_continuous_threshold():
    th = TH["D05_distraction"]
    assert not any(run_indicator(D05Distraction(th), Timeline().opened(3).add(1.4, dict(yaw=40)).opened(0.1)))
    assert run_indicator(D05Distraction(th), Timeline().opened(3).add(1.6, dict(yaw=40)))[-1]


def test_d05_look_down_lap():
    assert run_indicator(D05Distraction(TH["D05_distraction"]), Timeline().opened(3).add(1.6, dict(pitch=-35)))[-1]


def test_d05_alternating_every_1s_accumulates():
    # skenario 11: menoleh kanan-kiri tiap 1 s, melewati tengah 0,3 s -> tidak pernah 1,5 s berturut-turut
    th = dict(TH["D05_distraction"])
    tl = Timeline().opened(2)
    for i in range(15):
        tl.add(0.7, dict(yaw=40 if i % 2 == 0 else -40)).opened(0.3)
    out = run_indicator(D05Distraction(th), tl)
    ind = D05Distraction(th)
    for x in tl.frames:
        ind.update(x)
        assert ind.off_run < th["continuous_s"]   # memang tidak pernah kontinu
    assert any(out)                              # tapi aktif lewat akumulasi


def test_d05_normal_glances_do_not_trigger():
    tl = Timeline()
    for _ in range(10):
        tl.opened(5).add(0.8, dict(yaw=40))      # lirik spion 0,8 s tiap ~6 s
    assert not any(run_indicator(D05Distraction(TH["D05_distraction"]), tl))


def test_d05_require_moving():
    from src.cabin.indicators import IndicatorInput  # noqa
    tl = Timeline().opened(1).add(2, dict(yaw=40), moving=False)
    assert not any(run_indicator(D05Distraction(TH["D05_distraction"]), tl))


# ------------------------------------------------------------------ D-06
def test_d06_phone_1s():
    th = TH["D06_phone"]
    tl = Timeline().add(0.5, yolo_valid=True).add(1.2, phone_conf=0.6, yolo_valid=True)
    assert run_indicator(D06Phone(th), tl)[-1]


def test_d06_flicker_not_enough():
    th = TH["D06_phone"]
    tl = Timeline()
    for _ in range(20):
        tl.add(0.1, phone_conf=0.6, yolo_valid=True).add(0.2, yolo_valid=True)
    assert not any(run_indicator(D06Phone(th), tl))


def test_d06_off_without_yolo():
    tl = Timeline().add(3, phone_conf=0.9, yolo_valid=False)
    assert not any(run_indicator(D06Phone(TH["D06_phone"]), tl))


# ------------------------------------------------------------------ D-09
def test_d09_blocked_1s_no_face():
    th = TH["D09_camera_block"]
    assert not any(run_indicator(D09CameraBlock(th), Timeline().add(0.9, dict(face_present=False), blocked_frame=True)))
    assert run_indicator(D09CameraBlock(th), Timeline().add(1.1, dict(face_present=False), blocked_frame=True))[-1]


def test_d09_not_when_face_present():
    tl = Timeline().add(3, blocked_frame=True)
    assert not any(run_indicator(D09CameraBlock(TH["D09_camera_block"]), tl))


# ------------------------------------------------------------------ D-10
def test_d10_no_helmet():
    tl = Timeline().add(3.5, no_helmet=True, yolo_valid=True)
    assert run_indicator(D10PPE(TH["D10_ppe"]), tl)[-1]


def test_d10_no_vest():
    tl = Timeline().add(3.5, no_vest=True, yolo_valid=True)
    assert run_indicator(D10PPE(TH["D10_ppe"]), tl)[-1]


def test_d10_short_glitch_not_active():
    tl = Timeline()
    for _ in range(10):
        tl.add(0.3, no_helmet=True, yolo_valid=True).add(0.7, yolo_valid=True)
    assert not any(run_indicator(D10PPE(TH["D10_ppe"]), tl))
