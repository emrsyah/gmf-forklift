"""Skenario Lembar A dengan sinyal sintetis lewat SEMUA indikator + modul keputusan.

Yang diuji: event `start` PERTAMA (yang dinilai K4) punya kode dan level yang benar,
dan waktu deteksi skenario 3 detik <= 2,0 s. Ini BUKAN laporan lulus — hanya cek logika.
"""
import pytest

from tests.synth import Timeline, run_system, starts, thresholds


def first_start(tl, th=None):
    ev, _ = run_system(tl, th)
    s = starts(ev)
    return s[0] if s else None


def det_time(e, scen_start_s):
    return e["t_video_s"] - scen_start_s


def test_s05_eyes_closed_3s_is_D02_fast():
    tl = Timeline().opened(5).closed(3).opened(5)
    e = first_start(tl)
    assert (e["code"], e["level"]) == ("D02", "A-03")
    assert det_time(e, 5) <= 2.0


def test_s07_closed_while_slightly_bowed_stays_D02():
    tl = Timeline().opened(5).ramp(0.3, "pitch", 0, -18).closed(3, pitch=-18).opened(5)
    e = first_start(tl)
    assert (e["code"], e["level"]) == ("D02", "A-03")
    assert det_time(e, 5) <= 2.0


def test_s08_turn_right_is_D05():
    tl = Timeline().opened(5).add(3, dict(yaw=55)).opened(5)
    e = first_start(tl)
    assert (e["code"], e["level"]) == ("D05", "A-02")
    assert det_time(e, 5) <= 2.0


def test_s09_look_into_lap_is_D05_not_D02_or_D04():
    # mata tampak tertutup saat menunduk dalam -> eye_valid False
    tl = Timeline().opened(5).ramp(0.3, "pitch", 0, -40).closed(3, pitch=-40, eye_valid=False) \
        .ramp(0.3, "pitch", -40, 0).opened(5)
    e = first_start(tl)
    assert (e["code"], e["level"]) == ("D05", "A-02")


def test_s11_alternating_is_D05():
    tl = Timeline().opened(5)
    for i in range(15):
        tl.add(0.7, dict(yaw=45 if i % 2 == 0 else -45)).opened(0.3)
    e = first_start(tl)
    assert (e["code"], e["level"]) == ("D05", "A-02")


def test_s12_phone_typing_head_down_is_D06_not_D05():
    tl = Timeline().opened(5, ).add(0.0)
    tl.add(5, yolo_valid=True)
    tl.add(3, dict(pitch=-30), phone_conf=0.7, yolo_valid=True).add(5, yolo_valid=True)
    e = first_start(tl)
    assert (e["code"], e["level"]) == ("D06", "A-02")


def test_s17_no_helmet_no_vest_single_D10():
    tl = Timeline().add(5, yolo_valid=True).add(15, no_helmet=True, no_vest=True, yolo_valid=True) \
        .add(5, yolo_valid=True)
    ev, _ = run_system(tl)
    s = starts(ev)
    assert len(s) == 1 and (s[0]["code"], s[0]["level"]) == ("D10", "A-02")


def _nod_session(depth=-28, down_s=0.6, hold_s=0.5, up_s=0.6, gap_s=2.0, n=8):
    tl = Timeline().opened(5)
    for _ in range(n):
        tl.ramp(down_s, "pitch", 0, depth).add(hold_s, dict(pitch=depth)).ramp(up_s, "pitch", depth, 0).opened(gap_s)
    return tl


def test_s18_nodding_is_D04():
    e = first_start(_nod_session())
    assert (e["code"], e["level"]) == ("D04", "A-03")


def test_s18_slow_nods_count_and_fast_jerks_do_not():
    # sim_P1c no 18: anggukan "perlahan" sampai ~2-3 s di bawah ambang (dulu > nod_max_s 2.0 -> terlewat)
    # kedalaman -22: di bawah ambang anggukan (-15), di atas ambang D-05 (-25); risiko anggukan dalam -> lihat bawah
    slow = dict(depth=-22, down_s=1.0, hold_s=1.6, up_s=1.0, n=4)          # ~2,2 s di bawah -15 per anggukan
    e = first_start(_nod_session(**slow))
    assert (e["code"], e["level"]) == ("D04", "A-03")
    assert first_start(_nod_session(**slow), thresholds({"D04_head_drop": {"nod_max_s": 2.0}})) is None
    # uji_yawn: sentakan kepala saat menguap 0.2-0.5 s -> bukan anggukan
    assert first_start(_nod_session(depth=-25, down_s=0.15, hold_s=0.1, up_s=0.15, n=6)) is None


def test_s18_slow_deep_nod_risk_and_mitigation():
    """Risiko K4: anggukan lambat dan dalam -> pitch < -25 lebih dari 1,5 s -> D-05 terkirim duluan.

    Spec-literal (pitch_only_extra_hold_s = 0) memang gagal di profil ini; opsi mitigasi di
    thresholds.yaml memperbaikinya. Kedua perilaku diuji supaya keputusannya eksplisit.
    """
    slow = dict(depth=-35, down_s=0.5, hold_s=1.2, up_s=0.5, gap_s=2.0)
    e_literal = first_start(_nod_session(**slow))
    assert e_literal["code"] == "D05"          # dokumentasi risiko (spec apa adanya)

    th = thresholds({"D05_distraction": {"pitch_only_extra_hold_s": 1.0}})
    e_mit = first_start(_nod_session(**slow), th)
    assert (e_mit["code"], e_mit["level"]) == ("D04", "A-03")


def test_s18_mitigation_keeps_s09_detected_within_scenario():
    th = thresholds({"D05_distraction": {"pitch_only_extra_hold_s": 1.0}})
    tl = Timeline().opened(5).add(3, dict(pitch=-40, eye_valid=False)).opened(5)
    e = first_start(tl, th)
    assert e["code"] == "D05"
    assert det_time(e, 5) <= 3.0               # masih di dalam 3 s skenario (tapi > 2 s)


def test_s19_head_roll_held_is_D04():
    tl = Timeline().opened(5).ramp(0.5, "roll", 0, 35).add(5, dict(roll=35)).opened(5)
    e = first_start(tl)
    assert (e["code"], e["level"]) == ("D04", "A-03")


def test_s20_lens_covered_is_D09():
    tl = Timeline().opened(5).add(15, dict(face_present=False), blocked_frame=True).opened(5)
    e = first_start(tl)
    assert (e["code"], e["level"]) == ("D09", "A-03")
    assert det_time(e, 5) <= 1.5


@pytest.mark.parametrize("blink_s", [0.6, 0.9, 1.2])
def test_s01_slow_blinks_are_D01_not_D02(blink_s):
    tl = Timeline().opened(5)
    for _ in range(12):
        tl.opened(4.0).closed(blink_s)
    e = first_start(tl)
    assert (e["code"], e["level"]) == ("D01", "A-02")


def test_s01_risk_one_blink_over_1_5s_becomes_D02():
    """Risiko K4 yang diketahui: satu pejaman > 1,5 s saat peragaan D-01 akan terbaca D-02.
    Tidak ada mitigasi otomatis — peraga harus diarahkan (catat di STATUS.md)."""
    tl = Timeline().opened(5).closed(1.7).opened(4)
    assert first_start(tl)["code"] == "D02"


def _normal_session(dash_glances: bool):
    """Duduk normal: kedip normal, lirik spion sesekali, (opsional) lirik dasbor ke bawah."""
    tl = Timeline()
    for i in range(60):   # ~6 menit
        tl.opened(2.5).closed(0.15)
        if i % 4 == 0:
            tl.add(0.8, dict(yaw=40))
        if dash_glances and i % 9 == 0:
            tl.add(0.6, dict(pitch=-20))
    return tl


def test_normal_driving_no_alarm():
    ev, _ = run_system(_normal_session(dash_glances=False))
    assert starts(ev) == []


def test_dashboard_glances_false_D04_risk_and_mitigation():
    """Risiko alarm palsu: dua lirikan ke dasbor (pitch -20 selama 0,6 s) dalam 30 s memenuhi definisi
    'anggukan' di spec (turun > 15 lalu kembali < 2 s, >= 2 kali) -> D-04 A-03 palsu.
    Opsi mitigasi: nod_count 3 (skenario 18 berisi banyak anggukan dalam 30 s, tetap terdeteksi)."""
    spec = thresholds({"D04_head_drop": {"nod_count": 2, "nod_active_hold_s": None, "nod_min_s": 0.0,
                                         "nod_max_s": 2.0}})
    ev, _ = run_system(_normal_session(dash_glances=True), spec)
    assert any(e["code"] == "D04" for e in starts(ev))          # spec-literal: alarm palsu
    ev, _ = run_system(_normal_session(dash_glances=True))      # config sekarang (nod_count 3)
    assert starts(ev) == []

    th = thresholds({"D04_head_drop": {"nod_count": 3}})
    ev, _ = run_system(_normal_session(dash_glances=True), th)
    assert starts(ev) == []
    e = first_start(_nod_session(), th)
    assert (e["code"], e["level"]) == ("D04", "A-03")
