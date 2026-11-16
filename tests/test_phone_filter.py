"""Filter ponsel D-06 (src/cabin/phone_filter.py) dengan geometri sintetis.

Frame 640x480, kotak wajah landmark (260,120)-(380,270): u = 150 px, pusat (320,195).
Lembar A 12 (mengetik, dada/pangkuan), 13 (telinga), 14 (setinggi kemudi, sebagian tertutup tangan) harus
lolos dan memicu D-06 < 2 s; pola palsu dari log webcam (di dalam wajah, kecil di tepi frame, jauh di
samping, berkedip di tempat acak) harus ditolak.
"""
import pytest

from src.backends import Detection
from src.cabin.indicators import D06Phone, IndicatorInput
from src.cabin.phone_filter import PhoneFilter, face_box_from_landmarks
from src.cabin.yolo_cabin import interpret
from tests.synth import sig, thresholds

TH = thresholds()
FCFG = TH["D06_phone"]["filter"]
FACE = (260.0, 120.0, 380.0, 270.0)
WH = (640, 480)

EAR = (365, 142, 410, 262)            # skenario 13: di tepi kanan wajah, tegak, rasio ~2,7
TYPING = (267, 395, 402, 475)         # skenario 12: dada/pangkuan, mendatar
STEERING_COVERED = (196, 392, 264, 447)  # skenario 14: kecil, hampir persegi (sebagian tertutup tangan)
IN_FACE = (275, 210, 335, 255)        # mulut/dagu/janggut
EDGE_SMALL = (5, 340, 47, 383)        # diag4: kecil di tepi kiri frame
FAR_SIDE = (0, 300, 60, 380)          # pusat 2,7 u di kiri wajah (SMALL_FACE)
ABOVE_HEAD = (300, 0, 360, 40)        # di atas kepala


def phone(box, conf=0.6):
    return Detection(0, "phone", conf, tuple(float(v) for v in box))


def det(name, box, conf=0.8):
    return Detection(1, name, conf, tuple(float(v) for v in box))


def run(f, boxes_per_run, face=FACE, yaw=0.0, start_ms=0, period_ms=67, extra=()):
    """Jalankan filter beberapa run YOLO; kembalikan verdict terakhir."""
    v = None
    for i, boxes in enumerate(boxes_per_run):
        dets = [phone(b) if not isinstance(b, Detection) else b for b in boxes] + list(extra)
        v = f.update(dets, start_ms + i * period_ms, face, yaw, WH)
    return v


def new():
    return PhoneFilter(FCFG)


# ------------------------------------------------------------------ Lembar A 12-14 lolos
@pytest.mark.parametrize("box", [EAR, TYPING, STEERING_COVERED], ids=["s13_ear", "s12_typing", "s14_steering"])
def test_lembar_a_geometries_pass(box):
    v = run(new(), [[box]] * 3)
    assert v.conf == pytest.approx(0.6)
    assert v.ref == "face"


def test_typing_when_face_lost_uses_face_memory():
    # skenario 12: menunduk ke ponsel -> MediaPipe kehilangan wajah; kotak wajah terakhir dipakai
    f = new()
    f.update([], 0, FACE, 0.0, WH)                      # wajah terlihat sebelum menunduk
    v = run(f, [[TYPING]] * 3, face=None, start_ms=1000)
    assert v.conf > 0 and v.ref == "face_memory"


def test_face_lost_long_falls_back_to_torso_then_passes():
    f = new()
    f.update([], 0, FACE, 0.0, WH)
    torso = det("torso", (200, 300, 470, 480))           # u ~ 150, wajah ~ y 195
    v = run(f, [[TYPING]] * 3, face=None, start_ms=60_000, extra=[torso])
    assert v.conf > 0 and v.ref == "torso"


def test_head_box_reference_when_no_face():
    head = det("head_no_helmet", (250, 100, 390, 280))
    v = run(new(), [[EAR]] * 3, face=None, extra=[head])
    assert v.conf > 0 and v.ref == "head"


def test_no_reference_passes_only_high_conf():
    f = new()
    v = run(f, [[phone(TYPING, 0.45)]] * 3, face=None)
    assert v.conf == 0 and v.reason == "low_conf"
    v = run(new(), [[phone(TYPING, 0.7)]] * 3, face=None)
    assert v.conf == pytest.approx(0.7) and v.ref == "none"


def test_profile_view_disables_inside_face_rule():
    # kepala menoleh (profil): telinga + ponsel bisa tampak di tengah kotak wajah
    v = run(new(), [[IN_FACE]] * 3, yaw=50.0)
    assert v.conf > 0


# ------------------------------------------------------------------ pola palsu ditolak
SMALL_FACE = (290.0, 150.0, 370.0, 250.0)   # u = 100 (pengemudi lebih jauh) -> FAR_SIDE di dx -2,7 u


@pytest.mark.parametrize("box,reason,face", [(IN_FACE, "inside_face", FACE), (EDGE_SMALL, "too_small", FACE),
                                             (FAR_SIDE, "region_x", SMALL_FACE), (ABOVE_HEAD, "region_y", FACE)],
                         ids=["in_face", "edge_small", "far_side", "above_head"])
def test_false_patterns_rejected(box, reason, face):
    v = run(new(), [[box]] * 5, face=face)
    assert v.conf == 0
    assert v.reason == reason


def test_too_big_and_aspect_rejected():
    assert run(new(), [[(0, 100, 640, 480)]] * 3).reason == "too_big"
    assert run(new(), [[(300, 300, 312, 420)]] * 3).reason == "aspect"   # 120 x 12 -> rasio 10


def test_flicker_at_random_places_never_stable():
    a, b = TYPING, EAR                                   # bergantian, berjauhan > max_jump_u
    v_all = [run(new(), [[a], [b]] * 4)]
    f = new()
    out = [f.update([phone(x)], i * 67, FACE, 0.0, WH).conf for i, x in enumerate([a, b] * 6)]
    assert max(out) == 0 and v_all[0].conf == 0


def test_single_run_not_enough_then_gap_resets():
    f = new()
    assert f.update([phone(TYPING)], 0, FACE, 0.0, WH).conf == 0          # hit 1
    assert f.update([phone(TYPING)], 67, FACE, 0.0, WH).conf > 0          # hit 2 -> stabil
    assert f.update([], 1000, FACE, 0.0, WH).conf == 0                    # jeda > 0,5 s
    assert f.update([phone(TYPING)], 1067, FACE, 0.0, WH).conf == 0       # jejak baru, hit 1


def test_short_miss_keeps_track():
    # ponsel tertutup tangan sesaat (1 run YOLO tanpa deteksi) tidak mereset jejak
    f = new()
    for t in (0, 67):
        f.update([phone(STEERING_COVERED)], t, FACE, 0.0, WH)
    f.update([], 134, FACE, 0.0, WH)
    assert f.update([phone(STEERING_COVERED)], 201, FACE, 0.0, WH).conf > 0


def test_disabled_filter_is_old_behaviour():
    f = PhoneFilter({**FCFG, "enabled": False})
    assert f.update([phone(IN_FACE, 0.5), phone(EDGE_SMALL, 0.4)], 0, FACE, 0.0, WH).conf == pytest.approx(0.5)


def test_face_box_from_landmarks():
    import numpy as np
    assert face_box_from_landmarks(np.array([[1, 2], [5, 9], [3, 4]])) == (1, 2, 5, 9)
    assert face_box_from_landmarks(None) is None


# ------------------------------------------------------------------ D-06 end-to-end (latensi)
def _d06_first_active(box, face=FACE, period_frames=2, fps=30, dur_s=3.0, t_phone_s=0.5, conf=0.6):
    """30 FPS, YOLO tiap 2 frame (hasil dipakai ulang di frame lain), ponsel muncul di t_phone_s."""
    f, ind = new(), D06Phone(TH["D06_phone"])
    last = None
    for n in range(int(dur_s * fps)):
        ts = int(n * 1000 / fps)
        if n % period_frames == 0:
            dets = [phone(box, conf)] if ts >= t_phone_s * 1000 else []
            last = interpret(dets, TH["D06_phone"], TH["D10_ppe"], phone_filter=f, ts_ms=ts,
                             face_box=face, yaw=0.0, frame_wh=WH)
        x = IndicatorInput(ts_ms=ts, sig=sig(ts_ms=ts), phone_conf=last.phone_conf, yolo_valid=True)
        if ind.update(x):
            return ts / 1000 - t_phone_s
    return None


@pytest.mark.parametrize("box", [EAR, TYPING, STEERING_COVERED], ids=["s13_ear", "s12_typing", "s14_steering"])
def test_d06_fires_within_2s_for_lembar_a(box):
    dt = _d06_first_active(box)
    assert dt is not None and dt <= 2.0


def test_d06_typing_face_lost_fires_within_2s():
    f, ind = new(), D06Phone(TH["D06_phone"])
    f.update([], 0, FACE, 0.0, WH)                       # wajah terlihat, lalu hilang saat menunduk
    first = None
    for n in range(90):
        ts = 500 + int(n * 1000 / 30)
        if n % 2 == 0:
            last = interpret([phone(TYPING)], TH["D06_phone"], TH["D10_ppe"], phone_filter=f, ts_ms=ts,
                             face_box=None, yaw=None, frame_wh=WH)
        if ind.update(IndicatorInput(ts_ms=ts, sig=sig(ts_ms=ts, face_present=False),
                                     phone_conf=last.phone_conf, yolo_valid=True)):
            first = (ts - 500) / 1000
            break
    assert first is not None and first <= 2.0


def test_d06_never_fires_for_in_face_false_positive():
    assert _d06_first_active(IN_FACE, conf=0.8) is None
    assert _d06_first_active(EDGE_SMALL, conf=0.8) is None
