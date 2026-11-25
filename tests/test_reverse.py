"""Deteksi mundur dari kamera depan (egomotion) dan D-05 yang tidak menghitung menoleh saat mundur."""
import cv2
import numpy as np

from src.cabin.indicators import D05Distraction, IndicatorInput
from src.front.egomotion import EgoMotion, EgoSample, ReverseState
from tests.synth import sig


def _texture(seed=0, h=360, w=640):
    rng = np.random.default_rng(seed)
    img = (rng.random((h // 8, w // 8)) * 255).astype(np.uint8)
    return cv2.cvtColor(cv2.resize(img, (w, h), interpolation=cv2.INTER_NEAREST), cv2.COLOR_GRAY2BGR)


def _zoom(img, s):
    h, w = img.shape[:2]
    M = cv2.getRotationMatrix2D((w / 2, h / 2), 0, s)
    return cv2.warpAffine(img, M, (w, h), borderMode=cv2.BORDER_REFLECT)


def test_scene_growing_is_forward_shrinking_is_reverse():
    base = _texture()
    for step, sign in ((1.01, 1), (1 / 1.01, -1)):
        em = EgoMotion({})
        em.update(base, 0)
        s = em.update(_zoom(base, step), 100)              # 10 fps
        assert s.v is not None and np.sign(s.v) == sign and abs(s.v) > 0.05


def test_sideways_shift_is_not_reverse():
    base = _texture(1)
    em = EgoMotion({})
    em.update(base, 0)
    s = em.update(np.roll(base, 6, axis=1), 100)          # belok: geser mendatar
    assert s.v is not None and abs(s.v) < 0.02


def _feed(rs, vs, t0=0, dt=100):
    out = []
    for i, v in enumerate(vs):
        out.append(rs.update(EgoSample(t0 + i * dt, v, 50)))
    return out


def test_reverse_state_needs_sustained_negative():
    rs = ReverseState({})
    assert not any(_feed(rs, [-0.3, 0.0, 0.05] * 10))     # lonjakan sesaat (orang lewat): median ~0
    rs = ReverseState({})
    out = _feed(rs, [-0.1] * 20)
    assert not out[5] and out[-1]                         # masuk setelah ~0.7 s (+ jendela median)
    out = _feed(rs, [0.0] * 30, t0=2000)
    assert out[5] and not out[-1]                         # keluar setelah ~1.5 s diam


def _sig(yaw, pitch=0.0):
    return sig(yaw=yaw, pitch=pitch)


def _run_d05(reversing, yaw=60.0, pitch=0.0, secs=3.0):
    ind = D05Distraction({"yaw_deg": 30, "pitch_down_deg": 25, "continuous_s": 1.5, "window_s": 6,
                          "ratio": 0.5, "min_window_fill_s": 3.0, "suppress_when_reversing": True})
    hit = False
    for i in range(int(secs * 30)):
        hit |= ind.update(IndicatorInput(i * 33, _sig(yaw, pitch), reversing=reversing))
    return hit


def test_d05_head_turn_while_reversing_not_counted():
    assert _run_d05(False) is True
    assert _run_d05(True) is False


def test_d05_looking_down_while_reversing_still_counted():
    assert _run_d05(True, yaw=0.0, pitch=-40.0) is True
