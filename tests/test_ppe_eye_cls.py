"""A8/A9: geometri potongan, aturan gabung classifier APD, dan sinyal mata dari classifier pada frame cadangan pose."""
import numpy as np

from src.cabin.eye_cls import EyeClassifier
from src.cabin.face import FaceObs
from src.cabin.ppe_cls import PpeClassifier, cut, head_box, inside_frac_img, torso_box
from src.cabin.calibration import Calibration
from src.cabin.signals import SignalExtractor


def test_head_and_torso_boxes_from_face():
    fb = (100, 100, 160, 180)                     # u = 80
    assert np.allclose(head_box(fb), (130 - 72, 140 - 24 - 72, 130 + 72, 140 - 24 + 72))
    assert np.allclose(torso_box(fb), (130 - 96, 192, 130 + 96, 192 + 192))


def test_cut_pads_outside_with_grey_and_resizes():
    img = np.full((100, 100), 200, np.uint8)
    c = cut(img, (-50, -50, 50, 50), 20)
    assert c.shape == (20, 20)
    assert c[2, 2] == 114 and c[17, 17] == 200
    assert inside_frac_img((-50, -50, 50, 50), (100, 100)) == 0.25


def _ppe(mode):
    c = PpeClassifier({"enabled": False, "mode": mode})
    return c


def test_combine_modes():
    res = {"p_no_helmet": 0.8, "p_no_vest": None}
    assert _ppe("replace").combine(False, True, res) == (True, False)   # dada tak terlihat -> bukan pelanggaran
    assert _ppe("and").combine(False, True, res) == (False, True)
    assert _ppe("or").combine(False, True, res) == (True, True)
    assert _ppe("replace").combine(True, False, {"p_no_helmet": 0.1, "p_no_vest": 0.9}) == (False, True)


def test_eye_boxes_scale_with_eye_distance():
    e = EyeClassifier({"enabled": False, "side_d": 0.5, "min_eye_dist_px": 12})
    boxes = e.boxes([[100, 100], [140, 100]])
    assert np.allclose(boxes[0], (90, 90, 110, 110)) and np.allclose(boxes[1], (130, 90, 150, 90 + 20))
    assert e.boxes([[100, 100], [105, 100]]) == []
    assert e.p_closed(np.zeros((10, 10), np.uint8), [[1, 1], [9, 1]]) is None     # tanpa model


def _sig(p, prev=None):
    se = prev or SignalExtractor({"eye_classifier": {"enter_p": 0.6, "exit_p": 0.4, "yaw_max_deg": 30}},
                                 Calibration({}))
    obs = FaceObs(present=True, pose_only=True, yaw_raw=0.0, pitch_raw=float("nan"), roll_raw=0.0,
                  eye_closed_p=p)
    return se, se.update(obs, 0)


def test_pose_only_eye_state_from_classifier_with_hysteresis():
    se, s = _sig(None)
    assert not s.eye_valid and not s.eye_closed
    se, s = _sig(0.7, se)
    assert s.eye_valid and s.eye_closed and s.pose_only
    se, s = _sig(0.5, se)                          # di antara exit dan enter: tetap tertutup
    assert s.eye_closed and not s.eye_closed_strict
    se, s = _sig(0.3, se)
    assert not s.eye_closed
    se, s = _sig(0.5, se)                          # dari terbuka: 0,5 < enter -> tetap terbuka
    assert not s.eye_closed
