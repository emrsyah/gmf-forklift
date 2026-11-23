"""A7: decode pose, pilih pengemudi, potongan pergelangan tangan -> kandidat ponsel tambahan."""
import numpy as np

from src.backends import Detection
from src.backends.pose import L_ELBOW, L_SHOULDER, L_WRIST, NOSE, R_SHOULDER, R_WRIST, Person, decode_pose
from src.cabin.wrist_crop import WristCrop, pick_driver


def _person(box, kp: dict, conf=0.9):
    k = np.zeros((17, 3), np.float32)
    for i, (x, y, c) in kp.items():
        k[i] = (x, y, c)
    return Person(conf, box, k)


def test_decode_pose_maps_letterbox_back():
    out = np.zeros((1, 56, 3), np.float32)
    out[0, :5, 0] = [100, 120, 40, 80, 0.8]            # cx, cy, w, h, conf
    out[0, 5 + 3 * L_WRIST: 8 + 3 * L_WRIST, 0] = [110, 140, 0.9]
    out[0, 4, 1] = 0.1                                  # di bawah conf
    people = decode_pose(out, (400, 400, 3), r=0.5, pad=(10, 20), conf=0.3)
    assert len(people) == 1
    p = people[0]
    assert np.allclose(p.xyxy, ((80 - 10) / 0.5, (80 - 20) / 0.5, (120 - 10) / 0.5, (160 - 20) / 0.5))
    assert np.allclose(p.kpts[L_WRIST], [(110 - 10) / 0.5, (140 - 20) / 0.5, 0.9])


def test_pick_driver_uses_face_then_largest():
    small = _person((0, 0, 100, 200), {NOSE: (50, 40, 0.9)})
    big = _person((200, 0, 600, 400), {NOSE: (400, 60, 0.9)})
    assert pick_driver([small, big], face_box=(30, 20, 70, 60)) is small
    assert pick_driver([small, big]) is big
    assert pick_driver([]) is None


def test_boxes_centre_ahead_of_wrist_and_skip_low_conf():
    wc = WristCrop({"enabled": True, "side_u": 2.0, "ahead": 0.5, "kp_conf": 0.5, "min_side_px": 10})
    p = _person((0, 0, 640, 480), {L_WRIST: (300, 300, 0.9), L_ELBOW: (300, 200, 0.9),
                                   R_WRIST: (100, 300, 0.2)})
    boxes = wc.boxes(p, face_box=(280, 50, 340, 110), frame_wh=(640, 480))     # u = 60
    assert boxes == [(240, 290, 360, 410)]              # pusat (300, 350), sisi 120


def test_unit_from_shoulders_without_face():
    wc = WristCrop({"enabled": True, "shoulder_to_u": 0.5})
    p = _person((0, 0, 400, 400), {L_SHOULDER: (100, 100, 0.9), R_SHOULDER: (300, 100, 0.9)})
    assert wc.unit(p, None) == 100.0


class _Pose:
    def __init__(self, people):
        self.people = people

    def predict(self, img, conf=0.3):
        return self.people


class _Det:
    def __init__(self):
        self.calls = []

    def predict(self, img, conf=0.25):
        self.calls.append(img.shape)
        return [Detection(0, "phone", 0.6, (10, 10, 30, 40)), Detection(4, "torso", 0.9, (0, 0, 50, 50))]


def test_run_adds_phone_in_frame_coords_only():
    wc = WristCrop({"enabled": True, "side_u": 2.0, "ahead": 0.0, "min_side_px": 10})
    p = _person((0, 0, 640, 480), {L_WRIST: (300, 300, 0.9)})
    det = _Det()
    img = np.zeros((480, 640, 3), np.uint8)
    full = [Detection(1, "helmet", 0.8, (0, 0, 20, 20))]
    out = wc.run(det, _Pose([p]), img, full, (280, 50, 340, 110), 0.35)
    assert det.calls == [(120, 120, 3)]                 # u = 60 (wajah), sisi 2u
    phones = [d for d in out if d.cls_name == "phone"]
    assert [d.xyxy for d in phones] == [(250, 250, 270, 280)]
    assert not any(d.cls_name == "torso" for d in out)
    assert wc.last_boxes == [(240, 240, 360, 360)]


def test_run_without_person_or_disabled_returns_full():
    full = [Detection(1, "helmet", 0.8, (0, 0, 20, 20))]
    img = np.zeros((480, 640, 3), np.uint8)
    assert WristCrop({"enabled": True}).run(_Det(), _Pose([]), img, full, None, 0.35) == full
    assert WristCrop({"enabled": False}).run(_Det(), _Pose([]), img, full, None, 0.35) == full
