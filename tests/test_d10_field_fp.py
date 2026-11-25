"""D-10 aturan dari uji forklift 2026-11-25: veto helm berskor rendah, kepala terpotong tepi atas,
rentang x kursi pengemudi, dan tanpa wajah pengemudi = tidak dinilai."""
from src.backends import Detection
from src.cabin.yolo_cabin import interpret

BASE = {"conf": 0.40, "vest_iou_max": 0.3, "vest_inside_min": 0.6, "torso_min_h_over_w": 0.5,
        "torso_min_h_frac": 0.15, "helmet_conflict_iou": 0.3}
FRAME = (960, 540)


def det(cls, box, conf=0.8):
    return Detection(0, cls, conf, box)


def test_weak_helmet_covering_head_vetoes_no_helmet():
    # snapshot 13:24:20: head_no_helmet 0.61 di bawah kotak helm 0.19 (IoU 0.23, menutupi 41% kepala)
    dets = [det("head_no_helmet", (444, 167, 557, 309), 0.61), det("helmet", (399, 112, 569, 226), 0.19)]
    on = dict(BASE, helmet_veto_conf=0.15, helmet_veto_cover=0.3)
    assert interpret(dets, {}, BASE, frame_wh=FRAME).no_helmet is True       # tanpa veto: palsu
    assert interpret(dets, {}, on, frame_wh=FRAME).no_helmet is False


def test_weak_helmet_elsewhere_does_not_veto():
    dets = [det("head_no_helmet", (444, 167, 557, 309), 0.61), det("helmet", (100, 100, 200, 200), 0.19)]
    on = dict(BASE, helmet_veto_conf=0.15, helmet_veto_cover=0.3)
    assert interpret(dets, {}, on, frame_wh=FRAME).no_helmet is True


def test_helmet_below_veto_conf_ignored():
    dets = [det("head_no_helmet", (444, 167, 557, 309), 0.61), det("helmet", (399, 112, 569, 226), 0.10)]
    on = dict(BASE, helmet_veto_conf=0.15, helmet_veto_cover=0.3)
    assert interpret(dets, {}, on, frame_wh=FRAME).no_helmet is True


def test_head_touching_top_edge_not_judged():
    dets = [det("head_no_helmet", (432, 2, 598, 145), 0.6)]                  # puncak helm di luar gambar
    on = dict(BASE, head_top_margin_frac=0.04)
    assert interpret(dets, {}, on, frame_wh=FRAME).no_helmet is False
    dets = [det("head_no_helmet", (432, 60, 598, 200), 0.6)]                 # seluruh kepala terlihat
    assert interpret(dets, {}, on, frame_wh=FRAME).no_helmet is True


def test_person_outside_driver_x_range_ignored():
    on = dict(BASE, driver_x_range=[0.2, 0.8])
    side = [det("head_no_helmet", (791, 256, 883, 373)), det("torso", (760, 300, 940, 540))]   # x ~0,87
    assert interpret(side, {}, on, frame_wh=FRAME).no_helmet is False
    assert interpret(side, {}, on, frame_wh=FRAME).no_vest is False
    seat = [det("head_no_helmet", (440, 60, 540, 180))]                                        # x ~0,51
    assert interpret(seat, {}, on, frame_wh=FRAME).no_helmet is True


def test_require_driver_face_without_face_not_judged():
    dets = [det("torso", (386, 98, 631, 377))]                               # punggung, rompi tak dikenali
    on = dict(BASE, require_driver_face=True)
    assert interpret(dets, {}, BASE, face_box=None, frame_wh=FRAME).no_vest is True
    assert interpret(dets, {}, on, face_box=None, frame_wh=FRAME).no_vest is False
    face = (470, 110, 550, 200)
    front = [det("torso", (400, 200, 620, 540))]
    assert interpret(front, {}, on, face_box=face, frame_wh=FRAME).no_vest is True
