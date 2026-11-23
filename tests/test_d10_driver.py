"""D-10: hanya pengemudi yang dinilai (jika wajahnya terlihat); helm vs kepala-tanpa-helm di kepala yang sama."""
from src.backends import Detection
from src.cabin.yolo_cabin import interpret

D10 = {"conf": 0.40, "vest_iou_max": 0.3, "vest_inside_min": 0.6, "torso_min_h_over_w": 0.5,
       "torso_min_h_frac": 0.15, "driver_only": True, "driver_head_face_inside_min": 0.5,
       "driver_torso_x_margin": 0.25, "helmet_conflict_iou": 0.3}
FRAME = (960, 720)
FACE = (440, 200, 520, 290)             # wajah pengemudi di tengah


def det(cls, box, conf=0.8):
    return Detection(0, cls, conf, box)


def test_bystander_without_helmet_ignored_when_driver_face_known():
    dets = [det("helmet", (420, 150, 540, 300)), det("head_no_helmet", (100, 200, 180, 290))]
    assert interpret(dets, {}, D10, face_box=FACE, frame_wh=FRAME).no_helmet is False


def test_driver_without_helmet_still_flagged():
    dets = [det("head_no_helmet", (425, 170, 535, 300))]
    assert interpret(dets, {}, D10, face_box=FACE, frame_wh=FRAME).no_helmet is True


def test_bystander_torso_without_vest_ignored():
    dets = [det("torso", (420, 300, 560, 640)), det("vest", (430, 320, 550, 600)),   # pengemudi: rompi
            det("torso", (40, 300, 200, 640))]                                       # orang lain: tanpa
    assert interpret(dets, {}, D10, face_box=FACE, frame_wh=FRAME).no_vest is False


def test_driver_torso_without_vest_flagged():
    dets = [det("torso", (400, 300, 580, 660))]
    assert interpret(dets, {}, D10, face_box=FACE, frame_wh=FRAME).no_vest is True


def test_no_face_falls_back_to_all_people():
    dets = [det("head_no_helmet", (100, 200, 180, 290))]
    assert interpret(dets, {}, D10, face_box=None, frame_wh=FRAME).no_helmet is True


def test_helmet_beats_weaker_no_helmet_on_same_head():
    dets = [det("helmet", (425, 170, 535, 300), 0.7), det("head_no_helmet", (430, 175, 535, 300), 0.5)]
    assert interpret(dets, {}, D10, face_box=FACE, frame_wh=FRAME).no_helmet is False


def test_stronger_no_helmet_beats_helmet():
    dets = [det("helmet", (425, 170, 535, 300), 0.45), det("head_no_helmet", (430, 175, 535, 300), 0.8)]
    assert interpret(dets, {}, D10, face_box=FACE, frame_wh=FRAME).no_helmet is True
