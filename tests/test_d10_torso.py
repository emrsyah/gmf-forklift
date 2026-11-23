"""D-10 rompi: torso hanya dinilai jika cukup terlihat; rompi kecil di dalam torso tetap dihitung."""
from src.backends import Detection
from src.cabin.yolo_cabin import interpret, torso_judgeable

D10 = {"conf": 0.40, "vest_iou_max": 0.3, "vest_inside_min": 0.6,
       "torso_min_h_over_w": 0.5, "torso_min_h_frac": 0.15}
FRAME = (960, 720)


def det(cls, box, conf=0.8):
    return Detection(0, cls, conf, box)


def test_shoulders_only_is_not_no_vest():
    # forklift 2026-11-23: torso 240 x 80 px di tepi bawah (hanya bahu) -> tidak bisa dinilai
    r = interpret([det("torso", (300, 640, 540, 720))], {}, D10, frame_wh=FRAME)
    assert r.no_vest is False


def test_full_torso_without_vest_is_no_vest():
    r = interpret([det("torso", (300, 400, 540, 720))], {}, D10, frame_wh=FRAME)
    assert r.no_vest is True


def test_small_vest_inside_torso_counts():
    # IoU rompi kecil vs torso besar < 0.3, tapi rompi seluruhnya di dalam torso -> ada rompi
    dets = [det("torso", (300, 400, 540, 720)), det("vest", (360, 480, 480, 600))]
    r = interpret(dets, {}, D10, frame_wh=FRAME)
    assert r.no_vest is False


def test_vest_elsewhere_does_not_count():
    dets = [det("torso", (300, 400, 540, 720)), det("vest", (700, 100, 800, 200))]
    assert interpret(dets, {}, D10, frame_wh=FRAME).no_vest is True


def test_small_torso_relative_to_frame_not_judged():
    # proporsi bagus tapi kecil sekali (orang jauh di belakang) -> tidak dinilai
    assert torso_judgeable((10, 10, 60, 90), 720, D10) is False
    assert torso_judgeable((10, 10, 160, 250), 720, D10) is True


def test_without_frame_size_only_shape_rule():
    assert torso_judgeable((0, 0, 100, 60), None, D10) is True
    assert torso_judgeable((0, 0, 100, 30), None, D10) is False


def test_helmet_rule_unchanged():
    r = interpret([det("head_no_helmet", (400, 100, 480, 200))], {}, D10, frame_wh=FRAME)
    assert r.no_helmet is True and r.no_vest is False
