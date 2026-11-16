"""Pass kedua YOLO ponsel di potongan area pengemudi."""
import numpy as np

from src.backends import Detection
from src.cabin.phone_crop import PhoneCrop

CFG = {"enabled": True, "left_u": 2.6, "right_u": 2.6, "up_u": 1.2, "down_u": 4.3, "min_gain": 1.25,
       "min_side_px": 160, "face_memory_s": 5.0, "merge_iou": 0.5, "classes": ["phone"]}


class FakeDetector:
    """Mengembalikan deteksi dalam koordinat gambar yang diterimanya; mencatat ukuran input."""

    def __init__(self, dets):
        self.dets, self.shapes = dets, []

    def predict(self, img, conf=0.25):
        self.shapes.append(img.shape)
        return [Detection(d.cls_id, d.cls_name, d.conf, d.xyxy) for d in self.dets]


def test_crop_box_from_face_units_and_clamped():
    pc = PhoneCrop(CFG)
    # wajah 100 px di tengah atas frame 1640x1232
    b = pc.box((770, 200, 870, 300), (1640, 1232), 0)
    assert b == (560, 130, 1080, 680)   # cx 820 +- 260, cy 250 - 120 .. +430


def test_no_crop_when_it_covers_almost_whole_frame():
    pc = PhoneCrop(CFG)
    # wajah besar (webcam dekat): potongan ~ seluruh frame -> tidak ada keuntungan
    assert pc.box((220, 100, 420, 300), (640, 480), 0) is None


def test_face_memory_then_expiry():
    pc = PhoneCrop(CFG)
    assert pc.box((770, 200, 870, 300), (1640, 1232), 0) is not None
    assert pc.box(None, (1640, 1232), 4000) is not None       # menunduk ke ponsel: wajah hilang < 5 s
    assert pc.box(None, (1640, 1232), 6000) is None


def test_crop_dets_mapped_back_only_phone_and_merged():
    pc = PhoneCrop(CFG)
    img = np.zeros((1232, 1640, 3), np.uint8)
    full = [Detection(3, "vest", 0.8, (600, 400, 1000, 900))]
    det = FakeDetector([Detection(0, "phone", 0.7, (100, 200, 140, 280)),
                        Detection(1, "helmet", 0.9, (0, 0, 50, 50))])
    out = pc.run(det, img, full, (770, 200, 870, 300), 0, 0.35)
    assert det.shapes[0][:2] == (550, 520)                    # potongan 520x550, bukan frame penuh
    phones = [d for d in out if d.cls_name == "phone"]
    assert len(phones) == 1 and phones[0].xyxy == (660, 330, 700, 410)   # + offset (560, 130)
    assert not any(d.cls_name == "helmet" for d in out)       # kelas lain dari potongan dibuang
    assert any(d.cls_name == "vest" for d in out)


def test_duplicate_keeps_higher_conf():
    pc = PhoneCrop(CFG)
    full = [Detection(0, "phone", 0.40, (660, 390, 700, 470))]
    out = pc.merge(full, [Detection(0, "phone", 0.80, (662, 392, 701, 471))])
    assert len(out) == 1 and out[0].conf == 0.80


def test_disabled_is_passthrough():
    pc = PhoneCrop({**CFG, "enabled": False})
    det = FakeDetector([Detection(0, "phone", 0.7, (1, 1, 2, 2))])
    full = []
    assert pc.run(det, np.zeros((100, 100, 3), np.uint8), full, (10, 10, 20, 20), 0, 0.3) == []
    assert det.shapes == []
