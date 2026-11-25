"""Tukar/putar kamera dari dashboard (camera_config.json) dan RotatedSource."""
import json

import numpy as np

from src.capture import Frame, FrameSource, RotatedSource
from src.capture.camera_settings import apply_camera_settings

CAB, FRONT = "/dev/v4l/by-id/cam-A", "/dev/v4l/by-id/cam-B"


def _rt(tmp_path, data=None):
    f = tmp_path / "camera_config.json"
    if data is not None:
        f.write_text(json.dumps(data))
    return {
        "camera_settings": {"file": str(f), "cabin_device": CAB, "front_device": FRONT},
        # editan agen di jetson.yaml (sudah ditukar) tidak boleh menentukan hasil: file JSON yang menentukan
        "cabin": {"source": {"device": FRONT}},
        "front": {"source": {"device": CAB}},
        "front_preview": {"rotate": 0},
    }


def test_swap_and_rotate_from_file(tmp_path):
    rt = _rt(tmp_path, {"swapped": True, "cabinRotate": 180, "frontRotate": 270})
    note = apply_camera_settings(rt)
    assert rt["cabin"]["source"]["device"] == FRONT and rt["front"]["source"]["device"] == CAB
    assert rt["cabin"]["source"]["rotate"] == 180 and rt["front_preview"]["rotate"] == 270
    assert "putar kabin 180" in note


def test_not_swapped_restores_canonical_devices(tmp_path):
    # deploy dari git / editan agen tertinggal: swapped false -> path dari camera_settings, bukan dari source
    rt = _rt(tmp_path, {"swapped": False})
    apply_camera_settings(rt)
    assert rt["cabin"]["source"]["device"] == CAB and rt["front"]["source"]["device"] == FRONT
    assert "rotate" not in rt["cabin"]["source"]


def test_missing_file_and_bad_values(tmp_path):
    rt = _rt(tmp_path)                       # file belum ada (dashboard belum pernah mengatur)
    apply_camera_settings(rt)
    assert rt["cabin"]["source"]["device"] == CAB
    rt = _rt(tmp_path, {"cabinRotate": 45})
    note = apply_camera_settings(rt)
    assert "rotate" not in rt["cabin"]["source"] and "diabaikan" in note
    assert apply_camera_settings({"cabin": {"source": {}}}) is None   # fitur mati (laptop)


class _Src(FrameSource):
    def __init__(self, img):
        self.img, self.n = img, 0

    def read(self):
        self.n += 1
        return Frame(self.img.copy(), 1000 * self.n, self.n) if self.n <= 2 else None

    def clock_correction_ms(self):
        return 7


def test_rotated_source_clockwise():
    img = np.zeros((2, 3, 3), np.uint8)
    img[0, 0] = 255                          # piksel kiri atas
    src = RotatedSource(_Src(img), 90)
    out = src.read().image
    assert out.shape[:2] == (3, 2) and out[0, 1, 0] == 255    # searah jarum jam: kiri atas -> kanan atas
    assert src.clock_correction_ms() == 7
    assert RotatedSource(_Src(img), 180).read().image[1, 2, 0] == 255
