"""Dekode JPEG kamera depan di ukuran 1/N."""
import cv2
import numpy as np

from src.capture import Frame
from src.capture.uvch264 import decode_frame, decode_jpeg


def _jpeg():
    img = np.zeros((720, 1280, 3), np.uint8)
    cv2.rectangle(img, (600, 300), (700, 500), (255, 255, 255), -1)
    return cv2.imencode(".jpg", img)[1].tobytes()


def test_reduce_halves_size_and_keeps_content():
    full, half = decode_jpeg(_jpeg()), decode_jpeg(_jpeg(), 2)
    assert full.shape == (720, 1280, 3) and half.shape == (360, 640, 3)
    assert half[200, 325].mean() > 200 and half[50, 50].mean() < 30


def test_decode_frame_reduce_only_for_raw_jpeg():
    raw = Frame(None, 0, 0, meta={"jpeg": _jpeg()})
    assert decode_frame(raw, 2).shape[:2] == (360, 640)
    decoded = Frame(np.zeros((720, 1280, 3), np.uint8), 0, 0)
    assert decode_frame(decoded, 2).shape[:2] == (720, 1280)
