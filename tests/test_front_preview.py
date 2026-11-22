import queue
import time

import numpy as np
import pytest

from src.io.front_preview import rotate
from src.io.web_view import WebView


def test_rotate():
    img = np.zeros((2, 4, 3), np.uint8)
    img[0, 0] = 255                          # pojok kiri atas
    r = rotate(img, 90)                      # searah jarum jam: kiri atas -> kanan atas
    assert r.shape == (4, 2, 3) and r[0, 1, 0] == 255
    assert rotate(img, 0) is img
    with pytest.raises(ValueError):
        rotate(img, 45)


def test_web_view_channels():
    v = WebView(0, queue.Queue(), fps=0)
    try:
        v.add_channel("front", "Kamera depan")
        assert v.channel_for("/stream") == "cabin"
        assert v.channel_for("/stream/front") == "front"
        assert v.channel_for("/stream/x") is None
        v.publish(np.zeros((8, 8, 3), np.uint8), "front")
        assert v._jpg["front"] and v._jpg["cabin"] is None and v._seq["front"] == 1
    finally:
        v.close()
