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


class _FakeWeb:
    def __init__(self):
        self.want = False
        self.got = []

    def add_channel(self, *a):
        pass

    def wants(self, ch):
        return self.want

    def publish(self, img, ch):
        self.got.append(img.shape)


class _Sink:
    def __init__(self):
        self.aus = []

    def push(self, au, ts, idr=None):
        self.aus.append((au, ts, idr))

    def close(self):
        pass


class _FakeSrc:
    def __init__(self, frames):
        self.frames = frames

    def __iter__(self):
        return iter(self.frames)

    def __enter__(self):
        return self

    def __exit__(self, *a):
        pass


def _jpeg(h=4, w=6):
    import cv2
    return cv2.imencode(".jpg", np.full((h, w, 3), 100, np.uint8))[1].tobytes()


def test_front_preview_passes_h264_without_decoding():
    from src.capture import Frame
    from src.io.front_preview import FrontPreview
    jpg = _jpeg()
    frames = [Frame(None, 1000 + 33 * i, i, 0.0, {"jpeg": jpg, "h264": b"au%d" % i, "h264_idr": i == 0})
              for i in range(10)]
    web, sink = _FakeWeb(), _Sink()
    calls = []

    def factory(cfg):
        calls.append(cfg)
        if len(calls) > 1:
            raise RuntimeError("selesai")
        return _FakeSrc(frames)

    fp = FrontPreview({"type": "webcam", "h264_mux": {"enabled": True}}, {"rotate": 270, "retry_s": 60},
                      web, None, [sink], source_factory=factory)
    t0 = time.monotonic()
    while fp.frames < 10 and time.monotonic() - t0 < 3:
        time.sleep(0.01)
    assert calls[0]["h264_mux"]["decode"] is False           # pratinjau: sumber tidak mendekode
    assert [a for a, _, _ in sink.aus] == [b"au%d" % i for i in range(10)]
    assert fp.decoded == 0 and web.got == []                 # tanpa penonton: tidak ada dekode sama sekali
    web.want = True
    fp._handle(frames[0])
    assert fp.decoded == 1 and web.got == [(6, 4, 3)]        # dekode + putar hanya saat ditonton
    fp.close()


def test_front_preview_mjpg_fallback_decodes_only_sampled_frames(tmp_path):
    from src.capture import Frame
    from src.io.continuous import ContinuousRecorder
    from src.io.front_preview import FrontPreview
    jpg = _jpeg()
    rec = ContinuousRecorder({"dir": str(tmp_path), "fps": 10, "min_free_gb": 0, "max_side": 0})
    fp = FrontPreview({"type": "webcam"}, {"rotate": 0, "retry_s": 60}, None, rec, [_Sink()],
                      source_factory=lambda cfg: (_ for _ in ()).throw(RuntimeError("tidak dipakai")))
    for i in range(30):                                       # 1 s @ 30 fps, kamera TANPA H.264
        fp._handle(Frame(None, 1000 + 33 * i, i, 0.0, {"jpeg": jpg}))
    assert 9 <= fp.decoded <= 11                              # ~10 fps rekaman, bukan 30 dekode
    fp.close()
    assert rec.frames == fp.decoded


def test_front_preview_falls_back_to_mjpg_when_h264_recorder_stops(tmp_path):
    from src.capture import Frame
    from src.io.continuous import ContinuousRecorder
    from src.io.front_preview import FrontPreview

    class _H264Rec(_Sink):
        records = True
        stopped_reason = None

    jpg = _jpeg()
    h264rec = _H264Rec()
    rec = ContinuousRecorder({"dir": str(tmp_path), "fps": 10, "min_free_gb": 0, "max_side": 0})
    fp = FrontPreview({"type": "webcam"}, {"retry_s": 60}, None, rec, [h264rec],
                      source_factory=lambda cfg: (_ for _ in ()).throw(RuntimeError("tidak dipakai")))
    fr = lambda i: Frame(None, 1000 + 100 * i, i, 0.0, {"jpeg": jpg, "h264": b"au", "h264_idr": False})  # noqa: E731
    for i in range(5):
        fp._handle(fr(i))
    assert len(h264rec.aus) == 5 and fp.decoded == 0          # H.264 jalan: MJPG diam
    h264rec.stopped_reason = "ffmpeg mati"
    for i in range(5, 10):
        fp._handle(fr(i))
    assert len(h264rec.aus) == 5 and fp.decoded == 5          # perekam H.264 berhenti -> MJPG ambil alih
    fp.close()
    assert rec.frames == 5


def test_web_view_channels():
    v = WebView(0, queue.Queue(), fps=0)
    try:
        v.add_channel("front", "Kamera depan")
        assert v.channel_for("/stream") == "cabin"
        assert v.channel_for("/stream/front") == "front"
        assert v.channel_for("/stream/x") is None
        v.publish(np.zeros((8, 8, 3), np.uint8), "front")
        assert v._jpg["front"] is None                 # belum ada penonton
        v._viewers["front"] = 1
        v.publish(np.zeros((8, 8, 3), np.uint8), "front")
        assert v._jpg["front"] and v._jpg["cabin"] is None and v._seq["front"] == 1
    finally:
        v.close()
