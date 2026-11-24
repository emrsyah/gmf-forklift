import time

import numpy as np

from src.io.overlay_stream import OverlayStreams, VideoFanout, even_size, ffmpeg_cmd


class FakeConn:
    def __init__(self, url, w, h, fail_after=None):
        self.url, self.w, self.h = url, w, h
        self.frames = []
        self.fail_after = fail_after
        self.closed = False

    def alive(self):
        return not self.closed

    def write(self, data):
        if self.fail_after is not None and len(self.frames) >= self.fail_after:
            raise BrokenPipeError("server tutup")
        self.frames.append(data)

    def close(self, timeout_s=0):
        self.closed = True


def _wait(cond, t=2.0):
    end = time.monotonic() + t
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(0.01)
    return False


def test_even_size_limits_longest_side_and_is_even():
    assert even_size(1280, 720, 640) == (640, 360)
    assert even_size(720, 1280, 640) == (360, 640)      # depan diputar (potret)
    assert even_size(641, 361, None) == (640, 360)
    assert even_size(320, 240, 640) == (320, 240)        # tidak diperbesar


def test_ffmpeg_cmd_uses_config_values():
    cmd = ffmpeg_cmd("ffmpeg", "rtsp://h:8554/x", 640, 360, {"fps": 10, "gop_s": 2, "bitrate": "700k"})
    assert cmd[cmd.index("-s") + 1] == "640x360"
    assert cmd[cmd.index("-g") + 1] == "20"
    assert cmd[cmd.index("-b:v") + 1] == "700k"
    assert cmd[cmd.index("-profile:v") + 1] == "baseline"
    assert cmd[-1] == "rtsp://h:8554/x"


def test_frames_resized_to_fixed_size_and_rate_limited():
    conns = []

    def connect(url, w, h):
        conns.append(FakeConn(url, w, h))
        return conns[-1]

    s = OverlayStreams({"base_url": "rtsp://h:8554/", "channels": {"cabin": "fl-01/cabin"},
                        "fps": 5, "max_side": 640}, connect=connect)
    img = np.zeros((720, 1280, 3), np.uint8)
    assert s.wants("cabin") and not s.wants("front")     # kanal tak terdaftar
    s.publish(img, "cabin")
    assert not s.wants("cabin")                          # < 1/fps sejak publish
    assert _wait(lambda: conns and len(conns[0].frames) == 1)
    assert conns[0].url == "rtsp://h:8554/fl-01/cabin"
    assert (conns[0].w, conns[0].h) == (640, 360)
    assert len(conns[0].frames[0]) == 640 * 360 * 3
    time.sleep(0.21)
    s.publish(np.zeros((360, 600, 3), np.uint8), "cabin")   # ukuran lain -> diubah ke ukuran koneksi
    assert _wait(lambda: len(conns[0].frames) == 2)
    assert len(conns[0].frames[1]) == 640 * 360 * 3
    s.close()
    assert conns[0].closed


def test_reconnects_after_failure_with_backoff():
    conns = []

    def connect(url, w, h):
        conns.append(FakeConn(url, w, h, fail_after=1 if not conns else None))
        return conns[-1]

    s = OverlayStreams({"channels": {"front": "f"}, "fps": 100, "retry_s": 0.2}, connect=connect)
    img = np.zeros((64, 64, 3), np.uint8)
    s.publish(img, "front")
    assert _wait(lambda: conns and len(conns[0].frames) == 1)
    time.sleep(0.02)
    s.publish(img, "front")                              # gagal tulis -> putus, backoff
    assert _wait(lambda: conns[0].closed)
    assert not s.wants("front")                          # selama backoff overlay tidak perlu digambar
    assert _wait(lambda: s.wants("front"), 1.0)
    s.publish(img, "front")
    assert _wait(lambda: len(conns) == 2 and len(conns[1].frames) == 1)
    s.close()


class Rec:
    def __init__(self, want):
        self.want = want
        self.got = []
        self.labels = {}

    def wants(self, ch="cabin"):
        return self.want

    def publish(self, img, ch="cabin"):
        if self.want:
            self.got.append(ch)

    def add_channel(self, name, label):
        self.labels[name] = label


def test_fanout_wants_any_and_forwards():
    a, b = Rec(False), Rec(True)
    f = VideoFanout(a, None, b)
    assert f.wants("front")
    f.publish(np.zeros((2, 2, 3), np.uint8), "front")
    assert a.got == [] and b.got == ["front"]
    f.add_channel("front", "Kamera depan")
    assert a.labels == {"front": "Kamera depan"}
