import json
import time

from src.front.pipeline import FrontCandidate, FrontResult, FrontTrack, dets_message
from src.io.dets_post import DetsPoster


def _wait(cond, t=2.0):
    end = time.monotonic() + t
    while time.monotonic() < end:
        if cond():
            return True
        time.sleep(0.01)
    return False


def test_dets_message_normalised_boxes_colours_and_labels():
    t1 = FrontTrack(3, "person", 0.9, (36.0, 64.0, 180.0, 640.0), 4.2, 0.1, "foot", -0.5, True)
    t2 = FrontTrack(None, "obstacle", 0.5, (-5.0, 0.0, 400.0, 10.0), None, None, None, None, False)
    res = FrontResult(1234, 7, [t1, t2], [])
    m = dets_message(res, 360, 640, 270)
    assert m["t"] == 1234 and m["rot"] == 270
    a, b = m["boxes"]
    assert a["b"] == [0.1, 0.1, 0.5, 1.0]
    assert a["c"] == "#3cc83c" and a["l"].startswith("#3 person 4.2m")    # hijau: ada jarak, bukan kandidat
    assert b["b"] == [0.0, 0.0, 1.0, 0.0156] and b["c"] == "#a0a0a0"      # dipotong ke 0..1; abu-abu
    json.dumps(m)                                                         # bisa dikirim apa adanya


def test_dets_message_candidate_colour_matches_level():
    t = FrontTrack(1, "person", 0.9, (0, 0, 10, 10), 1.0, 0.0, "foot", None, False)
    c = FrontCandidate("F02", "A-03", 1, "person", 1.0, 0.9)
    m = dets_message(FrontResult(1, 1, [t], [c]), 10, 10)
    assert m["boxes"][0]["c"] == "#ff0000" and m["boxes"][0]["l"].endswith("F02?")


class _Resp:
    def __enter__(self):
        return self

    def __exit__(self, *a):
        pass

    def read(self):
        return b"{}"


def test_dets_poster_sends_latest_and_backs_off_on_error():
    got, fail = [], {"on": False}

    def opener(req, timeout):
        if fail["on"]:
            raise OSError("agen edge mati")
        got.append(json.loads(req.data))
        return _Resp()

    p = DetsPoster("http://127.0.0.1:8098/dets?cam=forward", retry_s=0.3, opener=opener)
    p.push({"t": 1})
    assert _wait(lambda: got == [{"t": 1}])
    fail["on"] = True
    p.push({"t": 2})
    assert _wait(lambda: p.errors == 1)
    p.push({"t": 3})                                    # selama jeda: dibuang, tidak antre
    fail["on"] = False
    time.sleep(0.1)
    assert got == [{"t": 1}]
    time.sleep(0.3)
    p.push({"t": 4})
    assert _wait(lambda: got[-1] == {"t": 4})
    p.close()
