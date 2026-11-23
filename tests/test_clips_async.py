"""Encode JPEG klip di thread terpisah harus menghasilkan isi klip yang sama dengan mode sinkron."""
import numpy as np

from src.io.clips import ClipRecorder

T = 1_800_000_000_000


def _run(tmp_path, async_encode: bool):
    rec = ClipRecorder({"pre_s": 2, "post_s": 1, "fps": 10, "width": 160, "async_encode": async_encode},
                       tmp_path / ("a" if async_encode else "s"))
    clips = []
    rec.on_done = clips.append
    ev = {"event_id": "e1", "device_id": "t", "code": "D02", "level": "A-03", "ts_event_ms": T + 3000}
    for i in range(60):                                   # 30 fps, 2 s -> disubsample ke 10 fps
        ts = T + i * 33 + (1000 if i >= 30 else 0)
        rec.push(np.full((120, 160, 3), i * 4, np.uint8), ts)
        if i == 45:
            rec.on_event_start(ev, np.zeros((120, 160, 3), np.uint8))
    rec.flush()
    return rec, clips


def test_async_matches_sync(tmp_path):
    s, cs = _run(tmp_path, False)
    a, ca = _run(tmp_path, True)
    assert [t for t, _ in s.buf] == [t for t, _ in a.buf] and len(a.buf) > 0
    assert len(cs) == len(ca) == 1 and ca[0]["clip_ok"] == cs[0]["clip_ok"]
    assert (ca[0]["clip_start_ms"], ca[0]["clip_end_ms"]) == (cs[0]["clip_start_ms"], cs[0]["clip_end_ms"])
