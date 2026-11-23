"""Encode JPEG klip di thread terpisah harus menghasilkan isi klip yang sama dengan mode sinkron."""
import numpy as np

from src.io.clips import ClipRecorder

T = 1_800_000_000_000


def _run(tmp_path, async_encode: bool, defer: bool = False, until: str = "fixed"):
    name = ("a" if async_encode else "s") + ("d" if defer else "") + until
    rec = ClipRecorder({"pre_s": 2, "post_s": 1, "fps": 10, "width": 160, "async_encode": async_encode,
                        "defer_event_start": defer, "clip_until": until, "post_end_s": 0.5, "max_s": 6},
                       tmp_path / name)
    clips = []
    rec.on_done = clips.append
    ev = {"event_id": "e1", "device_id": "t", "code": "D02", "level": "A-03", "ts_event_ms": T + 3000}
    out = None
    for i in range(60):                                   # 30 fps, 2 s -> disubsample ke 10 fps
        ts = T + i * 33 + (1000 if i >= 30 else 0)
        rec.push(np.full((120, 160, 3), i * 4, np.uint8), ts)
        if i == 45:
            out = rec.on_event_start(ev, np.zeros((120, 160, 3), np.uint8))
        if i == 50:
            rec.on_event_end({"event_id": "e1", "ts_event_ms": ts})
    rec.flush()
    return rec, clips, out


def test_async_matches_sync(tmp_path):
    s, cs, _ = _run(tmp_path, False)
    a, ca, _ = _run(tmp_path, True)
    assert [t for t, _ in s.buf] == [t for t, _ in a.buf] and len(a.buf) > 0
    assert len(cs) == len(ca) == 1 and ca[0]["clip_ok"] == cs[0]["clip_ok"]
    assert (ca[0]["clip_start_ms"], ca[0]["clip_end_ms"]) == (cs[0]["clip_start_ms"], cs[0]["clip_end_ms"])


def test_deferred_event_start_same_clip_and_snapshot(tmp_path):
    for until in ("fixed", "end"):
        s, cs, so = _run(tmp_path, False, until=until)
        d, cd, do = _run(tmp_path, True, defer=True, until=until)
        assert do["snapshot_path"].endswith("e1.jpg") and do["clip_path"].endswith("e1.mp4")
        assert do["snapshot_before_condition"] is None
        assert len(cs) == len(cd) == 1
        assert (cd[0]["clip_start_ms"], cd[0]["clip_end_ms"]) == (cs[0]["clip_start_ms"], cs[0]["clip_end_ms"])
        snap = next((tmp_path / ("ad" + until)).rglob("e1.jpg"))
        assert snap.stat().st_size > 0
