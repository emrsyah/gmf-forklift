"""Bukti + status perangkat: klip sampai kondisi selesai, snapshot D-09 sebelum tertutup, ts_buzzer_ms,
heartbeat, info media."""
import shutil

import cv2
import numpy as np
import pytest

from src.decision import Action
from src.io.clips import ClipRecorder
from src.io.heartbeat import Heartbeat
from src.io.sinks import FileSink, MemorySink, MultiSink
from src.main import handle_actions

T = 1_794_000_000_000


def _img(v):
    return np.full((120, 160, 3), v, np.uint8)


def _rec(tmp_path, **kw):
    # sinkron: tes ini memeriksa `pending` di antara push (async == sinkron diuji di test_clips_async)
    cfg = {"pre_s": 2, "post_s": 1, "fps": 10, "width": 160, "clip_until": "end", "post_end_s": 1, "max_s": 6,
           "async_encode": False}
    cfg.update(kw)
    return ClipRecorder(cfg, tmp_path)


def test_clip_runs_until_condition_end(tmp_path):
    rec = _rec(tmp_path)
    ev = {"event_id": "e1", "ts_event_ms": T + 3000, "code": "D02", "level": "A-03"}
    for i in range(31):
        ts = T + i * 100
        rec.push(_img(100), ts)
        if ts == T + 3000:
            rec.on_event_start(ev, _img(100))
    # kondisi selesai di t=4.0 s -> klip berhenti di 5.0 s (post_end 1 s), bukan di batas max (6 s dari awal)
    rec.on_event_end({"event_id": "e1", "ts_event_ms": T + 4000})
    p = rec.pending[0]
    assert p.until_ms == T + 5000
    for i in range(31, 60):
        rec.push(_img(100), T + i * 100)
    assert not rec.pending                                   # sudah diserahkan ke encoder
    rec.flush()


def test_clip_capped_at_max_s(tmp_path):
    rec = _rec(tmp_path)
    ev = {"event_id": "e2", "ts_event_ms": T + 3000, "code": "D09", "level": "A-03"}
    for i in range(31):
        rec.push(_img(100), T + i * 100)
    rec.on_event_start(ev, _img(100))
    t_first = rec.pending[0].frames[0][0]
    assert rec.pending[0].until_ms == t_first + 6000         # max_s dari frame pertama klip
    rec.on_event_end({"event_id": "e2", "ts_event_ms": T + 60_000})   # end jauh sesudahnya -> tetap dibatasi
    assert rec.pending[0].until_ms == t_first + 6000
    rec.flush()


def test_fixed_mode_unchanged(tmp_path):
    rec = _rec(tmp_path, clip_until="fixed")
    for i in range(31):
        rec.push(_img(100), T + i * 100)
    rec.on_event_start({"event_id": "e3", "ts_event_ms": T + 3000, "code": "D02", "level": "A-03"}, _img(100))
    assert rec.pending[0].until_ms == T + 4000               # post_s 1 s
    rec.on_event_end({"event_id": "e3", "ts_event_ms": T + 3200})
    assert rec.pending[0].until_ms == T + 4000
    rec.flush()


def test_d09_snapshot_is_taken_before_cover(tmp_path):
    rec = _rec(tmp_path, snapshot_before_condition_codes=["D09"], snapshot_before_condition_s=0.5)
    for i in range(30):                                      # 0..2.0 s terang, sesudahnya hitam (tertutup)
        rec.push(_img(200 if i <= 20 else 0), T + i * 100)
    ev = {"event_id": "e4", "ts_event_ms": T + 2900, "ts_condition_start_ms": T + 2100,
          "code": "D09", "level": "A-03"}
    out = rec.on_event_start(ev, _img(0))
    assert out["snapshot_before_condition"] is True
    snap = cv2.imread(str(next(tmp_path.rglob("e4.jpg"))))
    assert snap[60:, :].mean() > 150                         # frame sebelum tertutup, bukan hitam
    rec.flush()


def test_other_codes_snapshot_now(tmp_path):
    rec = _rec(tmp_path)
    for i in range(30):
        rec.push(_img(200), T + i * 100)
    ev = {"event_id": "e5", "ts_event_ms": T + 2900, "ts_condition_start_ms": T + 1000,
          "code": "D02", "level": "A-03"}
    out = rec.on_event_start(ev, _img(30))
    assert out["snapshot_before_condition"] is False
    snap = cv2.imread(str(next(tmp_path.rglob("e5.jpg"))))
    assert snap[60:, :].mean() < 60                          # gambar saat event


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg tidak ada")
def test_media_callback_after_clip_written(tmp_path):
    rec = _rec(tmp_path)
    sink = MultiSink([MemorySink(), FileSink(tmp_path)])
    rec.on_done = sink.media
    ev = {"event_id": "e6", "device_id": "d", "ts_event_ms": T + 2000, "code": "D05", "level": "A-02"}
    for i in range(50):
        ts = T + i * 100
        rec.push(_img(100), ts)
        if ts == T + 2000:
            rec.on_event_start(ev, _img(100))
        if ts == T + 2500:
            rec.on_event_end({"event_id": "e6", "ts_event_ms": ts})
    rec.flush()
    m = sink.sinks[0].medias
    assert len(m) == 1 and m[0]["clip_ok"] and m[0]["event_id"] == "e6"
    assert 2.5 <= m[0]["clip_duration_s"] <= 3.6             # 2 s sebelum .. 1 s sesudah end
    assert list(tmp_path.rglob("media.jsonl"))


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg tidak ada")
def test_slow_pipeline_clip_keeps_real_duration(tmp_path):
    # pipeline 9 fps < fps rekaman 15 -> durasi video harus tetap ~waktu nyata, bukan dipercepat
    import subprocess
    rec = _rec(tmp_path, fps=15, pre_s=2, max_s=60, post_end_s=0.5)
    ev = {"event_id": "slow", "ts_event_ms": T + 2000, "code": "D02", "level": "A-03"}
    step = 111                                               # ~9 fps
    for i in range(60):
        ts = T + i * step
        rec.push(_img(100), ts)
        if ev.get("_started") is None and ts >= T + 2000:
            ev["_started"] = True
            rec.on_event_start(ev, _img(100))
        if ts >= T + 5000 and ev.get("_ended") is None:
            ev["_ended"] = True
            rec.on_event_end({"event_id": "slow", "ts_event_ms": ts})
    rec.flush()
    p = rec.done_paths[0]
    dur = float(subprocess.run(["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "csv=p=0",
                                str(p)], capture_output=True, text=True).stdout.strip())
    frames_span = 5.0 + 0.5 - (2.0 - 2.0)                     # ~0..5.5 s dari awal buffer
    assert abs(dur - frames_span) < 0.8, dur


class _Alerter:
    def __init__(self):
        self.calls = []

    def buzz(self, level, code, first):
        self.calls.append((level, code, first))


def test_ts_buzzer_attached_to_start_event():
    ev = {"event_id": "e7", "phase": "start", "code": "D02", "level": "A-03", "ts_event_ms": T}
    acts = [Action("event", ev), Action("buzz", {"level": "A-03", "code": "D02", "first": True})]
    al, sink = _Alerter(), MemorySink()
    sent = handle_actions(acts, al, sink, None, None, now_ms=lambda: T + 7)
    assert al.calls == [("A-03", "D02", True)]
    assert sent[0]["ts_buzzer_ms"] == T + 7 and sink.events[0]["ts_buzzer_ms"] == T + 7


def test_heartbeat_interval_and_fields():
    hb = Heartbeat({"interval_s": 5, "clock_check_s": 60}, "dev", {"cabin": "v1"}, clock_fn=lambda: True, sync=True)
    assert hb.due(T)
    p = hb.build(T, fps=14.96, camera_ok=True, face_present=False, calibration="ok", yolo="on")
    assert p["clock_synced"] is True and p["fps"] == 15.0 and p["device_id"] == "dev"
    assert not hb.due(T + 4000) and hb.due(T + 5000)


def test_heartbeat_disabled():
    assert not Heartbeat({"enabled": False}, "dev").due(T)
