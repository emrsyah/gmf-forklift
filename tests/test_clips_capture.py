"""L8 klip + capture timestamp (replay = live)."""
import shutil
import subprocess

import numpy as np
import pytest

from src.capture import ArraySource, VideoFileSource, sidecar_ts_path
from src.io.clips import ClipRecorder


def test_array_source_timestamps():
    src = ArraySource([np.zeros((10, 10, 3), np.uint8)] * 31, fps=30, start_epoch_ms=1000)
    fr = list(src)
    assert fr[0].ts_ms == 1000 and fr[30].ts_ms == 2000


def test_video_file_uses_sidecar_timestamps(tmp_path):
    import cv2
    p = tmp_path / "v.mp4"
    vw = cv2.VideoWriter(str(p), cv2.VideoWriter_fourcc(*"mp4v"), 30, (32, 32))
    for _ in range(5):
        vw.write(np.zeros((32, 32, 3), np.uint8))
    vw.release()
    # fps nyata webcam tidak rata: 0, 50, 90, 200, 230 ms
    sidecar_ts_path(p).write_text("frame,ts_ms\n0,5000\n1,5050\n2,5090\n3,5200\n4,5230\n", encoding="utf-8")
    fr = list(VideoFileSource(p))
    assert [f.ts_ms for f in fr] == [5000, 5050, 5090, 5200, 5230]
    assert fr[3].video_t_s == pytest.approx(0.2)


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg tidak ada")
def test_clip_pre_post_and_playable(tmp_path):
    rec = ClipRecorder({"pre_s": 2, "post_s": 1, "fps": 10, "width": 320}, tmp_path)
    img = np.full((240, 320, 3), 128, np.uint8)
    t = 1_794_000_000_000
    ev = None
    for i in range(60):                       # 6 s pada 10 fps
        ts = t + i * 100
        rec.push(img, ts)
        if i == 40:
            ev = {"event_id": "x-1", "ts_event_ms": ts, "code": "D02", "level": "A-03"}
            paths = rec.on_event_start(ev, img)
    rec.flush()
    assert paths["clip_path"].endswith("x-1.mp4")
    mp4 = next(tmp_path.rglob("x-1.mp4"))
    assert next(tmp_path.rglob("x-1.jpg")).exists()
    out = subprocess.run(["ffprobe", "-v", "error", "-show_entries", "stream=codec_name,pix_fmt,nb_frames",
                          "-show_entries", "format=duration", "-of", "default=nw=1", str(mp4)],
                         capture_output=True, text=True).stdout
    assert "codec_name=h264" in out and "pix_fmt=yuv420p" in out
    dur = float([ln for ln in out.splitlines() if ln.startswith("duration=")][0].split("=")[1])
    assert 2.5 <= dur <= 3.5                 # 2 s sebelum + 1 s sesudah
