import csv
import shutil

import cv2
import numpy as np
import pytest

from src.io.continuous import ContinuousRecorder


def _cfg(tmp_path, **kw):
    c = {"dir": str(tmp_path), "segment_s": 1.0, "fps": 10, "max_side": 64, "min_free_gb": 0}
    c.update(kw)
    return c


def test_segments_and_timestamps(tmp_path):
    rec = ContinuousRecorder(_cfg(tmp_path), "t")
    img = np.full((96, 128, 3), 120, np.uint8)
    t0 = 1_800_000_000_000
    for i in range(90):                      # 30 fps selama 3 s -> diambil ~10 fps
        rec.push(img, t0 + i * 33)
    rec.close()
    vids = sorted(tmp_path.rglob("*.avi"))
    assert len(vids) == 3                    # segmen 1 s
    total = 0
    for v in vids:
        rows = list(csv.DictReader(open(str(v) + ".ts.csv", encoding="utf-8")))
        cap = cv2.VideoCapture(str(v))
        n = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        ok, fr = cap.read()
        cap.release()
        assert ok and max(fr.shape[:2]) == 64      # diperkecil ke max_side
        assert n == len(rows)
        assert [int(r["frame"]) for r in rows] == list(range(len(rows)))
        total += len(rows)
    assert 28 <= total <= 31
    assert rec.frames == total and rec.dropped == 0


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg tidak ada")
def test_h264_segments_frames_match_timestamps(tmp_path):
    # ukuran ganjil (97x65 setelah max_side) -> ffmpeg harus pad ke genap untuk yuv420p
    rec = ContinuousRecorder(_cfg(tmp_path, codec="H264", max_side=97), "t")
    rng = np.random.default_rng(0)
    t0 = 1_800_000_000_000
    for i in range(60):                      # 30 fps selama 2 s -> ~20 frame rekaman, segmen 1 s
        rec.push(rng.integers(0, 255, (130, 194, 3), np.uint8), t0 + i * 33)
    rec.close()
    assert rec.stopped_reason is None
    vids = sorted(tmp_path.rglob("*.mp4"))
    assert len(vids) == 2 and not list(tmp_path.rglob("*.avi"))
    total = 0
    for v in vids:
        rows = list(csv.DictReader(open(str(v) + ".ts.csv", encoding="utf-8")))
        cap = cv2.VideoCapture(str(v))
        n, shape = 0, None
        while True:
            ok, fr = cap.read()
            if not ok:
                break
            n, shape = n + 1, fr.shape
        cap.release()
        assert n == len(rows) > 0            # syarat replay: frame video = baris ts.csv
        assert shape[:2] == (66, 98)         # 65x97 dipad ke genap
        total += n
    assert 19 <= total <= 21 and rec.frames == total


def test_h264_without_ffmpeg_stops_cleanly(tmp_path, monkeypatch):
    monkeypatch.setattr(shutil, "which", lambda *_: None)
    rec = ContinuousRecorder(_cfg(tmp_path, codec="H264"), "t")
    rec.push(np.zeros((32, 32, 3), np.uint8), 0)
    rec.close()
    assert rec.stopped_reason and "ffmpeg" in rec.stopped_reason


def test_stops_when_disk_low(tmp_path):
    rec = ContinuousRecorder(_cfg(tmp_path, min_free_gb=1e9), "t")
    rec.push(np.zeros((32, 32, 3), np.uint8), 0)
    rec.close()
    assert rec.stopped_reason and "disk" in rec.stopped_reason
    assert not list(tmp_path.rglob("*.avi"))
