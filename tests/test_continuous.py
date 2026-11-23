import csv
import shutil

import cv2
import numpy as np
import pytest

from src.io.continuous import ContinuousRecorder, avi_needs_repair, repair_unfinished


def _killed_avi(d, name="cabin_x.avi", n=60):
    """Salinan .avi sebelum release() = keadaan file saat proses dibunuh di tengah segmen."""
    live = d / "live_src.avi"
    vw = cv2.VideoWriter(str(live), cv2.VideoWriter_fourcc(*"MJPG"), 10, (320, 240), True)
    rng = np.random.default_rng(0)
    for _ in range(n):                       # noise: JPEG besar -> buffer penulis sudah ter-flush ke disk
        vw.write(rng.integers(0, 255, (240, 320, 3), dtype=np.uint8))
    shutil.copy(live, d / name)
    vw.release()
    return d / name, live


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="butuh ffmpeg")
def test_repair_unfinished_avi(tmp_path):
    import time
    killed, healthy = _killed_avi(tmp_path)
    assert avi_needs_repair(killed) and not avi_needs_repair(healthy)
    assert repair_unfinished(tmp_path, shutil.which("ffmpeg"), older_than=0) == []    # segmen "baru": tidak disentuh
    fixed = repair_unfinished(tmp_path, shutil.which("ffmpeg"), older_than=time.time() + 1)
    assert fixed == [killed]
    cap = cv2.VideoCapture(str(killed))
    assert int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) > 0
    assert not list(tmp_path.glob("*.repair.*"))


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
