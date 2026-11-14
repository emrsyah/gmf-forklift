"""Tes replay harness: penilaian (runner palsu) + end-to-end lewat pipeline asli pada video sintetis."""
from pathlib import Path

import cv2
import numpy as np
import pytest

from eval.replay_eval import VideoRun, evaluate, load_truth, score, write_report
from src.config import REPO_ROOT, Config

SAMPLE = REPO_ROOT / "data" / "samples" / "zidane.jpg"


def _truth(tmp: Path, rows: list[str]) -> Path:
    p = tmp / "truth.csv"
    p.write_text("video,start_s,end_s,expected_code,expected_level,peraga,cahaya,scenario\n" + "\n".join(rows) + "\n",
                 encoding="utf-8")
    return p


def _ev(t, code, level, i):
    return {"event_id": f"e{i}", "phase": "start", "code": code, "level": level, "t_video_s": t}


def test_scoring_first_hit_and_false_alarms(tmp_path):
    truth = _truth(tmp_path, [
        "a.mp4,10,13,D02,A-03,P1,normal,5",
        "a.mp4,40,43,D05,A-02,P1,normal,8",
        "a.mp4,70,73,D06,A-02,P1,normal,12",
        "n.mp4,0,0,NONE,,P1,normal,",
    ])
    runs = {
        "a.mp4": VideoRun("a.mp4", 100, [_ev(11.6, "D02", "A-03", 1),
                                         _ev(41.0, "D04", "A-03", 2), _ev(42.0, "D05", "A-02", 3),
                                         _ev(90.0, "D10", "A-02", 4)]),
        "n.mp4": VideoRun("n.mp4", 1800, [_ev(100, "D05", "A-02", 5)]),
    }
    s = score(load_truth(truth), runs)
    assert s["n"] == 3 and s["detected"] == 2
    r = {x.row.scenario: x for x in s["results"]}
    assert r["5"].code_ok and r["5"].level_ok and r["5"].t_detect == pytest.approx(1.6)
    assert not r["8"].code_ok                       # hit PERTAMA salah kode -> K4 gagal
    assert not r["12"].detected
    assert s["fa_per_hour"] == pytest.approx(2.0)   # 1 event dalam 0,5 jam
    assert len(s["stray_events"]) == 1              # D10 di t=90
    md = write_report(s, {"when": "x", "truth": "t", "mode": "fast", "simulate_nir": False, "yolo": False},
                      tmp_path / "rep")
    txt = md.read_text(encoding="utf-8")
    assert "K3 terdeteksi" in txt and "Confusion matrix" in txt


def _make_video(path: Path, segments: list[tuple[float, np.ndarray]], fps: float = 15.0):
    h, w = segments[0][1].shape[:2]
    vw = cv2.VideoWriter(str(path), cv2.VideoWriter_fourcc(*"mp4v"), fps, (w, h))
    rng = np.random.default_rng(0)
    for dur, img in segments:
        for _ in range(int(dur * fps)):
            noise = rng.integers(-2, 3, img.shape, dtype=np.int16)
            vw.write(np.clip(img.astype(np.int16) + noise, 0, 255).astype(np.uint8))
    vw.release()


@pytest.mark.skipif(not SAMPLE.exists(), reason="data/samples/zidane.jpg tidak ada (unduh dari ultralytics.com/images)")
def test_end_to_end_lens_covered_D09(tmp_path):
    face = cv2.imread(str(SAMPLE))[0:720, 600:1280]
    face = cv2.resize(face, (480, 508))
    covered = np.full_like(face, 30)            # telapak tangan: gelap dan rata
    bright = np.full_like(face, 200)            # dengan lampu IR: terang dan rata
    vid = tmp_path / "s20.mp4"
    _make_video(vid, [(8, face), (4, covered), (6, face), (4, bright), (4, face)])
    truth = _truth(tmp_path, ["s20.mp4,8,12,D09,A-03,P1,normal,20",
                              "s20.mp4,18,22,D09,A-03,P1,normal,20"])
    cfg = Config.load()
    s = evaluate(truth, cfg, mode="fast", use_yolo=False)
    assert s["detected"] == 2, [r.__dict__ for r in s["results"]]
    assert all(r.code_ok and r.level_ok for r in s["results"])
    assert all(r.t_detect <= 1.5 for r in s["results"])
    assert s["stray_events"] == []              # wajah diam tidak memicu alert lain
