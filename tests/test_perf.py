"""Batas thread dari config: null tidak mengubah apa-apa, OpenCV diterapkan, threadpoolctl opsional."""
import sys

import cv2

from src.perf import apply_thread_limits


def test_null_changes_nothing():
    before = cv2.getNumThreads()
    assert apply_thread_limits({"blas": None, "openmp": None, "opencv": None}) == {}
    assert apply_thread_limits(None) == {}
    assert cv2.getNumThreads() == before


def test_opencv_limit_applied_and_restorable():
    before = cv2.getNumThreads()
    try:
        assert apply_thread_limits({"opencv": 1}) == {"opencv": 1}
        assert cv2.getNumThreads() == 1
    finally:
        cv2.setNumThreads(before)


def test_blas_limit_skipped_without_threadpoolctl(monkeypatch, capsys):
    monkeypatch.setitem(sys.modules, "threadpoolctl", None)     # import gagal
    assert apply_thread_limits({"blas": 1, "openmp": 1}) == {}
    assert "threadpoolctl tidak terpasang" in capsys.readouterr().out
