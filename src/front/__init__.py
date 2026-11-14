"""Kamera depan (milestone L9): deteksi orang/rintangan + ByteTrack + arah gerak + jarak titik kaki.

Kode event F01..F03 adalah USULAN (belum disepakati HSE GMF); modul ini hanya menghasilkan
indikator kandidat. Lihat docs/FRONT_CAMERA.md.

Impor detector/pipeline dibuat malas supaya distance/motion/zones bisa diuji tanpa ultralytics.
"""
from .distance import CameraModel, DistanceEstimate, DistanceEstimator
from .motion import MotionState, MotionTracker
from .zones import FrontCandidate, ZoneEvaluator

__all__ = [
    "CameraModel", "DistanceEstimate", "DistanceEstimator",
    "MotionState", "MotionTracker",
    "FrontCandidate", "ZoneEvaluator",
    "FrontDetector", "FrontPipeline", "FrontResult", "FrontTrack",
]


def __getattr__(name):
    if name == "FrontDetector":
        from .detector import FrontDetector
        return FrontDetector
    if name in ("FrontPipeline", "FrontResult", "FrontTrack"):
        from . import pipeline
        return getattr(pipeline, name)
    raise AttributeError(name)
