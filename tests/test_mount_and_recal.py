"""Sudut pasang kamera (pilar) di kalibrasi + kalibrasi ulang otomatis setelah wajah lama hilang."""
from src.cabin.calibration import Calibration
from src.cabin.pipeline import CabinPipeline
from src.config import Config
from tests.synth import thresholds

TH = thresholds()


def _feed(cal: Calibration, yaw: float, pitch: float = 0.0, roll: float = 0.0, n: int = 40, dt_ms: int = 150):
    for i in range(n):
        cal.observe(i * dt_ms, {"ear_l": 0.3, "ear_r": 0.3, "yaw": yaw, "pitch": pitch, "roll": roll})


def test_pillar_mount_rejected_without_offset():
    cal = Calibration(dict(TH["calibration"], max_attempts=99))
    _feed(cal, yaw=30.0)
    assert not cal.done and cal.attempts >= 1                # ditolak (pose tidak lurus), lalu mencoba lagi


def test_pillar_mount_accepted_with_offset():
    cal = Calibration(dict(TH["calibration"], mount_yaw_deg=30.0))
    _feed(cal, yaw=32.0)
    assert cal.done and abs(cal.yaw0 - 32.0) < 1e-6


def test_population_baseline_uses_mount_angle():
    cal = Calibration(dict(TH["calibration"], mount_yaw_deg=28.0, mount_pitch_deg=-10.0))
    cal.set_population()
    assert (cal.yaw0, cal.pitch0, cal.roll0) == (28.0, -10.0, 0.0)


def test_mount_offset_still_rejects_turned_head():
    cal = Calibration(dict(TH["calibration"], mount_yaw_deg=30.0, max_attempts=99))
    _feed(cal, yaw=-10.0)                                   # 40 deg dari arah pasang -> pengemudi menoleh
    assert not cal.done


def _pipe(after_s):
    cfg = Config.load()
    th = dict(cfg.thresholds)
    th["calibration"] = dict(th["calibration"], auto_recalibrate_after_absent_s=after_s)
    p = CabinPipeline(cfg.runtime, th, cfg.events, face_model=object(), use_yolo=False)
    p.calibration.active = False                            # anggap sudah terkalibrasi
    p.calibration.done = True
    return p


def test_recalibrates_after_long_absence():
    p = _pipe(20.0)
    p._auto_recalibrate(True, 0)
    p._auto_recalibrate(False, 10_000)                       # turun dari forklift
    p._auto_recalibrate(True, 25_000)                        # kembali / ganti pengemudi
    assert p.calibration.active and p.auto_recalibrations == 1


def test_no_recalibration_for_lembar_a_lens_cover():
    p = _pipe(20.0)
    p._auto_recalibrate(True, 0)
    p._auto_recalibrate(True, 1_000)
    p._auto_recalibrate(True, 1_000 + 17_000)                # D-09: lensa ditutup ~15-17 s
    assert not p.calibration.active and p.auto_recalibrations == 0


def test_auto_recalibration_disabled():
    p = _pipe(None)
    p._auto_recalibrate(True, 0)
    p._auto_recalibrate(True, 600_000)
    assert not p.calibration.active
