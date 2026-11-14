"""Uji kecepatan/arah gerak dengan deret jarak sintetis dan timestamp sumber skala epoch."""
import numpy as np
import pytest

from src.config import CONFIG_DIR, load_yaml
from src.front.motion import MotionTracker

EPOCH_MS = 1_794_130_000_000
FPS = 30


def _mt():
    thr = load_yaml(CONFIG_DIR / "thresholds.yaml")["front"]
    cam = load_yaml(CONFIG_DIR / "front_camera.yaml")
    return MotionTracker.from_config(thr, cam)


def _run(mt, speed_mps, d0=8.0, dur_s=2.0, fps=FPS, noise=0.0, seed=0, tid=1):
    rng = np.random.default_rng(seed)
    states = []
    for i in range(int(dur_s * fps)):
        t = i / fps
        d = d0 + speed_mps * t + (rng.normal(0, noise) if noise else 0.0)
        states.append(mt.update(tid, EPOCH_MS + int(round(t * 1000)), d, "foot"))
    return states


def test_approaching_1mps_flagged():
    st = _run(_mt(), -1.0)
    assert st[-1].approaching
    assert st[-1].speed_mps == pytest.approx(-1.0, abs=0.05)


def test_receding_not_flagged():
    st = _run(_mt(), +1.0, d0=3.0)
    assert not any(s.approaching for s in st)
    assert st[-1].speed_mps == pytest.approx(1.0, abs=0.05)


def test_stationary_not_flagged():
    st = _run(_mt(), 0.0)
    assert not any(s.approaching for s in st)


def test_slow_drift_below_threshold_not_flagged():
    thr = load_yaml(CONFIG_DIR / "thresholds.yaml")["front"]["approach_speed_mps"]
    st = _run(_mt(), -0.5 * thr)
    assert not any(s.approaching for s in st)


@pytest.mark.parametrize("seed", range(5))
def test_noise_robustness(seed):
    # noise 5 cm per frame (+-) di 30 fps
    assert not any(s.approaching for s in _run(_mt(), 0.0, noise=0.05, seed=seed, dur_s=10))
    st = _run(_mt(), -1.0, noise=0.05, seed=seed)
    assert all(s.approaching for s in st[FPS:])


def test_uses_source_timestamps_not_call_count():
    # 10 fps sumber, 1 m/s: kecepatan tetap -1 walau jumlah panggilan berbeda dari 30 fps
    st = _run(_mt(), -1.0, fps=10)
    assert st[-1].speed_mps == pytest.approx(-1.0, abs=0.05)


def test_not_enough_samples_gives_none():
    mt = _mt()
    s = mt.update(1, EPOCH_MS, 5.0, "foot")
    assert s.speed_mps is None and not s.approaching


def test_duplicate_timestamps_do_not_crash():
    mt = _mt()
    for _ in range(10):
        s = mt.update(1, EPOCH_MS, 5.0, "foot")
    assert s.speed_mps is None


def test_method_switch_resets_history():
    mt = _mt()
    _run(mt, 0.0, d0=6.0, dur_s=1.0)
    # loncatan 6 m -> 3 m karena ganti metode tidak boleh terbaca mendekat
    s = mt.update(1, EPOCH_MS + 1100, 3.0, "clipped")
    assert s.speed_mps is None and not s.approaching


def test_prune_stale_tracks():
    mt = _mt()
    mt.update(1, EPOCH_MS, 5.0, "foot")
    mt.update(2, EPOCH_MS + 5000, 5.0, "foot")
    dead = mt.prune(EPOCH_MS + 5000)
    assert dead == [1] and 2 in mt.tracks
