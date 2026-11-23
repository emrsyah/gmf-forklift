"""D-09 lensa tertutup sebagian (PartialBlock) dengan frame sintetis."""
import numpy as np

from src.cabin.camblock import PartialBlock

CFG = {"enabled": True, "grid": [4, 4], "downscale_px": 160, "drop_ratio": 0.25, "min_base_lap": 40,
       "min_cells": 4, "frac_min": 0.6, "baseline_alpha": 0.05, "warmup_frames": 15}
RNG = np.random.default_rng(0)
SCENE = RNG.integers(0, 255, (120, 160, 3), dtype=np.uint8)       # kabin bertekstur


def covered(frac_cols: float, value: int = 60) -> np.ndarray:
    """Telapak tangan blur menutup sebagian kiri frame (fraksi lebar)."""
    img = SCENE.copy()
    img[:, : int(160 * frac_cols)] = value
    return img


def warm(pb: PartialBlock, n: int = 20):
    for _ in range(n):
        pb.update(SCENE, face_present=True)


def test_no_baseline_no_alert():
    pb = PartialBlock(CFG)
    assert not pb.update(covered(1.0), face_present=False)        # belum ada baseline -> tidak menebak


def test_three_quarter_cover_detected():
    pb = PartialBlock(CFG)
    warm(pb)
    assert pb.update(covered(0.75), face_present=False)
    assert 0.7 <= pb.frac <= 0.8


def test_half_cover_below_threshold():
    pb = PartialBlock(CFG)
    warm(pb)
    assert not pb.update(covered(0.5), face_present=False)
    assert 0.4 <= pb.frac <= 0.6


def test_normal_scene_not_blocked():
    pb = PartialBlock(CFG)
    warm(pb)
    assert not pb.update(SCENE, face_present=False)
    assert pb.frac == 0.0


def test_plain_wall_cells_are_ignored():
    """Dinding polos (baseline sudah rata) tidak dihitung, jadi kamera yang melihat banyak dinding tidak
    dianggap tertutup."""
    scene = SCENE.copy()
    scene[:, :80] = 128                                            # separuh kiri dinding polos
    pb = PartialBlock(CFG)
    for _ in range(20):
        pb.update(scene, face_present=True)
    assert not pb.update(scene, face_present=False)
    img = scene.copy()
    img[:, 80:] = 60                                               # tangan menutup separuh kanan (bertekstur)
    # sel di tepi dinding/tangan tetap punya tepi tajam -> tidak semua sel informatif tertutup
    assert pb.update(img, face_present=False) and pb.frac >= 0.6


def test_dark_cells_count_as_covered():
    pb = PartialBlock(CFG)
    warm(pb)
    img = covered(0.75, value=3)
    assert pb.update(img, face_present=False)


def test_baseline_not_updated_while_face_absent():
    pb = PartialBlock(CFG)
    warm(pb)
    for _ in range(200):                                           # lama tertutup tanpa wajah
        pb.update(covered(0.75), face_present=False)
    assert pb.update(covered(0.75), face_present=False)            # baseline tidak "belajar" tangan


def test_hysteresis_holds_through_threshold_chatter():
    # uji_yawn 157-163 s: fraksi berayun di sekitar frac_min -> tanpa hysteresis status putus-sambung
    pb = PartialBlock({**CFG, "frac_exit": 0.3})
    warm(pb)
    assert pb.update(covered(0.75), face_present=False)
    assert pb.update(covered(0.5), face_present=False) and pb.frac < CFG["frac_min"]   # di bawah masuk, di atas keluar
    assert not pb.update(covered(0.2), face_present=False)                            # tangan dilepas
    assert not pb.update(covered(0.5), face_present=False)                            # masuk lagi butuh frac_min
    no_h = PartialBlock(CFG)
    warm(no_h)
    no_h.update(covered(0.75), face_present=False)
    assert not no_h.update(covered(0.5), face_present=False)


def test_disabled():
    pb = PartialBlock({**CFG, "enabled": False})
    warm(pb)
    assert not pb.update(covered(1.0), face_present=False)
