"""Simulasi profil kamera (src/capture/camera_sim.py) dengan frame sintetis: geometri HFOV, ukuran keluaran,
timestamp, batas FPS, peringatan target lebih lebar dari HP, mono tanpa CLAHE, determinisme, dan jarak
kamera depan end-to-end (titik lantai diproyeksikan ke kamera HP -> simulasi -> DistanceEstimator)."""
import math

import cv2
import numpy as np
import pytest

from src.capture import ArraySource
from src.cabin.camblock import frame_stats, is_blocked
from src.capture.camera_sim import (CameraSim, ProfileError, SimulatedSource, already_simulated, build_sim,
                                    list_profiles, load_profiles_file, plan_crop, resolve_profile,
                                    setup_camera_sim, window_title, write_camera_sidecar)
from src.config import CONFIG_DIR, load_yaml
from src.front.distance import DistanceEstimator
from src.front.pipeline import FrontPipeline

DEFAULTS = load_profiles_file()["defaults"]

# Konfigurasi kamera depan TETAP di test (front_camera.yaml bisa ditimpa tools/calibrate_camera.py)
MOUNT_H, THETA = 1.2, 10.0
CAM_CFG = {
    "camera": {"calibrated": False, "image_size": [1280, 720], "fisheye": False,
               "K": [[914.0, 0.0, 640.0], [0.0, 914.0, 360.0], [0.0, 0.0, 1.0]], "dist": [0, 0, 0, 0, 0]},
    "mount": {"h_m": MOUNT_H, "theta_deg": THETA},
    "distance": {"class_height_m": {"person": 1.65}, "bottom_margin_px": 4, "min_ray_angle_deg": 0.5,
                 "max_distance_m": 40.0},
}


def _sim(phone_hfov=70.0, log=lambda *_: None, **prof):
    p = {**DEFAULTS, "name": "uji", "output": [640, 480], "hfov_deg": 62.2, "fps": None, "nir": False}
    p.update(prof)
    return CameraSim(p, phone_hfov, log=log)


def _hfov(w, fx):
    return math.degrees(2 * math.atan((w / 2) / fx))


# ------------------------------------------------------------------------------- geometri
def test_crop_width_uses_tangent_not_angle_ratio():
    pl = plan_crop(1280, 960, 70.0, 640, 480, 62.2)
    expect_w = 1280 * math.tan(math.radians(31.1)) / math.tan(math.radians(35.0))
    assert abs(pl.crop[2] - expect_w) <= 1.0
    assert abs(pl.crop[2] - 1280 * 62.2 / 70.0) > 10          # rasio sudut akan salah
    assert pl.fov_ok and not any("LEBIH LEBAR" in w for w in pl.warnings)
    assert pl.eff_hfov_deg == pytest.approx(62.2, abs=0.15)
    assert _hfov(640, pl.K_out[0, 0]) == pytest.approx(62.2, abs=0.15)
    assert pl.K_out[0, 0] == pytest.approx(pl.K_out[1, 1], rel=0.01)   # resize seragam, tanpa peregangan
    x0, y0, cw, ch = pl.crop
    assert x0 == (1280 - cw) // 2 and y0 == (960 - ch) // 2


def test_target_wider_than_phone_warns_and_reports_effective_hfov():
    pl = plan_crop(1280, 960, 70.0, 640, 480, 102.0)
    assert not pl.fov_ok
    assert any("ultrawide" in w for w in pl.warnings)
    assert pl.eff_hfov_deg == pytest.approx(70.0, abs=0.3)
    assert pl.crop[2] == 1280


def test_phone_aspect_too_short_warns_vertical_shortfall():
    # HP 16:9 @70 deg tidak bisa mengisi keluaran 4:3 @62 deg (butuh tinggi ~0,65 W, ada 0,5625 W)
    pl = plan_crop(1280, 720, 70.0, 640, 480, 62.2)
    assert not pl.fov_ok
    assert any("4:3" in w for w in pl.warnings)
    assert pl.crop[3] == 720
    assert pl.eff_hfov_deg < 62.2


def test_upscale_warning_when_phone_stream_too_small():
    pl = plan_crop(640, 480, 70.0, 1640, 1232, 62.2)
    assert pl.upscale > 1.15 and any("upscale" in w for w in pl.warnings)


def test_output_size_and_nir_is_mono_without_clahe():
    rng = np.random.default_rng(0)
    img = rng.integers(100, 121, (960, 1280, 3), dtype=np.uint8)   # kontras rendah
    img[..., 2] = 110
    out = _sim(nir=True).transform(img)
    assert out.shape == (480, 640, 3) and out.dtype == np.uint8
    assert np.array_equal(out[..., 0], out[..., 1]) and np.array_equal(out[..., 1], out[..., 2])
    # CLAHE akan meregangkan 100..120 ke rentang lebar; simulasi tidak boleh (CLAHE = tugas pipeline)
    assert out.min() >= 95 and out.max() <= 125


def test_noise_and_vibration_deterministic_per_frame_index():
    img = np.full((960, 1280, 3), 128, np.uint8)
    cv2.rectangle(img, (500, 400), (780, 560), (255, 255, 255), -1)
    kw = dict(noise={"read_std": 3.0, "shot": 0.1}, vibration={"shift_px_std": 2.0, "rot_deg_std": 0.3})
    a, b = _sim(**kw), _sim(**kw)
    assert np.array_equal(a.transform(img, 7), b.transform(img, 7))
    assert not np.array_equal(a.transform(img, 7), a.transform(img, 8))


def test_color_noise_vignette_bloom_hit_all_channels_equally():
    img = np.full((960, 1280, 3), 128, np.uint8)
    out = _sim(noise={"read_std": 4.0, "shot": 0.1}, vignette=0.4,
               bloom={"strength": 0.5, "threshold": 100, "sigma_px": 6.0}).transform(img).astype(np.float64)
    c = out[200:280, 280:360]     # tengah
    stds = [c[..., k].std() for k in range(3)]
    assert min(stds) > 2.0 and max(stds) / min(stds) < 1.3
    corner = out[:20, :20].mean(axis=(0, 1))
    assert np.all(corner < c.mean(axis=(0, 1)) - 10)       # vignette di semua kanal


def test_photometric_off_leaves_geometry_only():
    img = np.random.default_rng(1).integers(0, 255, (960, 1280, 3), dtype=np.uint8)
    s = _sim()
    out = s.transform(img)
    x0, y0, cw, ch = s.plan.crop
    ref = cv2.resize(img[y0:y0 + ch, x0:x0 + cw], (640, 480), interpolation=cv2.INTER_AREA)
    assert np.array_equal(out, ref)


# ------------------------------------------------------------------------------- timestamp + FPS
def _frames(n, fps=30.0):
    imgs = [np.full((960, 1280, 3), i % 255, np.uint8) for i in range(n)]
    return ArraySource(imgs, fps=fps, start_epoch_ms=1_700_000_000_000)


def test_timestamps_index_and_video_time_unchanged():
    raw = list(_frames(10))
    out = list(SimulatedSource(_frames(10), _sim()))
    assert [(f.ts_ms, f.index, f.video_t_s) for f in out] == [(f.ts_ms, f.index, f.video_t_s) for f in raw]
    assert all(f.meta["camera_profile"] == "uji" for f in out)


@pytest.mark.parametrize("cap,expect", [(15, 30), (25, 50), (30, 60), (60, 60)])
def test_fps_cap_drops_by_source_timestamp(cap, expect):
    src = SimulatedSource(_frames(60), _sim(fps=cap))
    out = list(src)
    assert abs(len(out) - expect) <= 1
    assert src.fps_hint == pytest.approx(min(30.0, cap))
    ts = [f.ts_ms for f in out]
    assert ts == sorted(ts)
    if cap < 30:
        gaps = np.diff(ts)
        assert gaps.min() >= 1000 / cap * 0.74


# ------------------------------------------------------------------------------- profil YAML + CLI helper
def test_all_profiles_and_modifiers_resolve_and_plan():
    data = load_profiles_file()
    names = list_profiles()
    assert {"imx219_noir_62", "imx219_77ir", "imx477_m12_75", "imx708_noir_wide_102"} <= set(names)
    for n in names:
        p = resolve_profile(n, data)
        sim = CameraSim(p, 70.0, log=lambda *_: None)
        out = sim.transform(np.zeros((1440, 1920, 3), np.uint8))
        assert out.shape[:2] == (p["output"][1], p["output"][0])
    p = resolve_profile("imx219_77ir+forklift+redup", data)
    assert p["vibration"]["shift_px_std"] > 0 and p["exposure"] < 1 and p["nir"] is True
    assert p["name"] == "imx219_77ir+forklift+redup"
    with pytest.raises(ProfileError):
        resolve_profile("tidak_ada", data)
    with pytest.raises(ProfileError):
        resolve_profile("imx219_77ir+tidak_ada", data)


def test_build_sim_none_keeps_old_behaviour_and_title():
    rt = {"camera_sim": {"profiles": "configs/camera_profiles.yaml", "phone_hfov_deg": None},
          "cabin": {"camera_profile": None}}
    assert build_sim(None, rt) is None and build_sim("none", rt) is None
    assert setup_camera_sim(rt, "cabin", None, None, {"type": "webcam"}) is None
    sim = build_sim("imx219_77ir", rt, phone_hfov_deg=68.0, log=lambda *_: None)
    assert sim.phone_hfov_deg == 68.0 and sim.nir
    assert window_title("Kabin", sim) == "Kabin [profil: imx219_77ir]"
    assert window_title("Kabin", None) == "Kabin"


def test_camera_sidecar_marks_simulated_recordings(tmp_path):
    v = tmp_path / "a.mp4"
    write_camera_sidecar(v, "imx219_77ir", raw=False, phone_hfov_deg=70.0)
    assert already_simulated(v) == "imx219_77ir"
    write_camera_sidecar(v, "imx219_77ir", raw=True, phone_hfov_deg=70.0)
    assert already_simulated(v) is None
    assert already_simulated(tmp_path / "tidak_ada.mp4") is None


def test_main_fps_only_runs_with_profile_on_video_file(tmp_path):
    from src.main import main
    p = tmp_path / "v.mp4"
    w = cv2.VideoWriter(str(p), cv2.VideoWriter_fourcc(*"mp4v"), 30, (640, 480))
    for i in range(12):
        w.write(np.full((480, 640, 3), 20 * i % 255, np.uint8))
    w.release()
    main(["--source", str(p), "--fps-only", "--no-show", "--fast", "--max-frames", "10",
          "--camera-profile", "imx219_77ir+forklift"])


# ------------------------------------------------------------------------------- jarak kamera depan end-to-end

def _phone_ground_image(d_m, in_w=1920, in_h=1440, phone_hfov=70.0, lateral_m=0.0):
    f = (in_w / 2) / math.tan(math.radians(phone_hfov / 2))
    K = np.array([[f, 0, (in_w - 1) / 2], [0, f, (in_h - 1) / 2], [0, 0, 1.0]])
    t = math.radians(THETA)
    R = np.array([[1, 0, 0], [0, math.cos(t), -math.sin(t)], [0, math.sin(t), math.cos(t)]])
    rvec, _ = cv2.Rodrigues(R)
    obj = np.array([[[lateral_m, MOUNT_H, d_m]]], np.float64)
    uv, _ = cv2.projectPoints(obj, rvec, np.zeros(3), K, np.zeros(5))
    u, v = uv.reshape(2)
    img = np.zeros((in_h, in_w, 3), np.uint8)
    cv2.circle(img, (int(round(u * 16)), int(round(v * 16))), 7 * 16, (255, 255, 255), -1, cv2.LINE_AA, shift=4)
    return img


def _centroid(img):
    g = img[..., 0].astype(np.float64)
    g[g < 40] = 0
    m = cv2.moments(g)
    return m["m10"] / m["m00"], m["m01"] / m["m00"]


@pytest.mark.parametrize("k1", [0.0, -0.05])
@pytest.mark.parametrize("d_m", [3.0, 5.0, 8.0])
def test_front_distance_end_to_end_through_profile(d_m, k1):
    sim = _sim(output=[1280, 720], hfov_deg=60.0, distortion={"k1": k1, "k2": 0.0})
    out = sim.transform(_phone_ground_image(d_m))
    block = sim.camera_block()
    assert block["image_size"] == [1280, 720] and block["dist"][0] == k1
    est = DistanceEstimator.from_config({**CAM_CFG, "camera": block})
    u, v = _centroid(out)
    fwd, _ = est.ground_point(u, v, 1280, 720)
    assert fwd == pytest.approx(d_m, rel=0.03)


def test_front_distance_with_original_phone_K_would_be_wrong():
    """Kontrol: K YAML (resize penuh) pada frame yang di-crop memberi jarak salah -> alasan set_camera()."""
    sim = _sim(output=[1280, 720], hfov_deg=50.0)
    u, v = _centroid(sim.transform(_phone_ground_image(8.0)))
    wrong = DistanceEstimator.from_config(CAM_CFG).ground_point(u, v, 1280, 720)[0]
    assert abs(wrong - 8.0) / 8.0 > 0.05     # ~10% pada kasus ini; bandingkan toleransi 3% di atas


def test_front_pipeline_set_camera_replaces_intrinsics_keeps_mount():
    sim = _sim(output=[1280, 720], hfov_deg=60.0)
    sim.transform(np.zeros((1440, 1920, 3), np.uint8))
    pipe = FrontPipeline(None, DistanceEstimator.from_config(CAM_CFG), None, None, calibrated=True)
    pipe.set_camera(CAM_CFG, sim.camera_block())
    assert pipe.distance.camera.K[0, 0] == pytest.approx(sim.plan.K_out[0, 0])
    assert pipe.distance.h_m == MOUNT_H
    assert pipe.calibrated is False      # HFOV HP belum diukur


def test_phone_hfov_cli_does_not_mark_calibrated():
    sim = CameraSim.from_name("imx477_m12_75", phone_hfov_deg=70.0, log=lambda *_: None)
    sim.transform(np.zeros((1440, 1920, 3), np.uint8))
    assert sim.camera_block()["calibrated"] is bool(load_profiles_file()["phone"].get("hfov_measured", False))


def test_distortion_clamp_samples_inside_phone_frame():
    """Target lebar + distorsi barrel, HP 70 deg: tepi kiri/kanan baris tengah tetap di dalam frame HP (tanpa
    pita hitam) dan HFOV efektif tidak melebihi HP."""
    sim = _sim(output=[1280, 720], hfov_deg=102.0, distortion={"k1": -0.05, "k2": 0.0})
    out = sim.transform(np.full((960, 1280, 3), 200, np.uint8))
    assert not sim.plan.fov_ok
    assert sim.plan.eff_hfov_deg <= 70.0 + 0.2
    mid = out[360]
    assert mid[:3].min() > 100 and mid[-3:].min() > 100


@pytest.mark.parametrize("name", [n for n in list_profiles()
                                  if resolve_profile(n, load_profiles_file()).get("nir")])
@pytest.mark.parametrize("modifier", ["", "+forklift+redup"])
def test_d09_still_detects_palm_over_lens_through_nir_profiles(name, modifier):
    """Skenario 20 (wajib): lensa ditutup -> frame seragam gelap atau terang (lampu IR memantul di telapak).
    Noise/vignette/bloom simulasi tidak boleh membuat D-09 gagal dengan threshold asli."""
    th = load_yaml(CONFIG_DIR / "thresholds.yaml")["D09_camera_block"]
    sim = CameraSim.from_name(name + modifier, phone_hfov_deg=70.0, log=lambda *_: None)
    for level in (8, 225):
        out = sim.transform(np.full((960, 1280, 3), level, np.uint8), 3)
        assert is_blocked(frame_stats(out, th.get("downscale_px", 160)), th), (name, modifier, level)
