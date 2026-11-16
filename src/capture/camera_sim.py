"""Simulasi kamera target (Jetson: NoIR/IMX219/IMX477/IMX708) dari frame kamera HP atau webcam.

Profil di configs/camera_profiles.yaml. Transform dipasang TEPAT setelah capture (SimulatedSource membungkus
FrameSource apa pun), jadi live, file video, dan replay lewat jalur yang sama. Timestamp (`ts_ms`, `index`,
`video_t_s`) TIDAK diubah; batas FPS membuang frame berdasarkan `ts_ms` sumber, bukan jam dinding.

Urutan per frame:
  1. geometri: crop tengah frame HP ke HFOV target + resize ke resolusi keluaran (+ distorsi barrel opsional,
     satu remap dengan K/dist yang SAMA dengan yang diberikan ke kamera depan)
  2. getaran (geser/rotasi acak kecil, RNG deterministik per frame)
  3. mono (nir: true) -> grayscale tanpa CLAHE (CLAHE tetap tugas pipeline kabin `simulate_nir`)
  4. exposure, vignette, bloom, noise sensor
  5. kompresi JPEG (opsional)

Geometri: kamera HP dianggap rectilinear dengan HFOV `phone.hfov_deg`, titik pusat di tengah frame:
  f_hp      = (W_hp/2) / tan(HFOV_hp/2)
  lebar_crop = 2 * f_hp * tan(HFOV_target/2)      (bukan rasio sudut)
  tinggi_crop = lebar_crop * h_out / w_out         (aspek keluaran; resize seragam, tanpa peregangan)
Jika crop yang dibutuhkan lebih besar dari frame HP (target lebih lebar dari HP, atau aspek HP kurang tinggi),
crop diperkecil seragam -> HFOV efektif < target, dan peringatan dicetak.
"""
from __future__ import annotations

import json
import math
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import cv2
import numpy as np

from src.config import deep_merge, load_yaml, resolve_path
from src.preprocess import to_gray

from . import Frame, FrameSource

DEFAULT_PROFILES_PATH = "configs/camera_profiles.yaml"
# Konstanta implementasi (bukan parameter kamera): ukuran bank noise dan faktor perkecilan blur bloom.
_NOISE_BANK = 4
_BLOOM_DOWNSCALE = 4


class ProfileError(ValueError):
    pass


# ------------------------------------------------------------------------------------------- profil
def load_profiles_file(path: str | Path | None = None) -> dict:
    return load_yaml(resolve_path(path or DEFAULT_PROFILES_PATH))


def list_profiles(path: str | Path | None = None, role: str | None = None) -> list[str]:
    profs = load_profiles_file(path).get("profiles", {})
    return [k for k, v in profs.items() if role is None or v.get("role") == role]


def resolve_profile(name: str, data: dict) -> dict:
    """'profil+modifier+modifier' -> dict lengkap (defaults <- profil <- modifier)."""
    parts = [p.strip() for p in str(name).split("+") if p.strip()]
    if not parts:
        raise ProfileError("Nama profil kosong")
    profs, mods = data.get("profiles", {}), data.get("modifiers", {}) or {}
    if parts[0] not in profs:
        raise ProfileError(f"Profil kamera tidak dikenal: {parts[0]!r}. Pilihan: {', '.join(profs)}")
    prof = deep_merge(data.get("defaults", {}), profs[parts[0]])
    for m in parts[1:]:
        if m not in mods:
            raise ProfileError(f"Modifier tidak dikenal: {m!r}. Pilihan: {', '.join(mods)}")
        prof = deep_merge(prof, mods[m])
    prof["name"] = "+".join(parts)
    for k in ("output", "hfov_deg"):
        if prof.get(k) is None:
            raise ProfileError(f"Profil {prof['name']}: field '{k}' wajib diisi")
    return prof


# ------------------------------------------------------------------------------------------- geometri
def _distort_x(xu: float, k1: float, k2: float) -> float:
    r2 = xu * xu
    return xu * (1.0 + k1 * r2 + k2 * r2 * r2)


def focal_for_hfov(width_px: int, hfov_deg: float, k1: float = 0.0, k2: float = 0.0) -> float:
    """fx (piksel) agar tepi kiri/kanan frame (setelah distorsi radial k1,k2) jatuh di +-HFOV/2."""
    xd = _distort_x(math.tan(math.radians(hfov_deg) / 2.0), k1, k2)
    if xd <= 0:
        raise ProfileError("Distorsi terlalu kuat untuk HFOV ini (model radial tidak monoton)")
    return (width_px / 2.0) / xd


def hfov_for_focal(width_px: float, fx: float) -> float:
    return math.degrees(2.0 * math.atan((width_px / 2.0) / fx))


def _undistort_x(xd: float, k1: float, k2: float) -> float:
    """Kebalikan _distort_x di sepanjang sumbu (Newton)."""
    xu = xd
    for _ in range(30):
        g = _distort_x(xu, k1, k2) - xd
        dg = 1.0 + 3.0 * k1 * xu * xu + 5.0 * k2 * xu ** 4
        if not dg:
            break
        xu -= g / dg
    return xu


@dataclass
class CropPlan:
    in_size: tuple[int, int]         # (w, h) frame HP
    out_size: tuple[int, int]        # (w, h) keluaran
    crop: tuple[int, int, int, int]  # x0, y0, w, h (piksel HP, tanpa distorsi)
    f_phone: float
    K_out: np.ndarray                # 3x3 kamera simulasi pada out_size
    dist_out: np.ndarray             # [k1, k2, 0, 0, 0]
    eff_hfov_deg: float              # HFOV efektif keluaran (tepi frame, termasuk distorsi)
    target_hfov_deg: float
    fov_ok: bool                     # False = target tidak bisa ditiru penuh (crop diperkecil)
    upscale: float                   # >1 = frame HP diperbesar (detail kurang dari kamera target)
    warnings: list[str] = field(default_factory=list)


def plan_crop(in_w: int, in_h: int, phone_hfov_deg: float, out_w: int, out_h: int, target_hfov_deg: float,
              k1: float = 0.0, k2: float = 0.0, warn_upscale_above: float = 1.15) -> CropPlan:
    f_p = (in_w / 2.0) / math.tan(math.radians(phone_hfov_deg) / 2.0)
    fx_t = focal_for_hfov(out_w, target_hfov_deg, k1, k2)
    # daerah HP (TANPA distorsi) yang dilihat kamera target, diukur dari tepi tengah kiri/kanan dan atas/bawah
    # (dengan distorsi barrel, koordinat tanpa distorsi di tepi > koordinat terdistorsi)
    need_w = 2.0 * f_p * _undistort_x((out_w / 2.0) / fx_t, k1, k2)
    need_h = 2.0 * f_p * _undistort_x((out_h / 2.0) / fx_t, k1, k2)
    s = min(1.0, in_w / need_w, in_h / need_h)
    cw = max(2, min(in_w, int(round(need_w * s))))
    ch = max(2, min(in_h, int(round(need_h * s))))
    x0, y0 = (in_w - cw) // 2, (in_h - ch) // 2
    if k1 == 0.0 and k2 == 0.0:
        # K keluaran dari crop BULAT yang benar-benar dipakai (resize cw->out_w, ch->out_h)
        fx, fy = f_p * out_w / cw, f_p * out_h / ch
    else:
        # remap: perkecil kamera target seragam (barrel cekung -> tepi tetap di dalam frame HP)
        fx = fy = fx_t / s
    K = np.array([[fx, 0.0, (out_w - 1) / 2.0], [0.0, fy, (out_h - 1) / 2.0], [0.0, 0.0, 1.0]])
    dist = np.array([k1, k2, 0.0, 0.0, 0.0])
    eff = math.degrees(2.0 * math.atan(_undistort_x((out_w / 2.0) / fx, k1, k2)))
    warns = []
    fov_ok = s >= 0.999
    if not fov_ok:
        phone_vfov = math.degrees(2 * math.atan((in_h / 2.0) / f_p))
        if in_w / need_w <= in_h / need_h:
            warns.append(
                f"HFOV target {target_hfov_deg:.1f} deg LEBIH LEBAR dari kamera HP ({phone_hfov_deg:.1f} deg): "
                f"tidak bisa ditiru, HFOV efektif hanya {eff:.1f} deg. Pakai lensa ultrawide HP (0,5x) "
                f"lalu set --phone-hfov sesuai lensa itu.")
        else:
            warns.append(
                f"Aspek frame HP {in_w}x{in_h} (VFOV {phone_vfov:.1f} deg) kurang tinggi untuk keluaran "
                f"{out_w}x{out_h} pada HFOV {target_hfov_deg:.1f} deg: HFOV efektif hanya {eff:.1f} deg. "
                f"Set stream HP ke aspek 4:3 (mis. 1280x960 / 1440x1080) atau pakai lensa ultrawide.")
    up = out_w / cw
    if up > warn_upscale_above:
        warns.append(f"Frame HP di-upscale {up:.2f}x (crop {cw}x{ch} -> {out_w}x{out_h}): detail lebih sedikit "
                     f"dari kamera target. Naikkan resolusi stream HP bila jaringan kuat.")
    return CropPlan((in_w, in_h), (out_w, out_h), (x0, y0, cw, ch), f_p, K, dist, eff, target_hfov_deg,
                    fov_ok, up, warns)


# ------------------------------------------------------------------------------------------- simulator
class CameraSim:
    """Transform satu frame HP -> frame kamera target. Plan geometri dihitung saat frame pertama (ukuran HP)."""

    def __init__(self, profile: dict, phone_hfov_deg: float, phone_hfov_measured: bool = False,
                 log=print):
        self.p = profile
        self.name = profile["name"]
        self.phone_hfov_deg = float(phone_hfov_deg)
        self.phone_hfov_measured = bool(phone_hfov_measured)
        self.out_w, self.out_h = (int(v) for v in profile["output"])
        self.hfov = float(profile["hfov_deg"])
        d = profile.get("distortion") or {}
        self.k1, self.k2 = float(d.get("k1", 0.0) or 0.0), float(d.get("k2", 0.0) or 0.0)
        self.fps = float(profile["fps"]) if profile.get("fps") else None
        self.nir = bool(profile.get("nir", False))
        self.seed = int(profile.get("seed", 0) or 0)
        self.log = log
        self.plan: CropPlan | None = None
        self._maps: tuple | None = None          # (pre_scale, map1, map2) untuk distorsi
        self._vig: np.ndarray | None = None
        self._bank: np.ndarray | None = None
        self._next_ms: float | None = None
        self.timing_ms = 0.0                     # rata-rata waktu transform (EMA)

    # ---- konstruksi ------------------------------------------------------------------------------
    @classmethod
    def from_name(cls, name: str, profiles_path: str | Path | None = None,
                  phone_hfov_deg: float | None = None, log=print) -> "CameraSim":
        data = load_profiles_file(profiles_path)
        phone = data.get("phone", {}) or {}
        hfov = phone_hfov_deg if phone_hfov_deg is not None else phone.get("hfov_deg")
        if hfov is None:
            raise ProfileError("phone.hfov_deg belum diisi di camera_profiles.yaml (atau pakai --phone-hfov)")
        # hanya flag YAML yang menandai "terukur"; --phone-hfov saja tidak (bisa angka tebakan)
        return cls(resolve_profile(name, data), hfov, bool(phone.get("hfov_measured", False)), log)

    # ---- geometri --------------------------------------------------------------------------------
    def configure(self, in_w: int, in_h: int) -> CropPlan:
        if self.plan is not None and self.plan.in_size == (in_w, in_h):
            return self.plan
        self.plan = plan_crop(in_w, in_h, self.phone_hfov_deg, self.out_w, self.out_h, self.hfov,
                              self.k1, self.k2, float(self.p.get("warn_upscale_above", 1.15)))
        self._maps = None
        pl = self.plan
        self.log(f"[camera-sim] profil {self.name}: HP {in_w}x{in_h} HFOV {self.phone_hfov_deg:.1f} deg -> "
                 f"crop {pl.crop[2]}x{pl.crop[3]} -> {self.out_w}x{self.out_h}, HFOV efektif "
                 f"{pl.eff_hfov_deg:.1f} deg (target {self.hfov:.1f}), fx={pl.K_out[0, 0]:.1f}"
                 + (f", dist k1={self.k1:g} k2={self.k2:g}" if self.has_distortion else "")
                 + (", NIR mono" if self.nir else ""))
        for w in pl.warnings:
            self.log(f"[camera-sim] PERINGATAN: {w}")
        return pl

    @property
    def has_distortion(self) -> bool:
        return abs(self.k1) > 0 or abs(self.k2) > 0

    def camera_block(self) -> dict:
        """Blok `camera:` untuk configs/front_camera.yaml yang SESUAI dengan frame simulasi (bukan HP)."""
        if self.plan is None:
            raise RuntimeError("camera_block() dipanggil sebelum frame pertama (configure)")
        return {
            "calibrated": self.phone_hfov_measured,
            "image_size": [self.out_w, self.out_h],
            "fisheye": False,
            "K": self.plan.K_out.tolist(),
            "dist": self.plan.dist_out.tolist(),
            "rms_px": None,
            "source": f"camera_profile:{self.name}",
        }

    def _geometry(self, img: np.ndarray) -> np.ndarray:
        pl = self.plan
        x0, y0, cw, ch = pl.crop
        if not self.has_distortion:
            crop = img[y0:y0 + ch, x0:x0 + cw]
            interp = cv2.INTER_AREA if cw >= self.out_w else cv2.INTER_LINEAR
            return cv2.resize(crop, (self.out_w, self.out_h), interpolation=interp)
        if self._maps is None:
            # perkecil dulu (anti-aliasing) bila frame HP lebih rapat dari keluaran
            pre = min(1.0, self.out_w / cw)
            uv = np.stack(np.meshgrid(np.arange(self.out_w, dtype=np.float64),
                                      np.arange(self.out_h, dtype=np.float64)), -1).reshape(-1, 1, 2)
            # piksel keluaran (terdistorsi) -> koordinat ternormalisasi tanpa distorsi (fungsi yang sama dengan
            # CameraModel.normalize di kamera depan) -> piksel HP
            n = cv2.undistortPoints(uv, pl.K_out, pl.dist_out).reshape(self.out_h, self.out_w, 2)
            in_w, in_h = pl.in_size
            cx, cy = (in_w - 1) / 2.0, (in_h - 1) / 2.0
            mx = ((cx + pl.f_phone * n[..., 0]) + 0.5) * pre - 0.5
            my = ((cy + pl.f_phone * n[..., 1]) + 0.5) * pre - 0.5
            self._maps = (pre, mx.astype(np.float32), my.astype(np.float32))
        pre, mx, my = self._maps
        src = img
        if pre < 1.0:
            in_w, in_h = pl.in_size
            src = cv2.resize(img, (max(1, int(round(in_w * pre))), max(1, int(round(in_h * pre)))),
                             interpolation=cv2.INTER_AREA)
        return cv2.remap(src, mx, my, cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT)

    # ---- fotometri -------------------------------------------------------------------------------
    def _rng(self, index: int) -> np.random.Generator:
        return np.random.default_rng([self.seed, int(index)])

    def _vibrate(self, img: np.ndarray, rng: np.random.Generator) -> np.ndarray:
        v = self.p.get("vibration") or {}
        s, r = float(v.get("shift_px_std", 0) or 0), float(v.get("rot_deg_std", 0) or 0)
        if s <= 0 and r <= 0:
            return img
        dx, dy = rng.normal(0.0, s, 2) if s > 0 else (0.0, 0.0)
        ang = rng.normal(0.0, r) if r > 0 else 0.0
        M = cv2.getRotationMatrix2D(((self.out_w - 1) / 2.0, (self.out_h - 1) / 2.0), float(ang), 1.0)
        M[0, 2] += dx
        M[1, 2] += dy
        return cv2.warpAffine(img, M, (self.out_w, self.out_h), borderMode=cv2.BORDER_REFLECT101)

    def _gain_map(self, channels: int) -> np.ndarray | None:
        """exposure x vignette (dicache). Vignette: tengah 1, sudut 1 - vignette."""
        exposure = float(self.p.get("exposure", 1.0) or 1.0)
        vig = float(self.p.get("vignette", 0.0) or 0.0)
        if exposure == 1.0 and vig <= 0:
            return None
        if self._vig is None or self._vig.shape[2:] != ((channels,) if channels > 1 else ()):
            yy, xx = np.mgrid[0:self.out_h, 0:self.out_w].astype(np.float32)
            r2 = ((xx - (self.out_w - 1) / 2) / (self.out_w / 2)) ** 2 + \
                 ((yy - (self.out_h - 1) / 2) / (self.out_h / 2)) ** 2
            g = (exposure * np.clip(1.0 - vig * r2 / 2.0, 0.0, 1.0)).astype(np.float32)
            self._vig = cv2.merge([g] * channels) if channels > 1 else g
        return self._vig

    def _noise_bank(self, shape: tuple) -> np.ndarray:
        """_NOISE_BANK frame noise normal baku (dibuat sekali, seed profil). Per frame dipilih satu dan tandanya
        dibalik bergantian -> pola noise berulang tiap 2*_NOISE_BANK frame (cukup untuk uji model, jauh lebih murah
        daripada membangkitkan jutaan angka acak per frame)."""
        if self._bank is None or self._bank.shape[1:] != shape:
            rng = np.random.default_rng(self.seed)
            self._bank = rng.standard_normal((_NOISE_BANK, *shape), dtype=np.float32)
        return self._bank

    def _photometric(self, img: np.ndarray, index: int) -> np.ndarray:
        """Semua di uint8 dengan operasi OpenCV bersaturasi (bukan float numpy): ~10x lebih cepat, sehingga profil
        2 MP tetap jauh di bawah 1 periode frame. Operasi per kanal memakai LUT / array (bukan skalar cv2, yang
        hanya mengenai kanal 0)."""
        p = self.p
        bloom = p.get("bloom") or {}
        b_str = float(bloom.get("strength", 0.0) or 0.0)
        noise = p.get("noise") or {}
        read_std, shot = float(noise.get("read_std", 0) or 0), float(noise.get("shot", 0) or 0)
        ch = 1 if img.ndim == 2 else img.shape[2]
        gain = self._gain_map(ch)
        out = img
        if gain is not None:
            out = cv2.multiply(out, gain, dtype=cv2.CV_8U)
        if b_str > 0:
            # pendaran di sekitar area sangat terang (pantulan IR kulit/kacamata/rompi reflektif);
            # blur di 1/4 resolusi lalu diperbesar (sigma ikut diskalakan) -> murah
            thr = float(bloom.get("threshold", 200))
            hi = cv2.LUT(out, np.clip(np.arange(256) - thr, 0, 255).astype(np.uint8))
            q = _BLOOM_DOWNSCALE
            small = cv2.resize(hi, (max(1, self.out_w // q), max(1, self.out_h // q)), interpolation=cv2.INTER_AREA)
            small = cv2.GaussianBlur(small, (0, 0), max(0.5, float(bloom.get("sigma_px", 6.0)) / q))
            up = cv2.resize(small, (self.out_w, self.out_h), interpolation=cv2.INTER_LINEAR)
            out = cv2.addWeighted(out, 1.0, up, b_str, 0.0)
        if shot > 0 or read_std > 0:
            std = cv2.LUT(out, np.sqrt(read_std * read_std + shot * np.arange(256)).astype(np.float32))
            bank = self._noise_bank(std.shape)
            k = int(index) % (2 * _NOISE_BANK)
            n16 = cv2.multiply(bank[k % _NOISE_BANK], std, scale=-1.0 if k >= _NOISE_BANK else 1.0,
                               dtype=cv2.CV_16S)
            out = cv2.add(out, n16, dtype=cv2.CV_8U)
        return out

    # ---- API utama -------------------------------------------------------------------------------
    def keep(self, ts_ms: int) -> bool:
        """Batas FPS berbasis timestamp sumber. Toleransi 1/4 periode untuk jitter timestamp."""
        if not self.fps:
            return True
        period = 1000.0 / self.fps
        if self._next_ms is None or ts_ms >= self._next_ms - period / 4.0:
            base = ts_ms if self._next_ms is None or ts_ms - self._next_ms > period else self._next_ms
            self._next_ms = base + period
            return True
        return False

    def transform(self, img: np.ndarray, index: int = 0) -> np.ndarray:
        """Gambar HP (BGR) -> gambar kamera target (BGR 3 channel)."""
        t0 = time.perf_counter()
        h, w = img.shape[:2]
        self.configure(w, h)
        rng = self._rng(index)
        out = self._geometry(img)
        out = self._vibrate(out, rng)
        if self.nir:
            out = to_gray(out)
        out = self._photometric(out, index)
        q = self.p.get("jpeg_quality")
        if q:
            ok, buf = cv2.imencode(".jpg", out, [cv2.IMWRITE_JPEG_QUALITY, int(q)])
            if ok:
                out = cv2.imdecode(buf, cv2.IMREAD_UNCHANGED)
        if out.ndim == 2:
            out = cv2.cvtColor(out, cv2.COLOR_GRAY2BGR)
        dt = (time.perf_counter() - t0) * 1000
        self.timing_ms = dt if self.timing_ms == 0 else 0.9 * self.timing_ms + 0.1 * dt
        return out

    def apply(self, frame: Frame) -> Optional[Frame]:
        """Frame HP -> Frame simulasi (timestamp tetap), atau None jika dibuang oleh batas FPS."""
        if not self.keep(frame.ts_ms):
            return None
        meta = dict(frame.meta)
        meta["camera_profile"] = self.name
        return Frame(self.transform(frame.image, frame.index), frame.ts_ms, frame.index, frame.video_t_s, meta)

    def reset(self) -> None:
        self._next_ms = None

    def summary(self) -> str:
        p = self.p
        return (f"{self.name}: {p.get('desc', '')} | keluaran {self.out_w}x{self.out_h} @ {self.fps or '-'} fps, "
                f"HFOV {self.hfov:g} deg, NIR {'ya' if self.nir else 'tidak'}")


class SimulatedSource(FrameSource):
    """Bungkus FrameSource: setiap frame lewat CameraSim. Interface sama -> live dan replay identik."""

    def __init__(self, inner: FrameSource, sim: CameraSim):
        self.inner = inner
        self.sim = sim
        self.fps_hint = min(inner.fps_hint, sim.fps) if sim.fps else inner.fps_hint

    def read(self) -> Optional[Frame]:
        while True:
            f = self.inner.read()
            if f is None:
                return None
            out = self.sim.apply(f)
            if out is not None:
                return out

    def release(self) -> None:
        self.inner.release()

    def __getattr__(self, item):   # frame_count, path, dst. dari sumber asli
        if item in ("inner", "sim"):
            raise AttributeError(item)
        return getattr(self.inner, item)


# ------------------------------------------------------------------------------------------- helper CLI
def profiles_path_from_runtime(rt: dict) -> str:
    return (rt.get("camera_sim") or {}).get("profiles", DEFAULT_PROFILES_PATH)


def build_sim(name: str | None, rt: dict, phone_hfov_deg: float | None = None, log=print) -> CameraSim | None:
    """None/'' /'none' -> tanpa simulasi (perilaku lama)."""
    if not name or str(name).lower() in ("none", "null", "off"):
        return None
    if phone_hfov_deg is None:
        phone_hfov_deg = (rt.get("camera_sim") or {}).get("phone_hfov_deg")
    sim = CameraSim.from_name(name, profiles_path_from_runtime(rt), phone_hfov_deg, log)
    log(f"[camera-sim] {sim.summary()}")
    if sim.p.get("note"):
        log(f"[camera-sim] catatan: {sim.p['note']}")
    return sim


def setup_camera_sim(rt: dict, section: str, profile_arg: str | None, phone_hfov: float | None,
                     src_cfg: dict, log=print) -> CameraSim | None:
    """CameraSim untuk `cabin`/`front` (CLI > runtime.yaml <section>.camera_profile), atau None."""
    name = profile_arg if profile_arg is not None else (rt.get(section) or {}).get("camera_profile")
    sim = build_sim(name, rt, phone_hfov, log)
    if sim is not None and src_cfg.get("type") == "file":
        prev = already_simulated(src_cfg.get("path") or "")
        if prev:
            log(f"[camera-sim] PERINGATAN: video ini direkam dengan profil {prev!r}; simulasi diterapkan DUA KALI. "
                f"Pakai --camera-profile none untuk replay apa adanya.")
    return sim


def window_title(base: str, sim: CameraSim | None) -> str:
    return f"{base} [profil: {sim.name}]" if sim is not None else base


def camera_sidecar_path(video: str | Path) -> Path:
    """<video>.camera.json: profil yang dipakai saat merekam (tools/recorder.py)."""
    return Path(str(video) + ".camera.json")


def write_camera_sidecar(video: str | Path, profile: str | None, raw: bool, phone_hfov_deg: float | None) -> None:
    with open(camera_sidecar_path(video), "w", encoding="utf-8") as f:
        json.dump({"camera_profile": profile, "raw": bool(raw), "phone_hfov_deg": phone_hfov_deg}, f)


def read_camera_sidecar(video: str | Path) -> dict | None:
    p = camera_sidecar_path(video)
    if not p.exists():
        return None
    try:
        return json.loads(p.read_text(encoding="utf-8"))
    except (OSError, ValueError):
        return None


def already_simulated(video: str | Path) -> str | None:
    """Nama profil jika video sudah berisi frame simulasi (jangan disimulasikan dua kali)."""
    meta = read_camera_sidecar(video)
    if meta and meta.get("camera_profile") and not meta.get("raw", False):
        return str(meta["camera_profile"])
    return None


def main(argv=None) -> None:
    """python -m src.capture.camera_sim --list [--role kabin|depan]  (dipakai run.bat)"""
    import argparse
    ap = argparse.ArgumentParser(description="Daftar profil kamera (configs/camera_profiles.yaml)")
    ap.add_argument("--list", action="store_true")
    ap.add_argument("--role", default=None, help="kabin | depan")
    ap.add_argument("--profiles", default=None, help="path camera_profiles.yaml")
    a = ap.parse_args(argv)
    data = load_profiles_file(a.profiles)
    print("  Profil kamera" + (f" ({a.role})" if a.role else "") + ":")
    for n in list_profiles(a.profiles, a.role):
        p = resolve_profile(n, data)
        print(f"    {n:24s} {p['output'][0]}x{p['output'][1]}  HFOV {p['hfov_deg']:g}  "
              f"{'NIR ' if p.get('nir') else 'warna'}  [{p.get('role', '-')}]  {p.get('desc', '')}")
    mods = ", ".join(data.get("modifiers", {}) or {})
    if mods:
        print(f"  Modifier (tambahkan dengan '+', mis. imx219_77ir+forklift): {mods}")
    print(f"  HFOV kamera HP default: {data.get('phone', {}).get('hfov_deg')} deg (ubah dengan --phone-hfov)")


if __name__ == "__main__":
    main()
