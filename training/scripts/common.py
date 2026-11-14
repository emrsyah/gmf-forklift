"""Utilitas bersama untuk pipeline data kabin (L5).

- Path repo + import `src.*` (simulate_nir) tanpa instalasi paket.
- Baca config YAML (training/configs/cabin_build.yaml).
- I/O label YOLO, IoU, sampling deterministik.
"""
from __future__ import annotations

import hashlib
import random
import sys
from dataclasses import dataclass
from pathlib import Path

import numpy as np
import yaml

REPO = Path(__file__).resolve().parents[2]
if str(REPO) not in sys.path:
    sys.path.insert(0, str(REPO))

DEFAULT_BUILD_CFG = REPO / "training" / "configs" / "cabin_build.yaml"
IMG_EXTS = {".jpg", ".jpeg", ".png", ".bmp", ".webp"}


def load_yaml(path: str | Path) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def rpath(p: str | Path) -> Path:
    """Path relatif terhadap root repo -> absolut."""
    p = Path(p)
    return p if p.is_absolute() else (REPO / p)


def load_build_cfg(path: str | Path | None = None) -> dict:
    return load_yaml(path or DEFAULT_BUILD_CFG)


def class_names(cfg: dict) -> list[str]:
    return list(cfg["classes"])


def clahe_params(cfg: dict) -> tuple[float, int]:
    """clip/tile CLAHE diambil dari configs/runtime.yaml (cabin.clahe_*) supaya sama dengan runtime."""
    nir = cfg.get("nir", {})
    clip, tile = float(nir.get("clahe_clip_default", 2.0)), int(nir.get("clahe_tile_default", 8))
    rt = rpath(nir.get("runtime_config", "configs/runtime.yaml"))
    try:
        cab = load_yaml(rt).get("cabin", {})
        clip = float(cab.get("clahe_clip", clip))
        tile = int(cab.get("clahe_tile", tile))
    except OSError:
        pass
    return clip, tile


# ---------------------------------------------------------------- label YOLO
@dataclass
class Box:
    cls: int
    cx: float
    cy: float
    w: float
    h: float
    pseudo: bool = False  # True jika dibuat otomatis (pose/derive), bukan label manusia

    def xyxy(self) -> tuple[float, float, float, float]:
        return (self.cx - self.w / 2, self.cy - self.h / 2, self.cx + self.w / 2, self.cy + self.h / 2)

    @staticmethod
    def from_xyxy(cls: int, x1: float, y1: float, x2: float, y2: float, pseudo: bool = False) -> "Box":
        x1, x2 = max(0.0, min(x1, x2)), min(1.0, max(x1, x2))
        y1, y2 = max(0.0, min(y1, y2)), min(1.0, max(y1, y2))
        return Box(cls, (x1 + x2) / 2, (y1 + y2) / 2, x2 - x1, y2 - y1, pseudo)

    def line(self) -> str:
        return f"{self.cls} {self.cx:.6f} {self.cy:.6f} {self.w:.6f} {self.h:.6f}"


def read_yolo(path: Path) -> list[Box]:
    """Baca label YOLO. Baris poligon (segmentasi) diubah ke bbox."""
    out: list[Box] = []
    if not path.exists():
        return out
    for ln in path.read_text(encoding="utf-8", errors="ignore").splitlines():
        v = ln.split()
        if len(v) < 5:
            continue
        c = int(float(v[0]))
        nums = [float(x) for x in v[1:]]
        if len(nums) == 4:
            out.append(Box(c, *nums))
        elif len(nums) >= 6 and len(nums) % 2 == 0:
            xs, ys = nums[0::2], nums[1::2]
            out.append(Box.from_xyxy(c, min(xs), min(ys), max(xs), max(ys)))
    return out


def write_yolo(path: Path, boxes: list[Box]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("".join(b.line() + "\n" for b in boxes), encoding="utf-8")


def iou(a, b) -> float:
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    ua = (a[2] - a[0]) * (a[3] - a[1]) + (b[2] - b[0]) * (b[3] - b[1]) - inter
    return inter / ua if ua > 0 else 0.0


def ioa(a, b) -> float:
    """Fraksi area a yang tertutup b."""
    ix1, iy1, ix2, iy2 = max(a[0], b[0]), max(a[1], b[1]), min(a[2], b[2]), min(a[3], b[3])
    inter = max(0.0, ix2 - ix1) * max(0.0, iy2 - iy1)
    aa = (a[2] - a[0]) * (a[3] - a[1])
    return inter / aa if aa > 0 else 0.0


def sample(items: list, k: int | None, seed: int) -> list:
    items = sorted(items)
    if k is None or k >= len(items):
        return items
    rnd = random.Random(seed)
    return sorted(rnd.sample(items, k))


def stable_unit(key: str, seed: int) -> float:
    """Hash stabil -> [0,1). Dipakai untuk split per grup (tidak bergantung urutan file)."""
    h = hashlib.sha1(f"{seed}:{key}".encode("utf-8")).hexdigest()
    return int(h[:12], 16) / float(16 ** 12)


def list_images(d: Path) -> list[Path]:
    return sorted(p for p in d.iterdir() if p.suffix.lower() in IMG_EXTS) if d.exists() else []


def imread(path: Path) -> np.ndarray | None:
    import cv2
    data = np.fromfile(str(path), dtype=np.uint8)  # aman untuk path non-ASCII di Windows
    return cv2.imdecode(data, cv2.IMREAD_COLOR) if data.size else None


def imwrite(path: Path, img: np.ndarray, quality: int = 95) -> bool:
    import cv2
    path.parent.mkdir(parents=True, exist_ok=True)
    ext = path.suffix.lower()
    params = [cv2.IMWRITE_JPEG_QUALITY, int(quality)] if ext in (".jpg", ".jpeg") else []
    ok, buf = cv2.imencode(ext, img, params)
    if ok:
        buf.tofile(str(path))
    return bool(ok)
