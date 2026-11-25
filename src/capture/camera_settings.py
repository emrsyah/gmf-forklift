"""Pengaturan kamera dari dashboard MINERVA (tukar kamera kabin/depan, putar tiap kamera).

Alur: dashboard -> server MINERVA -> agen edge di Jetson (`{type: 'camera', action: 'swap'|'rotate'}`) ->
agen menulis `camera_config.json` ({"swapped", "cabinRotate", "frontRotate"}) lalu me-restart detector.
Detector membaca file itu saat start dan menimpa config runtime:

- swapped: kamera kabin memakai `front_device`, kamera depan memakai `cabin_device` (dan sebaliknya).
  Path kamera diambil dari blok `camera_settings`, BUKAN dari `source.device`: agen juga mengedit baris
  `device:` di jetson.yaml, dan deploy dari git menimpa editan itu. File JSON = satu-satunya sumber, jadi
  hasilnya sama setelah deploy maupun setelah editan agen.
- cabinRotate -> cabin.source.rotate (diputar sebelum pipeline, lihat RotatedSource).
- frontRotate -> front_preview.rotate (deteksi depan + `rot` di pesan kotak untuk dashboard).

Config (runtime):
  camera_settings:
    file: /home/jetson/minerva/camera_config.json
    cabin_device: /dev/v4l/by-id/...      # kamera yang dipasang di kabin saat swapped = false
    front_device: /dev/v4l/by-id/...
"""
from __future__ import annotations

import json
from pathlib import Path

_DEGS = (0, 90, 180, 270)


def _deg(v) -> int | None:
    try:
        d = int(v) % 360
    except (TypeError, ValueError):
        return None
    return d if d in _DEGS else None


def apply_camera_settings(rt: dict) -> str | None:
    """Terapkan file pengaturan kamera ke config runtime `rt` (in place). Mengembalikan ringkasan untuk log,
    atau None jika fitur mati / file belum ada."""
    cs = rt.get("camera_settings") or {}
    path = cs.get("file")
    if not path:
        return None
    p = Path(path).expanduser()
    try:
        data = json.loads(p.read_text())
    except FileNotFoundError:
        data = {}
    except (OSError, ValueError) as e:
        return f"{p} tidak terbaca ({e}); pakai config apa adanya"
    if not isinstance(data, dict):
        return f"{p} bukan objek JSON; pakai config apa adanya"
    cab_src = rt.setdefault("cabin", {}).setdefault("source", {})
    front_src = rt.setdefault("front", {}).setdefault("source", {})
    parts = []
    swapped = data.get("swapped") is True
    cab_dev, front_dev = cs.get("cabin_device"), cs.get("front_device")
    if cab_dev and front_dev:
        cab_src["device"], front_src["device"] = (front_dev, cab_dev) if swapped else (cab_dev, front_dev)
        parts.append(f"tukar={'ya' if swapped else 'tidak'}")
    for key, target, name in (("cabinRotate", cab_src, "kabin"),
                              ("frontRotate", rt.setdefault("front_preview", {}), "depan")):
        if key in data:
            d = _deg(data[key])
            if d is None:
                parts.append(f"{key}={data[key]!r} diabaikan (0/90/180/270)")
                continue
            target["rotate"] = d
        parts.append(f"putar {name} {int(target.get('rotate', 0) or 0)}")
    return f"{p}: " + ", ".join(parts)
