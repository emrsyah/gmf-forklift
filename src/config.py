"""Loader config YAML. Semua angka (threshold, path, FPS) masuk lewat sini, bukan hard-code."""
from __future__ import annotations

import copy
from pathlib import Path
from typing import Any

import yaml

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_DIR = REPO_ROOT / "configs"


def load_yaml(path: str | Path) -> dict:
    p = Path(path)
    if not p.is_absolute():
        p = REPO_ROOT / p
    with open(p, "r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def deep_merge(base: dict, override: dict) -> dict:
    out = copy.deepcopy(base)
    for k, v in (override or {}).items():
        if isinstance(v, dict) and isinstance(out.get(k), dict):
            out[k] = deep_merge(out[k], v)
        else:
            out[k] = copy.deepcopy(v)
    return out


def resolve_path(p: str | Path | None) -> Path | None:
    if p is None:
        return None
    p = Path(p)
    return p if p.is_absolute() else REPO_ROOT / p


class Config:
    """Gabungan runtime.yaml, thresholds.yaml, events.yaml."""

    def __init__(self, runtime: dict, thresholds: dict, events: dict):
        self.runtime = runtime
        self.thresholds = thresholds
        self.events = events

    @classmethod
    def load(
        cls,
        runtime: str | Path = CONFIG_DIR / "runtime.yaml",
        thresholds: str | Path = CONFIG_DIR / "thresholds.yaml",
        events: str | Path = CONFIG_DIR / "events.yaml",
        overrides: dict | None = None,
    ) -> "Config":
        cfg = cls(load_yaml(runtime), load_yaml(thresholds), load_yaml(events))
        if overrides:
            cfg.runtime = deep_merge(cfg.runtime, overrides.get("runtime", {}))
            cfg.thresholds = deep_merge(cfg.thresholds, overrides.get("thresholds", {}))
            cfg.events = deep_merge(cfg.events, overrides.get("events", {}))
        return cfg


def set_dotted(d: dict, dotted: str, value: Any) -> None:
    """set_dotted(cfg, 'cabin.source.type', 'file')"""
    keys = dotted.split(".")
    cur = d
    for k in keys[:-1]:
        cur = cur.setdefault(k, {})
    cur[keys[-1]] = value
