"""Uji wiring buzzer GPIO di Jetson: bunyikan pola A-02 (1 pulsa) lalu A-03 (2 pulsa) beberapa kali.

  python -m tools.jetson.buzzer_test                # pakai alert.* dari runtime.yaml + configs/jetson.yaml
  python -m tools.jetson.buzzer_test --pin 12 --mode BOARD --active-low
"""
from __future__ import annotations

import argparse
import time

from src.config import Config, load_yaml
from src.io.alert import GpioBuzzer


def main() -> None:
    cfg = Config.load(overrides=load_yaml("configs/jetson.yaml")).runtime.get("alert", {})
    ap = argparse.ArgumentParser()
    ap.add_argument("--pin", type=int, default=cfg.get("gpio_pin", 12))
    ap.add_argument("--mode", default=cfg.get("gpio_mode", "BOARD"))
    ap.add_argument("--active-low", action="store_true", default=bool(cfg.get("active_low", False)))
    ap.add_argument("--ms", type=int, default=cfg.get("beep_ms", 250))
    ap.add_argument("--repeat", type=int, default=3)
    a = ap.parse_args()
    bz = GpioBuzzer(a.pin, a.ms, a.mode, a.active_low)
    print(f"pin {a.pin} ({a.mode}, active_low={a.active_low})")
    try:
        for i in range(a.repeat):
            print(f"[{i + 1}/{a.repeat}] A-02 (1 pulsa)")
            bz.buzz("A-02", "D05", True)
            time.sleep(1.5)
            print(f"[{i + 1}/{a.repeat}] A-03 (2 pulsa)")
            bz.buzz("A-03", "D02", True)
            time.sleep(1.5)
    finally:
        time.sleep(0.6)
        bz.close()


if __name__ == "__main__":
    main()
