"""Keluaran alert kabin. Interface sama untuk beep laptop (Fase 1) dan buzzer GPIO (Fase 2)."""
from __future__ import annotations

import sys
import threading
import time


class Alerter:
    def buzz(self, level: str, code: str, first: bool) -> None:
        raise NotImplementedError

    def close(self) -> None:
        pass


class NullAlerter(Alerter):
    def __init__(self):
        self.calls: list[tuple] = []

    def buzz(self, level, code, first):
        self.calls.append((time.monotonic(), level, code, first))


class LaptopBeeper(Alerter):
    """Beep non-blocking. A-02: dipanggil sekali; A-03: decision engine memanggil berulang tiap 1 s."""

    def __init__(self, freq_hz: int = 2000, ms: int = 250):
        self.freq = int(freq_hz)
        self.ms = int(ms)

    def _beep(self, n: int):
        try:
            if sys.platform == "win32":
                import winsound
                for _ in range(n):
                    winsound.Beep(self.freq, self.ms)
            else:
                for _ in range(n):
                    sys.stdout.write("\a")
                    sys.stdout.flush()
                    time.sleep(self.ms / 1000)
        except Exception:
            pass

    def buzz(self, level, code, first):
        n = 2 if (level == "A-03" and first) else 1
        threading.Thread(target=self._beep, args=(n,), daemon=True).start()


class GpioBuzzer(Alerter):
    """Fase 2 (Jetson): buzzer di pin GPIO lewat Jetson.GPIO."""

    def __init__(self, pin: int, ms: int = 250):
        import Jetson.GPIO as GPIO  # type: ignore

        self.GPIO = GPIO
        self.pin = int(pin)
        self.ms = int(ms)
        GPIO.setmode(GPIO.BCM)
        GPIO.setup(self.pin, GPIO.OUT, initial=GPIO.LOW)

    def _pulse(self):
        self.GPIO.output(self.pin, self.GPIO.HIGH)
        time.sleep(self.ms / 1000)
        self.GPIO.output(self.pin, self.GPIO.LOW)

    def buzz(self, level, code, first):
        threading.Thread(target=self._pulse, daemon=True).start()

    def close(self):
        self.GPIO.cleanup(self.pin)


def make_alerter(cfg: dict) -> Alerter:
    t = cfg.get("type", "laptop_beep")
    if t == "laptop_beep":
        return LaptopBeeper(cfg.get("beep_freq_hz", 2000), cfg.get("beep_ms", 250))
    if t == "gpio":
        return GpioBuzzer(cfg.get("gpio_pin", 18), cfg.get("beep_ms", 250))
    return NullAlerter()
