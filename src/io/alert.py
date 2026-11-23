"""Keluaran alert kabin. Interface sama untuk beep laptop (Fase 1) dan buzzer GPIO (Fase 2)."""
from __future__ import annotations

import sys
import threading
import time


class Alerter:
    # False = tidak ada buzzer fisik -> event tidak boleh membawa ts_buzzer_ms (angka K5 kabin palsu)
    real: bool = True

    def buzz(self, level: str, code: str, first: bool) -> None:
        raise NotImplementedError

    def close(self) -> None:
        pass


class NullAlerter(Alerter):
    real = False

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
    """Fase 2 (Jetson): buzzer aktif (atau relay) di pin GPIO lewat Jetson.GPIO.
    Pola sama dengan beep laptop: A-03 pertama 2 pulsa, lainnya 1 pulsa (A-03 diulang tiap 1 s oleh decision).
    `gpio_mode`: BOARD (nomor pin header 40-pin, paling aman di Orin) atau BCM. `active_low`: untuk modul relay
    / buzzer yang menyala saat LOW."""

    def __init__(self, pin: int, ms: int = 250, mode: str = "BOARD", active_low: bool = False,
                 gap_ms: int = 120, gpio=None):
        if gpio is None:
            import Jetson.GPIO as gpio  # type: ignore
        self.GPIO = gpio
        self.pin = int(pin)
        self.ms = int(ms)
        self.gap_ms = int(gap_ms)
        self.on = gpio.LOW if active_low else gpio.HIGH
        self.off = gpio.HIGH if active_low else gpio.LOW
        gpio.setwarnings(False)
        gpio.setmode(getattr(gpio, str(mode).upper()))
        gpio.setup(self.pin, gpio.OUT, initial=self.off)
        self._lock = threading.Lock()      # pulsa tidak saling tumpang (A-03 berulang + event baru)

    def _pulse(self, n: int):
        with self._lock:
            for i in range(n):
                self.GPIO.output(self.pin, self.on)
                time.sleep(self.ms / 1000)
                self.GPIO.output(self.pin, self.off)
                if i < n - 1:
                    time.sleep(self.gap_ms / 1000)

    def buzz(self, level, code, first):
        n = 2 if (level == "A-03" and first) else 1
        threading.Thread(target=self._pulse, args=(n,), daemon=True).start()

    def close(self):
        try:
            self.GPIO.output(self.pin, self.off)
        finally:
            self.GPIO.cleanup(self.pin)


def make_alerter(cfg: dict) -> Alerter:
    t = cfg.get("type", "laptop_beep")
    if t == "laptop_beep":
        return LaptopBeeper(cfg.get("beep_freq_hz", 2000), cfg.get("beep_ms", 250))
    if t == "gpio":
        try:
            return GpioBuzzer(cfg.get("gpio_pin", 12), cfg.get("beep_ms", 250), cfg.get("gpio_mode", "BOARD"),
                              bool(cfg.get("active_low", False)), cfg.get("gap_ms", 120))
        except Exception as e:  # noqa: BLE001 - deteksi + event tetap jalan tanpa buzzer
            print(f"[alert] BUZZER GPIO GAGAL ({type(e).__name__}: {e}) -> tanpa buzzer; event tanpa ts_buzzer_ms",
                  flush=True)
    return NullAlerter()
