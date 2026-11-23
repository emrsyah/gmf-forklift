"""Buzzer GPIO (Jetson.GPIO palsu): pola pulsa, active-low, fallback, dan ts_buzzer_ms hanya dengan buzzer nyata."""
import time

from src.decision import Action
from src.io.alert import GpioBuzzer, NullAlerter, make_alerter
from src.io.sinks import MemorySink
from src.main import handle_actions


class FakeGPIO:
    BOARD, BCM, OUT, HIGH, LOW = "BOARD", "BCM", "OUT", 1, 0

    def __init__(self):
        self.log = []

    def setwarnings(self, v):
        pass

    def setmode(self, m):
        self.mode = m

    def setup(self, pin, d, initial):
        self.log.append(("init", initial))

    def output(self, pin, v):
        self.log.append(v)

    def cleanup(self, pin):
        self.log.append("cleanup")


def _wait(bz):
    time.sleep(0.05)
    with bz._lock:
        pass


def test_a03_first_two_pulses_a02_one():
    g = FakeGPIO()
    bz = GpioBuzzer(12, ms=1, gap_ms=1, gpio=g)
    bz.buzz("A-03", "D02", True)
    _wait(bz)
    assert g.log[1:] == [1, 0, 1, 0]
    g.log.clear()
    bz.buzz("A-02", "D05", True)
    _wait(bz)
    assert g.log == [1, 0]
    assert g.mode == "BOARD"


def test_active_low_inverts():
    g = FakeGPIO()
    bz = GpioBuzzer(12, ms=1, active_low=True, gpio=g)
    assert g.log[0] == ("init", 1)            # mati = HIGH
    bz.buzz("A-02", "D10", True)
    _wait(bz)
    assert g.log[1:] == [0, 1]


def test_gpio_failure_falls_back_to_null():
    al = make_alerter({"type": "gpio"})        # Jetson.GPIO tidak ada di laptop
    assert isinstance(al, NullAlerter) and al.real is False


def test_no_buzzer_no_ts_buzzer_ms():
    ev = {"event_id": "e", "phase": "start", "code": "D02", "level": "A-03", "ts_event_ms": 1}
    acts = [Action("buzz", {"level": "A-03", "code": "D02", "first": True}), Action("event", ev)]
    sent = handle_actions(acts, NullAlerter(), MemorySink(), None, None)
    assert "ts_buzzer_ms" not in sent[0]
