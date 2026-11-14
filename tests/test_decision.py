"""Unit test modul keputusan: prioritas, cooldown, A-03 berulang, hysteresis, start/end."""
from src.decision import DecisionEngine
from tests.synth import events_cfg

EV = events_cfg()


def _run(seq, ev=EV, fps=30):
    """seq: list of (dur_s, {code: conf}) -> (events, buzzes)"""
    d = DecisionEngine(ev)
    t = 1_000_000.0
    events, buzz = [], []
    for dur, act in seq:
        for _ in range(int(round(dur * fps))):
            a = {c: {"confidence": v, "condition_start_ms": int(t)} for c, v in act.items()}
            for x in d.step(int(t), a):
                (events if x.kind == "event" else buzz).append((int(t), x.payload))
            t += 1000 / fps
    return events, buzz


def test_priority_picks_highest():
    ev, _ = _run([(1, {"D05": 0.9, "D02": 0.9, "D10": 0.9})])
    assert ev[0][1]["code"] == "D02" and ev[0][1]["level"] == "A-03"


def test_levels_match_logbook():
    expect = {"D01": "A-02", "D05": "A-02", "D06": "A-02", "D10": "A-02",
              "D02": "A-03", "D04": "A-03", "D09": "A-03"}
    for code, lvl in expect.items():
        ev, _ = _run([(0.1, {code: 1.0})])
        assert ev[0][1]["level"] == lvl, code


def test_a02_single_buzz_and_cooldown():
    ev, buzz = _run([(2, {"D05": 1}), (2, {}), (2, {"D05": 1}), (2, {})])
    starts = [e for _, e in ev if e["phase"] == "start"]
    assert len(starts) == 1           # dalam 20 s tidak dikirim ulang
    assert len(buzz) == 1


def test_a02_after_cooldown_resent():
    ev, _ = _run([(2, {"D05": 1}), (21, {}), (1, {"D05": 1})])
    assert len([e for _, e in ev if e["phase"] == "start"]) == 2


def test_a03_repeating_buzzer_until_condition_ends():
    _, buzz = _run([(5, {"D09": 1}), (3, {})])
    assert 5 <= len(buzz) <= 6        # 1x per detik selama 5 s
    assert all(b["level"] == "A-03" for _, b in buzz)


def test_end_after_hysteresis():
    ev, _ = _run([(3, {"D02": 1}), (1.4, {}), (0.2, {})])
    phases = [e["phase"] for _, e in ev]
    assert phases == ["start", "end"]
    t_start, t_end = ev[0][0], ev[1][0]
    assert 4.4 <= (t_end - t_start) / 1000 <= 4.6


def test_brief_dropout_keeps_one_event():
    ev, _ = _run([(1, {"D05": 1}), (0.5, {}), (1, {"D05": 1}), (2, {})])
    assert [e["phase"] for _, e in ev] == ["start", "end"]


def test_higher_priority_preempts():
    ev, _ = _run([(2, {"D05": 1}), (2, {"D05": 1, "D02": 1}), (2, {})])
    seq = [(e["phase"], e["code"]) for _, e in ev]
    assert seq[:3] == [("start", "D05"), ("end", "D05"), ("start", "D02")]


def test_lower_priority_waits_during_hysteresis():
    # D-02 hilang, D-05 muncul -> D-05 tidak boleh dibuka sampai D-02 benar-benar ditutup
    ev, _ = _run([(2, {"D02": 1}), (1.0, {"D05": 1})])
    assert [(e["phase"], e["code"]) for _, e in ev] == [("start", "D02")]


def test_event_schema_fields():
    ev, _ = _run([(1, {"D09": 0.8})])
    e = ev[0][1]
    for k in ("event_id", "device_id", "phase", "code", "level", "ts_event_ms",
              "ts_condition_start_ms", "confidence", "snapshot_path", "clip_path", "model_versions"):
        assert k in e


def test_no_escalation_two_a02_stay_a02():
    ev, _ = _run([(5, {"D05": 1, "D10": 1, "D01": 1})])
    assert all(e["level"] == "A-02" for _, e in ev)
