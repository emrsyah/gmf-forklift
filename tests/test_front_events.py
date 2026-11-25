"""Event kamera depan: kandidat harus bertahan event_min_active_s; start/end lewat DecisionEngine terpisah."""
from src.front.events import FrontEventGate
from src.front.pipeline import FrontResult
from src.front.zones import FrontCandidate

EVENTS = {"priority": [], "levels": {"F01": "A-02", "F02": "A-03", "F03": "A-03"},
          "A02_cooldown_s": 20, "end_hysteresis_s": 1.5}


def gate():
    return FrontEventGate(EVENTS, ["F02", "F03", "F01"], 0.5, "jetson-dev", "jetson-front", {"front": "front_v2"})


T0 = 1_795_000_000_000


def res(ts, codes):
    cands = [FrontCandidate(c, EVENTS["levels"][c], 1, "person", 2.5, 0.8) for c in codes]
    return FrontResult(T0 + ts, ts // 100, [], cands, list(codes))


def events(acts):
    return [a.payload for a in acts if a.kind == "event"]


def test_flicker_below_min_active_no_event():
    g = gate()
    out = []
    for ts in (0, 100, 200, 300):
        out += events(g.step(res(ts, ["F02"])))
    out += events(g.step(res(400, [])))
    assert out == []


def test_person_in_danger_zone_start_then_end():
    g = gate()
    out = []
    for ts in range(0, 1100, 100):
        out += events(g.step(res(ts, ["F02"])))
    assert [(e["phase"], e["code"], e["level"]) for e in out] == [("start", "F02", "A-03")]
    assert out[0]["event_id"].startswith("jetson-front-")
    assert out[0]["distance_m"] == 2.5 and out[0]["camera"] == "front"
    assert out[0]["ts_condition_start_ms"] == T0
    for ts in range(1100, 3000, 100):
        out += events(g.step(res(ts, [])))
    assert [e["phase"] for e in out] == ["start", "end"]


def test_higher_priority_code_wins():
    g = gate()
    out = []
    for ts in range(0, 700, 100):
        out += events(g.step(res(ts, ["F01", "F02"])))
    assert [e["code"] for e in out] == ["F02"]
