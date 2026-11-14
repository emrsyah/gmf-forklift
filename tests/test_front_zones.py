"""Uji kandidat F01/F02/F03 (kode usulan) dengan track sintetis."""
from dataclasses import dataclass

import pytest

from src.config import CONFIG_DIR, load_yaml
from src.front.zones import ZoneEvaluator


@dataclass
class T:
    track_id: int
    cls: str
    distance_m: float | None
    approaching: bool = False
    conf: float = 0.9


@pytest.fixture
def ze():
    thr = load_yaml(CONFIG_DIR / "thresholds.yaml")["front"]
    cam = load_yaml(CONFIG_DIR / "front_camera.yaml")
    ev = load_yaml(CONFIG_DIR / "events.yaml")
    return ZoneEvaluator.from_config(thr, cam, ev, "person")


def test_person_in_warning_zone_approaching_is_F01(ze):
    d = (ze.warning_zone_m + ze.danger_zone_m) / 2
    c = ze.evaluate([T(1, "person", d, approaching=True)])
    assert [x.code for x in c] == ["F01"] and c[0].level == "A-02"


def test_person_in_warning_zone_not_approaching_is_nothing(ze):
    d = (ze.warning_zone_m + ze.danger_zone_m) / 2
    assert ze.evaluate([T(1, "person", d, approaching=False)]) == []


def test_person_outside_warning_zone_is_nothing(ze):
    assert ze.evaluate([T(1, "person", ze.warning_zone_m + 1, approaching=True)]) == []


def test_person_in_danger_zone_is_F02_even_if_approaching(ze):
    c = ze.evaluate([T(1, "person", ze.danger_zone_m - 0.5, approaching=True)])
    assert [x.code for x in c] == ["F02"] and c[0].level == "A-03"


def test_aircraft_or_gse_in_danger_zone_is_F03(ze):
    c = ze.evaluate([T(1, "aircraft_part", ze.danger_zone_m - 0.5), T(2, "gse_vehicle", 1.0)])
    assert {x.code for x in c} == {"F03"} and all(x.level == "A-03" for x in c)


def test_other_obstacle_in_danger_zone_is_nothing(ze):
    assert ze.evaluate([T(1, "pallet_jack", 0.5)]) == []


def test_unknown_distance_is_ignored(ze):
    assert ze.evaluate([T(1, "person", None, approaching=True)]) == []


def test_priority_order(ze):
    d_warn = (ze.warning_zone_m + ze.danger_zone_m) / 2
    c = ze.evaluate([T(1, "person", d_warn, True), T(2, "gse_vehicle", 1.0), T(3, "person", 1.0)])
    assert ze.active_codes(c) == ["F02", "F03", "F01"]
