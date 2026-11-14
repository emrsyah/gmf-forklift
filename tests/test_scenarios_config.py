"""configs/scenarios.yaml harus terbaca dengan kunci string 'no' (YAML 1.1: `no:` tanpa kutip = False)."""
from src.config import load_yaml


def test_scenarios_have_string_no_key():
    s = load_yaml("configs/scenarios.yaml")["scenarios"]
    assert [x["no"] for x in s] == list(range(1, 21))
    assert all({"code", "level", "duration_s", "desc"} <= set(x) for x in s)
