import json
from pathlib import Path


def test_frozen_targets_are_documented_not_claimed_as_reproduced():
    record = json.loads(Path("registry/frozen/r1_r2.json").read_text())
    assert record["documented_targets"]["differential_mae"] == 15.373316
    assert record["reproduction_status"].startswith("BLOCKED_")

