import json
from pathlib import Path

import pytest

from cbb.data.build_historical import inventory
from cbb.governance.sealed import SealedHoldoutAccessError


def test_inventory_classifies_required_families_and_excludes_2026(tmp_path: Path):
    paths = [
        "team_box/2025_team_box.csv",
        "player_box/2025_player_box.csv",
        "rosters/2025_roster.json",
        "schedules/2025_schedule.csv",
        "team_box/2026_team_box.csv",
    ]
    for name in paths:
        path = tmp_path / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps({"fixture": name}))

    report = inventory(tmp_path, [2025])

    assert report["missing"] == []
    assert all("2026" not in item["path"] for items in report["files"].values() for item in items)
    assert all(len(items) == 1 for items in report["files"].values())


def test_inventory_rejects_sealed_season(tmp_path: Path):
    with pytest.raises(SealedHoldoutAccessError, match="SEALED_HOLDOUT_ACCESS_ERROR"):
        inventory(tmp_path, [2026])
