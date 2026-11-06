#!/usr/bin/env python3
"""Download R017 research-only public ESPN/NCAA Parquet inputs, hash and inspect.

No mutation of any production model, original data, results ledger or forecast.
Usage:
  python scripts/cbb_r017_fetch_sources.py --out /tmp/cbb --refresh
  python scripts/cbb_r013_possession_comparison.py
  python scripts/cbb_r013_continuity_research.py
The two research experiments expect files named below in /tmp/cbb (edit their
source directory variables when using another output directory).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import urllib.request
from datetime import datetime, timezone
from pathlib import Path

import pyarrow.parquet as pq

SOURCES = {}
for y in (2024, 2025, 2026):
    SOURCES[f"ep{y}"] = ("hoopR-mbb-data", f"mbb/player_box/parquet/player_box_{y}.parquet")
    SOURCES[f"eb{y}"] = ("hoopR-mbb-data", f"mbb/team_box/parquet/team_box_{y}.parquet")
for y in (2025, 2026):
    SOURCES[f"er{y}"] = ("hoopR-mbb-data", f"mbb/rosters/parquet/rosters_{y}.parquet")
SOURCES.update({
    "ncaa_poss27": ("ncaa-mbb-hoops-data", "mbb/possessions/parquet/ncaa_mbb_possessions_2027.parquet"),
    "ncaa_team27": ("ncaa-mbb-hoops-data", "mbb/team_box/parquet/ncaa_mbb_team_box_2027.parquet"),
    "ncaa_player27": ("ncaa-mbb-hoops-data", "mbb/player_box/parquet/ncaa_mbb_player_box_2027.parquet"),
    "espn_team27": ("hoopR-mbb-data", "mbb/team_box/parquet/team_box_2027.parquet"),
})

def run(out: Path, refresh: bool = False) -> list[dict]:
    out.mkdir(parents=True, exist_ok=True)
    records = []
    for name, (repo, path) in SOURCES.items():
        url = f"https://raw.githubusercontent.com/sportsdataverse/{repo}/main/{path}"
        target = out / f"{name}.parquet"
        if refresh or not target.exists():
            with urllib.request.urlopen(url, timeout=50) as response:
                data = response.read()
            if len(data) < 16 or data[:4] != b"PAR1" or data[-4:] != b"PAR1":
                raise ValueError(f"Not a Parquet file: {name}")
            target.write_bytes(data)
        blob = target.read_bytes()
        if blob[:4] != b"PAR1" or blob[-4:] != b"PAR1":
            raise ValueError(f"Corrupt/incomplete local Parquet file: {name}")
        meta = pq.read_metadata(target)
        item = {
            "key": name,
            "path": str(target),
            "source_url": url,
            "retrieved_or_checked_utc": datetime.now(timezone.utc).isoformat(),
            "source_revision": "main_UNPINNED",
            "sha256": hashlib.sha256(blob).hexdigest(),
            "size_bytes": len(blob),
            "footer_rows": meta.num_rows,
            "columns": meta.num_columns,
            "record_validation": "REQUIRES_EXPERIMENT_QA",
            "historical_as_of_certified": False,
        }
        records.append(item)
        print(name, meta.num_rows, item["sha256"], flush=True)
    manifest = out / "r017_source_manifest.json"
    manifest.write_text(json.dumps(records, indent=2), encoding="utf-8")
    return records

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=Path("/tmp/cbb"))
    parser.add_argument("--refresh", action="store_true")
    args = parser.parse_args()
    run(args.out, args.refresh)
