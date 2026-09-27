"""Acquire and inventory the public SportsDataverse historical repository."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
from pathlib import Path

from cbb.governance.sealed import assert_unsealed

SOURCE_REPOSITORY = "https://github.com/sportsdataverse/ncaa-mbb-hoops-data.git"
FAMILY_TOKENS = {
    "team_games": ("team", "box"),
    "player_games": ("player", "box"),
    "rosters": ("roster",),
    "schedules": ("schedule",),
}
TABULAR_SUFFIXES = {".csv", ".parquet", ".json", ".gz", ".zip"}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def classify(path: Path) -> str | None:
    lowered = path.as_posix().lower()
    for family, tokens in FAMILY_TOKENS.items():
        if all(token in lowered for token in tokens):
            return family
    return None


def inventory(source: Path, seasons: list[int]) -> dict:
    assert_unsealed(seasons)
    files: dict[str, list[dict]] = {family: [] for family in FAMILY_TOKENS}
    for path in sorted(source.rglob("*")):
        if not path.is_file() or path.suffix.lower() not in TABULAR_SUFFIXES:
            continue
        relative = path.relative_to(source)
        text = relative.as_posix()
        if "2026" in text:
            continue
        if not any(str(season) in text for season in seasons):
            continue
        family = classify(relative)
        if family:
            files[family].append(
                {"path": text, "bytes": path.stat().st_size, "sha256": sha256(path)}
            )
    missing = sorted(family for family, entries in files.items() if not entries)
    return {"seasons": seasons, "source": SOURCE_REPOSITORY, "files": files, "missing": missing}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--seasons", nargs="+", type=int, default=[2021, 2022, 2023, 2024, 2025])
    parser.add_argument("--output", type=Path, default=Path("data/canonical"))
    parser.add_argument("--source-dir", type=Path, default=Path("data/cache/ncaa-mbb-hoops-data"))
    args = parser.parse_args()
    assert_unsealed(args.seasons)
    if not args.source_dir.exists():
        args.source_dir.parent.mkdir(parents=True, exist_ok=True)
        subprocess.run(
            ["git", "clone", "--depth", "1", SOURCE_REPOSITORY, str(args.source_dir)],
            check=True,
        )
    report = inventory(args.source_dir, args.seasons)
    args.output.mkdir(parents=True, exist_ok=True)
    report_path = args.output / "source_inventory.json"
    report_path.write_text(json.dumps(report, indent=2, sort_keys=True) + "\n")
    counts = {family: len(entries) for family, entries in report["files"].items()}
    print(json.dumps({"inventory": str(report_path), "counts": counts}, sort_keys=True))
    if report["missing"]:
        raise RuntimeError(f"required source families not discovered: {report['missing']}")


if __name__ == "__main__":
    main()
