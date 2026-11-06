#!/usr/bin/env python3
"""CBB R010 research-only NCAA/ESPN possession, player, and roster source QA.

Governance: CBB-OBJ-003 / CBB-PROG-007 / CBB-Q-014; CBB-Q-012.
Does not predict games, alter production models, or amend frozen ledgers.

Usage:
 python cbb_r010_source_qa.py --source-dir data/raw --output-dir data/r010

Required Parquet filenames in source-dir:
 ncaa_mbb_possessions_2027.parquet
 ncaa_mbb_player_box_2027.parquet
 game_rosters_2027.parquet
 rosters_2027.parquet

Optional historical source pair:
 player_box_2026.parquet (ESPN player box, previous season)
 rosters_2026.parquet (ESPN previous-season roster)

This script never interprets a game-roster's postgame active flag as pregame availability.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import urllib.request
import urllib.error
from pathlib import Path
from typing import Any

import pandas as pd

SOURCES = {
    "ncaa_possessions": "ncaa_mbb_possessions_2027.parquet",
    "ncaa_player_box": "ncaa_mbb_player_box_2027.parquet",
    "espn_game_rosters": "game_rosters_2027.parquet",
    "espn_season_rosters": "rosters_2027.parquet",
}
OPTIONAL_TEAM_BOX = "ncaa_mbb_team_box_2027.parquet"
OPTIONAL_TEAM_BOX_URL = "https://raw.githubusercontent.com/sportsdataverse/ncaa-mbb-hoops-data/main/mbb/team_box/parquet/ncaa_mbb_team_box_2027.parquet"
SOURCE_URLS = {
    "ncaa_possessions": "https://raw.githubusercontent.com/sportsdataverse/ncaa-mbb-hoops-data/main/mbb/possessions/parquet/ncaa_mbb_possessions_2027.parquet",
    "ncaa_player_box": "https://raw.githubusercontent.com/sportsdataverse/ncaa-mbb-hoops-data/main/mbb/player_box/parquet/ncaa_mbb_player_box_2027.parquet",
    "espn_game_rosters": "https://raw.githubusercontent.com/sportsdataverse/hoopR-mbb-data/main/mbb/game_rosters/parquet/game_rosters_2027.parquet",
    "espn_season_rosters": "https://raw.githubusercontent.com/sportsdataverse/hoopR-mbb-data/main/mbb/rosters/parquet/rosters_2027.parquet",
}
REQUIRED = {
    "ncaa_possessions": {"contest_id", "espn_game_id", "period", "poss_team_espn_team_id", "poss_num", "pts"},
    "ncaa_player_box": {"contest_id", "espn_game_id", "team_espn_team_id", "player_id", "clean_name", "mins", "pts"},
    "espn_game_rosters": {"game_id", "team_id", "athlete_id", "athlete_display_name", "active", "did_not_play"},
    "espn_season_rosters": {"season", "team_id", "athlete_id"},
}


def identifier(v: Any) -> str | None:
    """Normalize a source-native identifier without equating namespaces."""
    if v is None or pd.isna(v):
        return None
    s = str(v).strip()
    if s.lower() in {"", "none", "nan", "null", "<na>"}:
        return None
    if re.fullmatch(r"[0-9]+\.0", s):
        return s[:-2]
    return s


def clean_name(v: Any) -> str:
    if v is None or pd.isna(v):
        return ""
    return " ".join(re.sub(r"[^a-z0-9]+", " ", str(v).casefold()).split())


def truthy(v: Any) -> bool | None:
    if v is None or pd.isna(v):
        return None
    if isinstance(v, bool):
        return v
    x = str(v).lower().strip()
    if x in {"true", "t", "1", "yes"}:
        return True
    if x in {"false", "f", "0", "no"}:
        return False
    return None


def key_frame(df: pd.DataFrame, mapping: dict[str, str]) -> pd.DataFrame:
    out = df.copy()
    for key, old in mapping.items():
        out[key] = df[old].map(identifier)
    return out


def validate_schema(source: str, df: pd.DataFrame) -> list[str]:
    missing = sorted(REQUIRED[source] - set(df.columns))
    return [f"{source} missing required column: {x}" for x in missing]


def source_sha(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for data in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(data)
    return h.hexdigest()


def safe_fetch(url: str, target: Path, max_bytes: int = 50_000_000) -> dict:
    """Download an explicitly approved GitHub raw file and validate PAR1 markers.

    No runtime market data is consulted. Existing local files are never overwritten.
    """
    if target.exists():
        return {"status": "ALREADY_PRESENT", "path": str(target),
                "size": target.stat().st_size, "sha256": source_sha(target)}
    target.parent.mkdir(parents=True, exist_ok=True)
    temp = target.with_name(target.name + ".partial")
    try:
        with urllib.request.urlopen(urllib.request.Request(url, headers={"User-Agent": "CBB-R010-Source-QA/1.0"}), timeout=35) as src:
            total = 0
            with temp.open("wb") as dst:
                while True:
                    chunk = src.read(1024 * 1024)
                    if not chunk:
                        break
                    total += len(chunk)
                    if total > max_bytes:
                        raise ValueError("source exceeds configured size limit")
                    dst.write(chunk)
        with temp.open("rb") as f:
            prefix = f.read(4)
            f.seek(-4, 2)
            suffix = f.read(4)
        if prefix != b"PAR1" or suffix != b"PAR1":
            raise ValueError("download is not a complete Parquet file")
        temp.replace(target)
        return {"status": "DOWNLOADED", "path": str(target), "size": total,
                "sha256": source_sha(target)}
    finally:
        if temp.exists():
            temp.unlink()


def qa_sources(frames: dict[str, pd.DataFrame]) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    """Audit sources and build conservative join evidence.

    Returns (source_summary, exceptions, game_possessions, player_matches).
    No writes occur inside this function.
    """
    issues: list[dict] = []
    summary: list[dict] = []
    for source, df in frames.items():
        failures = validate_schema(source, df) if source in REQUIRED else []
        summary.append({"dataset": source, "rows": len(df), "columns": len(df.columns),
                        "schema_pass": not failures, "unique_rows": len(df.drop_duplicates()),
                        "status": "SCHEMA_PASS_ROWS_UNAUDITED" if not failures else "SCHEMA_FAIL"})
        for issue in failures:
            issues.append({"source": source, "kind": "SCHEMA", "key": "", "detail": issue})
    if any(not x["schema_pass"] for x in summary) or not set(["ncaa_possessions", "ncaa_player_box", "espn_game_rosters"]).issubset(frames):
        return pd.DataFrame(summary), pd.DataFrame(issues), pd.DataFrame(), pd.DataFrame()

    poss = key_frame(frames["ncaa_possessions"], {
        "g": "espn_game_id", "contest": "contest_id", "team": "poss_team_espn_team_id"})
    poss["period_number"] = pd.to_numeric(poss["period"], errors="coerce")
    poss["pts_num"] = pd.to_numeric(poss["pts"], errors="coerce")
    invalid_poss = poss[poss["g"].isna() | poss["team"].isna() | poss["period_number"].isna()]
    for _, r in invalid_poss.head(150).iterrows():
        issues.append({"source": "ncaa_possessions", "kind": "MISSING_EVENT_ID",
                       "key": r["contest"], "detail": "game, team, or period missing"})
    clean = poss[~poss.index.isin(invalid_poss.index)]
    # Exact duplicated event identifiers can inflate event counts; quarantine them.
    event_key = ["g", "contest", "team", "period_number", "poss_num"]
    duplicates = clean[clean.duplicated(event_key, keep=False)]
    for _, r in duplicates.head(200).iterrows():
        issues.append({"source": "ncaa_possessions", "kind": "DUPLICATE_POSS_NUM",
                       "key": r["contest"], "detail": f"period={r['period_number']} team={r['team']}"})
    game_possessions = clean.groupby(["g", "contest", "team", "period_number"], dropna=False).agg(
        event_possessions=("poss_num", "size"), event_points=("pts_num", "sum")
    ).reset_index()
    # Explicitly do not presume an NCAA 'possession' event count exactly matches scoring possessions.
    game_possessions["definition"] = "NCAA_SOURCE_POSSESSION_ROWS_NOT_OFFICIAL_GROUND_TRUTH"
    game_possessions["asof_status"] = "POSTGAME_DIAGNOSTIC"

    player = key_frame(frames["ncaa_player_box"], {
        "g": "espn_game_id", "contest": "contest_id", "team": "team_espn_team_id", "ncaa_player": "player_id"})
    player["name_key"] = player["clean_name"].map(clean_name)
    roster = key_frame(frames["espn_game_rosters"], {
        "g": "game_id", "team": "team_id", "espn_player": "athlete_id"})
    roster["name_key"] = roster["athlete_display_name"].map(clean_name)
    roster["active_bool"] = roster["active"].map(truthy)
    roster["dnp_bool"] = roster["did_not_play"].map(truthy)

    for df, source, cols in [
        (player, "ncaa_player_box", ["g", "team", "ncaa_player"]),
        (roster, "espn_game_rosters", ["g", "team", "espn_player"]),
    ]:
        bad = df[df[cols].isna().any(axis=1)]
        for _, r in bad.head(150).iterrows():
            issues.append({"source": source, "kind": "MISSING_ENTITY_ID",
                           "key": r["g"], "detail": "Missing game/team/player source ID"})
        dup = df[df.duplicated(cols, keep=False)]
        for _, r in dup.head(150).iterrows():
            issues.append({"source": source, "kind": "DUPLICATE_NATIVE_PLAYER",
                           "key": r["g"], "detail": f"duplicate {cols}"})

    for _, r in roster[roster["active_bool"].eq(True) & roster["dnp_bool"].eq(True)].head(150).iterrows():
        issues.append({"source": "espn_game_rosters", "kind": "ACTIVE_DNP_CONFLICT",
                       "key": r["g"], "detail": f"athlete {r['espn_player']} active and DNP"})

    # Identity match: exact same ESPN game, ESPN team, and strictly unique normalized full name.
    # DO NOT assume NCAA 'player_id' is identical to ESPN 'athlete_id'.
    usable = roster.dropna(subset=["g", "team", "espn_player"])
    unique_name = usable.groupby(["g", "team", "name_key"])["espn_player"].nunique()
    unambig = unique_name[unique_name == 1].index
    candidates = usable.set_index(["g", "team", "name_key"])
    matches = []
    for _, p in player.iterrows():
        key = (p["g"], p["team"], p["name_key"])
        status = "UNMATCHED"
        athlete = None
        if p["g"] is None or p["team"] is None or not p["name_key"]:
            status = "MISSING_JOIN_KEY"
        elif key in unambig:
            lookup = candidates.loc[key]
            if isinstance(lookup, pd.DataFrame):
                status = "AMBIGUOUS_DUPLICATE_ROW"
            else:
                athlete, status = lookup["espn_player"], "MATCH_EXACT_GAME_TEAM_NAME"
        elif key in unique_name.index:
            status = "AMBIGUOUS_SAME_NAME"
        matches.append({"espn_game_id": p["g"], "ncaa_contest_id": p["contest"],
                        "team_espn_id": p["team"], "ncaa_player_id": p["ncaa_player"],
                        "espn_athlete_id": athlete, "name_key": p["name_key"],
                        "match_status": status, "feature_availability": "POSTGAME_DIAGNOSTIC"})
        if status != "MATCH_EXACT_GAME_TEAM_NAME":
            issues.append({"source": "crosswalk", "kind": status, "key": p["g"],
                           "detail": f"team={p['team']}; name={p['name_key']}"})

    season_roster = (key_frame(frames["espn_season_rosters"], {
        "team": "team_id", "espn_player": "athlete_id"}) if "espn_season_rosters" in frames else pd.DataFrame())
    if "season" in season_roster.columns:
        season_roster["year"] = season_roster["season"].map(identifier)
        duplicate = season_roster[season_roster.duplicated(["year", "team", "espn_player"], keep=False)]
        for _, r in duplicate.head(150).iterrows():
            issues.append({"source": "espn_season_rosters", "kind": "DUPLICATE_SEASON_PLAYER",
                           "key": r["espn_player"], "detail": f"team={r['team']} year={r['year']}"})

    return (pd.DataFrame(summary), pd.DataFrame(issues), game_possessions,
            pd.DataFrame(matches))


def returning_minutes(prior_box: pd.DataFrame, current_roster: pd.DataFrame) -> pd.DataFrame:
    """Compute preseason prior-year returning-minutes share by stable ESPN athlete ID.

    Prior box expected columns team_id, athlete_id, minutes (ESPN names may differ, so
    caller must rename explicitly and verify). Current roster must represent membership
    known by the forecast cutoff, not a retroactive season-end snapshot.
    """
    required_prior = {"team_id", "athlete_id", "minutes"}
    required_cur = {"team_id", "athlete_id"}
    if not required_prior.issubset(prior_box) or not required_cur.issubset(current_roster):
        raise ValueError("roster continuity requires explicit verified ID and minutes columns")
    prior = key_frame(prior_box, {"team": "team_id", "athlete": "athlete_id"})
    current = key_frame(current_roster, {"team": "team_id", "athlete": "athlete_id"})
    prior["minutes"] = pd.to_numeric(prior["minutes"], errors="coerce")
    if prior["minutes"].isna().any() or (prior["minutes"] < 0).any():
        raise ValueError("invalid previous-season minutes; no inferred zeros allowed")
    if current[["team", "athlete"]].isna().any().any():
        raise ValueError("current roster has unresolved team/player IDs")
    current_keys = set(zip(current.team, current.athlete))
    prior["returned"] = [(t, a) in current_keys for t, a in zip(prior.team, prior.athlete)]
    prior["retained_minutes"] = prior.minutes * prior.returned.astype(int)
    g = prior.groupby("team", as_index=False).agg(prior_minutes=("minutes", "sum"),
                   retained_minutes=("retained_minutes", "sum"))
    g["returning_minutes_share"] = g.retained_minutes / g.prior_minutes.where(g.prior_minutes > 0)
    g["data_status"] = "SOURCE_PIT_REVIEW_REQUIRED"
    return g


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--report-json", action="store_true")
    parser.add_argument("--fetch", action="store_true", help="download only missing 2027 Parquet assets")
    parser.add_argument("--partial", action="store_true", help="allow three-way QA if season roster is missing")
    parser.add_argument("--team-box", action="store_true", help="also fetch/validate optional NCAA team box")
    args = parser.parse_args()
    out = args.output_dir
    out.mkdir(parents=True, exist_ok=True)
    frames, manifest = {}, []
    if args.fetch:
        for name, filename in SOURCES.items():
            path = args.source_dir / filename
            try:
                download = safe_fetch(SOURCE_URLS[name], path)
                print(json.dumps({"dataset": name, **download}))
            except (OSError, ValueError, urllib.error.URLError) as error:
                print(json.dumps({"dataset": name, "download_status": "BLOCKED",
                                  "detail": str(error)}))
    for name, filename in SOURCES.items():
        path = args.source_dir / filename
        if not path.exists():
            manifest.append({"dataset": name, "file": filename, "status": "MISSING_FILE",
                             "bytes": 0, "sha256": ""})
            continue
        df = pd.read_parquet(path, engine="pyarrow")
        frames[name] = df
        manifest.append({"dataset": name, "file": filename, "status": "PARSED",
                         "bytes": path.stat().st_size, "sha256": source_sha(path),
                         "rows": len(df), "columns": len(df.columns)})
    if args.team_box:
        tpath = args.source_dir / OPTIONAL_TEAM_BOX
        if args.fetch:
            try:
                print(json.dumps({"dataset": "ncaa_team_box", **safe_fetch(OPTIONAL_TEAM_BOX_URL, tpath)}))
            except (OSError, ValueError, urllib.error.URLError) as error:
                print(json.dumps({"dataset": "ncaa_team_box", "download_status": "BLOCKED", "detail": str(error)}))
        if tpath.exists():
            tb = pd.read_parquet(tpath, engine="pyarrow")
            frames["ncaa_team_box"] = tb
            manifest.append({"dataset": "ncaa_team_box", "file": OPTIONAL_TEAM_BOX,
                             "status": "PARSED_OPTIONAL", "bytes": tpath.stat().st_size,
                             "sha256": source_sha(tpath), "rows": len(tb), "columns": len(tb.columns)})
    pd.DataFrame(manifest).to_csv(out / "source_manifest.csv", index=False)
    required_run = set(SOURCES) if not args.partial else {"ncaa_possessions", "ncaa_player_box", "espn_game_rosters"}
    if not required_run.issubset(frames):
        (out / "gate.json").write_text(json.dumps({"status": "BLOCKED_MISSING_SOURCE",
                     "datasets_missing": sorted(required_run - set(frames))}, indent=2))
        raise SystemExit(2)

    sources, exceptions, game_possessions, crosswalk = qa_sources(frames)
    sources.to_csv(out / "source_qa.csv", index=False)
    exceptions.to_csv(out / "qa_exceptions.csv", index=False)
    game_possessions.to_csv(out / "game_possessions.csv", index=False)
    crosswalk.to_csv(out / "player_id_crosswalk.csv", index=False)
    matches = int(crosswalk.match_status.eq("MATCH_EXACT_GAME_TEAM_NAME").sum())
    # Game IDs are ESPN-native in all three datasets, but independent crosswalk
    # of true game scores and participant identities is still necessary.
    game_sets = {
        "ncaa_possessions": set(frames["ncaa_possessions"]["espn_game_id"].map(identifier).dropna()),
        "ncaa_player_box": set(frames["ncaa_player_box"]["espn_game_id"].map(identifier).dropna()),
        "espn_game_rosters": set(frames["espn_game_rosters"]["game_id"].map(identifier).dropna()),
    }
    games_matched = len(set.intersection(*game_sets.values()))
    if "ncaa_team_box" in frames:
        tb = frames["ncaa_team_box"]
        required_tb = {"espn_game_id", "team_espn_team_id", "pts"}
        if required_tb.issubset(tb):
            pb = frames["ncaa_player_box"].copy()
            pb["g"] = pb["espn_game_id"].map(identifier)
            pb["team"] = pb["team_espn_team_id"].map(identifier)
            team = tb.copy()
            team["g"] = team["espn_game_id"].map(identifier)
            team["team"] = team["team_espn_team_id"].map(identifier)
            player_sum = pb.groupby(["g", "team"], dropna=False)["pts"].sum().reset_index(name="player_points")
            team["team_points"] = pd.to_numeric(team["pts"], errors="coerce")
            chk = team.merge(player_sum, how="left", on=["g", "team"], validate="one_to_one")
            chk["points_diff"] = chk["player_points"] - chk["team_points"]
            chk.to_csv(out / "team_points_reconciliation.csv", index=False)
            if chk["points_diff"].isna().any() or chk["points_diff"].abs().gt(0.1).any():
                gate_status = "TEAM_SCORE_QA_FAIL"
            else:
                gate_status = "RESEARCH_ONLY_NOT_PRODUCTION"
        else:
            gate_status = "OPTIONAL_TEAM_BOX_SCHEMA_FAIL"
    else:
        gate_status = "RESEARCH_ONLY_NOT_PRODUCTION"
    gate = {"status": gate_status,
            "sources_with_valid_schema": int(sources.schema_pass.sum()),
            "total_sources": len(SOURCES), "exceptions": len(exceptions),
            "ncaa_player_rows": len(crosswalk), "matched_player_game_rows": matches,
            "cross_source_player_game_match_pct": round(matches / len(crosswalk), 5) if len(crosswalk) else None,
            "game_possession_rows": len(game_possessions),
            "distinct_games_by_source": {k: len(v) for k, v in game_sets.items()},
            "matched_three_way_espn_game_ids": games_matched,
            "espn_season_roster_present": "espn_season_rosters" in frames,
            "caution": "No official ground truth/availability certification from parsed data alone."}
    (out / "gate.json").write_text(json.dumps(gate, indent=2))
    if args.report_json:
        print(json.dumps(gate, indent=2))


if __name__ == "__main__":
    main()