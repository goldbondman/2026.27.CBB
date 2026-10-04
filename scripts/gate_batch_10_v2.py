from __future__ import annotations

import pandas as pd
import gate_batch_10 as base


def _turnover_column(df: pd.DataFrame) -> str:
    known = ["total_turnovers", "totalTurnovers", "turnovers", "tov", "to", "turnover", "turnovers_total", "total_tov", "tos"]
    for c in known:
        if c in df.columns:
            return c
    candidates = [c for c in df.columns if ("turn" in c.lower() or "tov" in c.lower() or c.lower() == "to") and "team" not in c.lower()]
    if not candidates:
        raise KeyError(f"No turnover-like field found. Columns={list(df.columns)}")
    print("TURNOVER_CANDIDATES", candidates)
    return candidates[0]


def load_team_games() -> tuple[pd.DataFrame, dict]:
    frames = []
    raw_counts = {}
    for sy in base.SEASONS:
        d = pd.read_parquet(base.RAW + f"mbb/team_box/parquet/ncaa_mbb_team_box_{sy}.parquet")
        raw_counts[str(sy)] = int(len(d))
        d["season"] = sy
        frames.append(d)
    g = pd.concat(frames, ignore_index=True)
    sched = pd.read_parquet(base.RAW + "mbb/ncaa_mbb_schedule_master.parquet")

    gid = base.pick(g, ["contest_id", "game_id"])
    team = base.pick(g, ["team", "team_name", "team_display_name"])
    pts = base.pick(g, ["pts", "points"])
    opos = base.pick(g, ["o_poss", "offensive_possessions"])
    dpos = base.pick(g, ["d_poss", "defensive_possessions"])
    fga = base.pick(g, ["fga", "field_goal_attempts"])
    orb = base.pick(g, ["orb", "offensive_rebounds"])
    tov = _turnover_column(g)
    fta = base.pick(g, ["fta", "free_throw_attempts"])
    print("TEAM_BOX_FIELDS", {"gid": gid, "team": team, "pts": pts, "opos": opos, "dpos": dpos, "fga": fga, "orb": orb, "tov": tov, "fta": fta})

    sgid = base.pick(sched, ["contest_id", "game_id"])
    sdate = base.pick(sched, ["date", "game_date", "start_date"])
    g = g.rename(columns={gid: "game_id", team: "team", pts: "pts", opos: "o_poss_raw", dpos: "d_poss_raw", fga: "fga", orb: "orb", tov: "tov", fta: "fta"})
    for c in ["pts", "o_poss_raw", "d_poss_raw", "fga", "orb", "tov", "fta"]:
        g[c] = base.safe_numeric(g[c])
    g["poss_44"] = g.fga - g.orb + g.tov + 0.44 * g.fta

    s = sched[[sgid, sdate]].drop_duplicates(sgid).rename(columns={sgid: "game_id", sdate: "date"})
    g = g.merge(s, on="game_id", how="left")
    g["date"] = pd.to_datetime(g.date, errors="coerce")
    g = g.dropna(subset=["date", "pts", "o_poss_raw", "d_poss_raw", "poss_44", "team"])
    g = g.sort_values(["date", "game_id", "team"], kind="mergesort").reset_index(drop=True)
    counts = g.groupby("game_id").size()
    g = g[g.game_id.isin(counts[counts == 2].index)].copy()
    g["ix"] = g.groupby("game_id").cumcount()
    opp = g[["game_id", "ix", "team", "pts"]].copy()
    opp["ix"] = 1 - opp.ix
    opp = opp.rename(columns={"team": "opp", "pts": "opp_pts"})
    g = g.merge(opp, on=["game_id", "ix"], validate="one_to_one")
    g["ortg_raw"] = 100 * g.pts / g.o_poss_raw
    g["ortg_44"] = 100 * g.pts / g.poss_44
    g["gposs_raw"] = g.groupby("game_id").o_poss_raw.transform("mean")
    g["gposs_44"] = g.groupby("game_id").poss_44.transform("mean")

    validation = {
        "raw_counts": raw_counts,
        "post_date_two_team_rows": int(len(g)),
        "games": int(g.game_id.nunique()),
        "duplicate_game_team_keys": int(g.duplicated(["game_id", "team"]).sum()),
        "missing_dates": int(g.date.isna().sum()),
        "bad_game_row_counts": int((g.groupby("game_id").size() != 2).sum()),
        "turnover_source_column": tov,
        "mean_abs_team_poss_44_minus_raw": float((g.poss_44 - g.o_poss_raw).abs().mean()),
        "mean_abs_game_poss_44_minus_raw": float((g.gposs_44 - g.gposs_raw).abs().mean()),
        "max_abs_game_poss_44_minus_raw": float((g.gposs_44 - g.gposs_raw).abs().max()),
    }
    return g, validation


def team_season_table(g: pd.DataFrame) -> pd.DataFrame:
    return (
        g.groupby(["season", "team"], as_index=False)
        .agg(pts=("pts", "sum"), opp_pts=("opp_pts", "sum"), op=("o_poss_raw", "sum"), dp=("d_poss_raw", "sum"))
        .assign(ortg=lambda x: 100 * x.pts / x.op, drtg=lambda x: 100 * x.opp_pts / x.dp)
    )


base.load_team_games = load_team_games
base.team_season_table = team_season_table
base.main()
