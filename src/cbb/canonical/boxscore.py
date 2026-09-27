"""Canonical, versioned box-score semantics."""

import pandas as pd

REQUIRED = {"fga", "orb", "total_turnovers", "fta", "points"}


def compile_team_games(frame: pd.DataFrame) -> pd.DataFrame:
    missing = REQUIRED - set(frame.columns)
    if missing:
        raise ValueError(f"missing canonical fields: {sorted(missing)}")
    out = frame.copy()
    if {"game_id", "team_id"} <= set(out) and out.duplicated(["game_id", "team_id"]).any():
        raise ValueError("duplicate game/team key")
    if "game_id" in out and not (out.groupby("game_id").size() == 2).all():
        raise ValueError("each game must contain exactly two team rows")
    # total_turnovers already includes team turnovers in this source representation.
    out["tov"] = out["total_turnovers"].astype(float)
    out["possessions"] = out["fga"] - out["orb"] + out["tov"] + 0.44 * out["fta"]
    if (out["possessions"] <= 0).any():
        raise ValueError("non-positive canonical possessions")
    out["ortg"] = 100 * out["points"] / out["possessions"]
    if "opponent_points" in out:
        out["drtg"] = 100 * out["opponent_points"] / out["possessions"]
    if {"fgm", "three_pm"} <= set(out):
        out["efg"] = (out["fgm"] + 0.5 * out["three_pm"]) / out["fga"]
    out["tov_rate"] = out["tov"] / out["possessions"]
    out["ftr"] = out["fta"] / out["fga"]
    return out
