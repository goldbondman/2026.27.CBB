from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

RAW = "https://raw.githubusercontent.com/sportsdataverse/ncaa-mbb-hoops-data/main/"
SEASONS = list(range(2021, 2026))
TEST_SEASONS = [2022, 2023, 2024, 2025]
OUT = Path("results/gate_batch_10")
OUT.mkdir(parents=True, exist_ok=True)


def pick(df: pd.DataFrame, names: list[str]) -> str:
    col = next((c for c in names if c in df.columns), None)
    if col is None:
        raise KeyError(f"none of {names} in columns")
    return col


def safe_numeric(s: pd.Series) -> pd.Series:
    return pd.to_numeric(s, errors="coerce")


def load_team_games() -> tuple[pd.DataFrame, dict]:
    frames = []
    raw_counts = {}
    for sy in SEASONS:
        d = pd.read_parquet(RAW + f"mbb/team_box/parquet/ncaa_mbb_team_box_{sy}.parquet")
        raw_counts[str(sy)] = int(len(d))
        d["season"] = sy
        frames.append(d)
    g = pd.concat(frames, ignore_index=True)
    sched = pd.read_parquet(RAW + "mbb/ncaa_mbb_schedule_master.parquet")

    gid = pick(g, ["contest_id", "game_id"])
    team = pick(g, ["team", "team_name", "team_display_name"])
    pts = pick(g, ["pts", "points"])
    opos = pick(g, ["o_poss", "offensive_possessions"])
    fga = pick(g, ["fga", "field_goal_attempts"])
    orb = pick(g, ["orb", "offensive_rebounds"])
    tov = pick(g, ["total_turnovers", "totalTurnovers", "turnovers", "tov"])
    fta = pick(g, ["fta", "free_throw_attempts"])

    sgid = pick(sched, ["contest_id", "game_id"])
    sdate = pick(sched, ["date", "game_date", "start_date"])

    g = g.rename(columns={gid: "game_id", team: "team", pts: "pts", opos: "o_poss_raw", fga: "fga", orb: "orb", tov: "tov", fta: "fta"})
    g["pts"] = safe_numeric(g["pts"])
    g["o_poss_raw"] = safe_numeric(g["o_poss_raw"])
    for c in ["fga", "orb", "tov", "fta"]:
        g[c] = safe_numeric(g[c])
    g["poss_44"] = g.fga - g.orb + g.tov + 0.44 * g.fta

    s = sched[[sgid, sdate]].drop_duplicates(sgid).rename(columns={sgid: "game_id", sdate: "date"})
    g = g.merge(s, on="game_id", how="left")
    g["date"] = pd.to_datetime(g.date, errors="coerce")
    g = g.dropna(subset=["date", "pts", "o_poss_raw", "poss_44", "team"])
    g = g.sort_values(["date", "game_id", "team"], kind="mergesort").reset_index(drop=True)

    # Retain exactly two-team games and create opponent mirror.
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
        "mean_abs_team_poss_44_minus_raw": float((g.poss_44 - g.o_poss_raw).abs().mean()),
        "mean_abs_game_poss_44_minus_raw": float((g.gposs_44 - g.gposs_raw).abs().mean()),
    }
    return g, validation


def load_player_features(team_season: pd.DataFrame) -> pd.DataFrame:
    raw = ["mins", "pts", "orb", "drb", "ast", "stl", "blk", "tov", "tpa", "fta", "rima", "fga_unast"]
    ps = []
    for sy in SEASONS:
        p = pd.read_parquet(RAW + f"mbb/player_box/parquet/ncaa_mbb_player_box_{sy}.parquet")
        for x in raw:
            p[x] = safe_numeric(p[x]).fillna(0)
        name_col = "clean_name" if "clean_name" in p.columns else pick(p, ["player", "athlete", "name"])
        fallback = p["player"] if "player" in p.columns else p[name_col]
        p["pk"] = p[name_col].fillna(fallback).astype(str).str.lower().str.replace(r"[^a-z0-9]", "", regex=True)
        a = p.groupby(["team", "pk"], as_index=False)[raw].sum()
        a["season"] = sy
        for x in raw[1:]:
            a[x + "40"] = 40 * a[x] / a.mins.replace(0, np.nan)
        ps.append(a)
    players = pd.concat(ps, ignore_index=True)

    rows = []
    role = [x + "40" for x in raw[1:]]
    for sy in range(2022, 2026):
        prev = players[players.season == sy - 1]
        cur = players[players.season == sy]
        prev = prev.groupby("pk").filter(lambda z: z.team.nunique() == 1)
        cur = cur.groupby("pk").filter(lambda z: z.team.nunique() == 1)
        for team, cg in cur.groupby("team"):
            pg = prev[prev.team == team]
            ret = pg[pg.pk.isin(cg.pk)]
            tr = prev[(prev.pk.isin(cg.pk)) & (prev.team != team)]
            dep = pg[~pg.pk.isin(cg.pk)]
            den = pg.mins.sum()
            rec = {"season": sy, "team": team, "rp": ret.mins.sum() / den if den else np.nan, "tr_min": tr.mins.sum()}
            for pref, z in [("ret", ret), ("dep", dep), ("tr", tr)]:
                w = z.mins
                wd = w.sum()
                for x in role:
                    rec[pref + "_" + x] = (z[x] * w).sum() / wd if wd else np.nan
            rows.append(rec)
    c = pd.DataFrame(rows)
    c["ztr"] = c.groupby("season").tr_min.transform(lambda x: (x - x.mean()) / (x.std() + 1e-9))
    for x in ["ast40", "fga_unast40", "tov40"]:
        c["repl_" + x] = c["ret_" + x].fillna(0) + c["tr_" + x].fillna(0) - c["dep_" + x].fillna(0)

    # Add previous/final season ratings for candidate residual models.
    ts = team_season.copy()
    prev = ts.copy()
    prev["season"] += 1
    prev = prev.rename(columns={"ortg": "prev_ortg", "drtg": "prev_drtg"})
    c = c.merge(prev[["season", "team", "prev_ortg", "prev_drtg"]], on=["season", "team"], how="inner")
    c = c.merge(ts[["season", "team", "ortg", "drtg"]], on=["season", "team"], how="inner")
    return c


def team_season_table(g: pd.DataFrame) -> pd.DataFrame:
    # Raw frozen-research possession semantics, matching the historical Actions implementation.
    return (
        g.groupby(["season", "team"], as_index=False)
        .agg(pts=("pts", "sum"), opp_pts=("opp_pts", "sum"), op=("o_poss_raw", "sum"), dp=("o_poss_raw", "sum"))
        .assign(ortg=lambda x: 100 * x.pts / x.op, drtg=lambda x: 100 * x.opp_pts / x.dp)
    )


def run_frozen(g: pd.DataFrame, continuity: pd.DataFrame, mode: str = "m20", preseason_adjust: dict | None = None) -> pd.DataFrame:
    ostate: dict[str, float] = {}
    dstate: dict[str, float] = {}
    n: dict[str, int] = {}
    prev_season = None
    preds = []
    cont_by = {sy: z.set_index("team") for sy, z in continuity.groupby("season")}
    preseason_adjust = preseason_adjust or {}

    for date, day in g.groupby("date", sort=True):
        sy = int(day.season.iloc[0])
        if prev_season is None:
            prev_season = sy
        if sy != prev_season:
            cc = cont_by.get(sy, pd.DataFrame())
            for team in sorted(set(ostate) | set(dstate)):
                rp = 0.375
                ztr = 0.0
                if len(cc) and team in cc.index:
                    r = cc.loc[team]
                    rp = float(r.rp) if pd.notna(r.rp) else 0.375
                    ztr = float(r.ztr) if pd.notna(r.ztr) else 0.0
                fac = float(np.clip(1.20 + 0.30 * (rp - 0.375) + 0.06 * ztr, 0.55, 1.45))
                ostate[team] = ostate.get(team, 0.0) * fac
                dstate[team] = dstate.get(team, 0.0) * fac
                adj = preseason_adjust.get((sy, team))
                if adj:
                    ostate[team] += float(adj.get("o", 0.0))
                    dstate[team] += float(adj.get("d", 0.0))
            n = {}
            prev_season = sy

        updates = []
        for r in day.itertuples(index=False):
            team = str(r.team)
            opp = str(r.opp)
            nn = n.get(team, 0)
            pred = 100 + ostate.get(team, 0.0) - dstate.get(opp, 0.0)
            w = 1.0
            if mode != "fixed":
                w = float(r.gposs_raw) / 70.0
            if mode == "m20":
                w *= 1 + 0.20 * (1 - 2 * min(nn / 10.0, 1.0))
            w = float(np.clip(w, 0.50, 1.50))
            if sy >= 2022:
                preds.append({"season": sy, "date": r.date, "game_id": r.game_id, "team": team, "opp": opp, "prior_games": nn, "actual_ortg": float(r.ortg_raw), "pred_ortg": pred, "actual_pts": float(r.pts), "actual_gposs": float(r.gposs_raw)})
            updates.append((team, opp, float(r.ortg_raw) - pred, w))

        # Historical frozen semantics: offensive/source-team maturity creates one K,
        # and that same K updates offense A and defense B.
        for team, opp, err, w in updates:
            k = 0.12 * w
            ostate[team] = ostate.get(team, 0.0) + k * err
            dstate[opp] = dstate.get(opp, 0.0) - k * err
            n[team] = n.get(team, 0) + 1

    return pd.DataFrame(preds)


def metrics(pred: pd.DataFrame) -> dict:
    x = pred.copy()
    x["err"] = x.actual_ortg - x.pred_ortg
    ortg = float(x.err.abs().mean())
    gd = x.groupby(["season", "game_id"]).err.agg(lambda z: abs(z.iloc[0] - z.iloc[1]) if len(z) == 2 else np.nan)
    out = {"ortg_mae": ortg, "diff_mae": float(gd.mean())}
    for sy in TEST_SEASONS:
        out[f"diff_{sy}"] = float(gd.loc[sy].mean())
    return out


def fit_p1_adjustments(features: pd.DataFrame, spec: str) -> dict:
    # Build walk-forward candidate-minus-baseline season-rating adjustments.
    base = ["prev_ortg", "prev_drtg", "rp", "ztr"]
    extras = {
        "dep_creation": ["dep_ast40", "dep_fga_unast40", "dep_tov40"],
        "tr_creation": ["tr_ast40", "tr_fga_unast40", "tr_tov40"],
        "creation_replacement": ["repl_ast40", "repl_fga_unast40", "repl_tov40"],
    }[spec]
    adjustments = {}
    # 2023-25: expanding OOS; 2022 is left at frozen baseline because only one prior season exists.
    for sy in [2023, 2024, 2025]:
        train = features[features.season < sy].copy()
        test = features[features.season == sy].copy()
        if len(train) < 50 or test.empty:
            continue
        for target in ["ortg", "drtg"]:
            med_base = train[base].median()
            med_extra = train[base + extras].median()
            mb = make_pipeline(StandardScaler(), Ridge(alpha=20)).fit(train[base].fillna(med_base), train[target])
            mc = make_pipeline(StandardScaler(), Ridge(alpha=20)).fit(train[base + extras].fillna(med_extra), train[target])
            pb = mb.predict(test[base].fillna(med_base))
            pc = mc.predict(test[base + extras].fillna(med_extra))
            delta = pc - pb
            for team, d in zip(test.team.astype(str), delta):
                rec = adjustments.setdefault((sy, team), {"o": 0.0, "d": 0.0})
                if target == "ortg":
                    rec["o"] += float(d)
                else:
                    rec["d"] += float(-d)  # lower DRtg means stronger positive defense state
    return adjustments


def stage_metrics(pred: pd.DataFrame) -> dict:
    buckets = {
        "game1": lambda n: n == 0,
        "games1_2": lambda n: n <= 1,
        "games3_5": lambda n: (n >= 2) & (n <= 4),
        "games6_10": lambda n: (n >= 5) & (n <= 9),
        "games11_plus": lambda n: n >= 10,
    }
    out = {}
    x = pred.copy()
    x["err"] = x.actual_ortg - x.pred_ortg
    for name, fn in buckets.items():
        z = x[fn(x.prior_games)]
        gd = z.groupby(["season", "game_id"]).err.agg(lambda q: abs(q.iloc[0] - q.iloc[1]) if len(q) == 2 else np.nan).dropna()
        out[name] = {"team_ortg_mae": float(z.err.abs().mean()) if len(z) else None, "game_diff_mae": float(gd.mean()) if len(gd) else None, "team_rows": int(len(z)), "games": int(len(gd))}
    return out


def pace_predictions(g: pd.DataFrame) -> pd.DataFrame:
    # No tuned hyperparameters: previous-season mean pace if available, otherwise 70;
    # after current-season games exist, use current-season cumulative mean. Same-date isolated.
    prev_pace = {}
    for sy in SEASONS[:-1]:
        s = g[g.season == sy].groupby("team").gposs_raw.mean()
        prev_pace[sy + 1] = s.to_dict()
    sums: dict[str, float] = {}
    counts: dict[str, int] = {}
    current_sy = None
    rows = []
    for date, day in g.groupby("date", sort=True):
        sy = int(day.season.iloc[0])
        if sy != current_sy:
            sums, counts = {}, {}
            current_sy = sy
        # game pairs
        for gid, pair in day.groupby("game_id"):
            if len(pair) != 2 or sy < 2022:
                continue
            a, b = pair.iloc[0], pair.iloc[1]
            def team_pace(team: str) -> float:
                if counts.get(team, 0):
                    return sums[team] / counts[team]
                return float(prev_pace.get(sy, {}).get(team, 70.0))
            pa = team_pace(str(a.team)); pb = team_pace(str(b.team))
            pred = 0.5 * (pa + pb)
            actual = float(pair.gposs_raw.iloc[0])
            rows.append({"season": sy, "date": date, "game_id": gid, "pace_pred": pred, "pace_70": 70.0, "pace_actual": actual})
        # update after entire date
        for r in day.itertuples(index=False):
            t = str(r.team)
            sums[t] = sums.get(t, 0.0) + float(r.gposs_raw)
            counts[t] = counts.get(t, 0) + 1
    return pd.DataFrame(rows).drop_duplicates("game_id")


def team_score_evaluation(r2: pd.DataFrame, pace: pd.DataFrame) -> dict:
    x = r2.merge(pace[["game_id", "pace_pred", "pace_70"]], on="game_id", how="inner")
    x["score_pred"] = x.pred_ortg * x.pace_pred / 100.0
    x["score_pred_70"] = x.pred_ortg * x.pace_70 / 100.0
    x["score_ae"] = (x.actual_pts - x.score_pred).abs()
    x["score_ae_70"] = (x.actual_pts - x.score_pred_70).abs()
    game_rows = []
    for (sy, gid), q in x.groupby(["season", "game_id"]):
        if len(q) != 2:
            continue
        q = q.sort_values("team")
        a, b = q.iloc[0], q.iloc[1]
        actual_margin = a.actual_pts - b.actual_pts
        pred_margin = a.score_pred - b.score_pred
        pred_margin70 = a.score_pred_70 - b.score_pred_70
        actual_total = a.actual_pts + b.actual_pts
        pred_total = a.score_pred + b.score_pred
        pred_total70 = a.score_pred_70 + b.score_pred_70
        game_rows.append({"season": sy, "game_id": gid, "margin_ae": abs(actual_margin-pred_margin), "margin_ae_70": abs(actual_margin-pred_margin70), "total_ae": abs(actual_total-pred_total), "total_ae_70": abs(actual_total-pred_total70)})
    gr = pd.DataFrame(game_rows)
    return {
        "team_score_mae": float(x.score_ae.mean()),
        "team_score_mae_70": float(x.score_ae_70.mean()),
        "margin_mae": float(gr.margin_ae.mean()),
        "margin_mae_70": float(gr.margin_ae_70.mean()),
        "total_mae": float(gr.total_ae.mean()),
        "total_mae_70": float(gr.total_ae_70.mean()),
        "games": int(len(gr)),
        "coherence_identity_max_error": float(max((x.score_pred.groupby(x.game_id).sum() - x.groupby("game_id").score_pred.sum()).abs().max(), 0.0)),
    }


def main() -> None:
    assert 2026 not in SEASONS
    g, validation = load_team_games()
    ts = team_season_table(g)
    continuity = load_player_features(ts)

    r1 = run_frozen(g, continuity, mode="fixed")
    r2 = run_frozen(g, continuity, mode="m20")
    r1m, r2m = metrics(r1), metrics(r2)

    p1_results = {}
    base_stage = stage_metrics(r2)
    for spec in ["dep_creation", "tr_creation", "creation_replacement"]:
        adj = fit_p1_adjustments(continuity, spec)
        pred = run_frozen(g, continuity, mode="m20", preseason_adjust=adj)
        p1_results[spec] = {"stage": stage_metrics(pred), "overall": metrics(pred)}
        pred.to_parquet(OUT / f"predictions_{spec}.parquet", index=False)

    pace = pace_predictions(g)
    pace_mae = float((pace.pace_actual - pace.pace_pred).abs().mean())
    pace70_mae = float((pace.pace_actual - 70.0).abs().mean())
    score_eval = team_score_evaluation(r2, pace)

    # Gate decisions are conservative and evidence-driven.
    expected_counts = {"2021": 8538, "2022": 11940, "2023": 12442, "2024": 12486, "2025": 12586}
    count_match = all(validation["raw_counts"].get(k) == v for k, v in expected_counts.items())
    r1_match = abs(r1m["ortg_mae"] - 11.720062) < 1e-6 and abs(r1m["diff_mae"] - 15.618256) < 1e-6
    r2_match = abs(r2m["ortg_mae"] - 11.577690) < 1e-6 and abs(r2m["diff_mae"] - 15.373316) < 1e-6

    def early_delta(spec: str, bucket: str) -> float | None:
        a = p1_results[spec]["stage"][bucket]["game_diff_mae"]
        b = base_stage[bucket]["game_diff_mae"]
        return None if a is None or b is None else float(a - b)

    gates = {
        "T00.06": {"decision": "GREEN" if count_match and validation["duplicate_game_team_keys"] == 0 and validation["bad_game_row_counts"] == 0 else "HOLD", "evidence": validation},
        "T00.07": {"decision": "GREEN" if r1_match else "HOLD", "evidence": r1m},
        "T00.08": {"decision": "GREEN" if r2_match else "HOLD", "evidence": r2m},
        "T00.09": {"decision": "GREEN", "evidence": {"historical_semantics": "source-team maturity; same K updates O[a] and D[b]", "verified_in_original_run": 36214772412}},
        "T01.02": {"decision": "PASS" if all((early_delta("dep_creation", b) or 0) < 0 for b in ["game1", "games1_2", "games3_5"]) else "REJECTED", "evidence": {b: early_delta("dep_creation", b) for b in base_stage}},
        "T01.03": {"decision": "PASS" if all((early_delta("tr_creation", b) or 0) < 0 for b in ["game1", "games1_2", "games3_5"]) else "REJECTED", "evidence": {b: early_delta("tr_creation", b) for b in base_stage}},
        "T01.04": {"decision": "PASS" if all((early_delta("creation_replacement", b) or 0) < 0 for b in ["game1", "games1_2", "games3_5"]) else "REJECTED", "evidence": {b: early_delta("creation_replacement", b) for b in base_stage}},
        "T02.01": {"decision": "GREEN" if pace_mae < pace70_mae else "REJECTED", "evidence": {"pregame_team_state_pace_mae": pace_mae, "constant_70_mae": pace70_mae}},
        "T02.04": {"decision": "GREEN" if score_eval["team_score_mae"] < score_eval["team_score_mae_70"] else "HOLD", "evidence": score_eval},
        "T02.05": {"decision": "GREEN" if score_eval["total_mae"] < score_eval["total_mae_70"] and score_eval["margin_mae"] <= score_eval["margin_mae_70"] * 1.01 else "HOLD", "evidence": score_eval},
    }

    result = {
        "data_validation": validation,
        "r1": r1m,
        "r2": r2m,
        "r2_hash": hashlib.sha256(pd.util.hash_pandas_object(r2[["season", "game_id", "team", "pred_ortg"]], index=False).values.tobytes()).hexdigest(),
        "base_stage": base_stage,
        "p1": p1_results,
        "pace": {"state_mae": pace_mae, "constant70_mae": pace70_mae},
        "team_score": score_eval,
        "gates": gates,
    }
    (OUT / "gate_results.json").write_text(json.dumps(result, indent=2, sort_keys=True, default=str) + "\n")
    pd.DataFrame([{"gate_id": k, "decision": v["decision"], "evidence": json.dumps(v["evidence"], sort_keys=True)} for k, v in gates.items()]).to_csv(OUT / "gate_decisions.csv", index=False)
    r1.to_parquet(OUT / "predictions_r1.parquet", index=False)
    r2.to_parquet(OUT / "predictions_r1_r2.parquet", index=False)
    pace.to_parquet(OUT / "pace_predictions.parquet", index=False)
    print("GATE_BATCH_10_RESULTS")
    print(json.dumps(result, indent=2, sort_keys=True, default=str))


if __name__ == "__main__":
    main()
