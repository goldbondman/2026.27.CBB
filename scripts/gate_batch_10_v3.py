from __future__ import annotations

import numpy as np
import pandas as pd
import gate_batch_10 as base

ORIG_P1_FEATURES = base.load_player_features


def _turnover_column(df: pd.DataFrame) -> str:
    for c in ["total_turnovers", "totalTurnovers", "turnovers", "tov", "to", "turnover", "turnovers_total", "total_tov", "tos"]:
        if c in df.columns:
            return c
    candidates = [c for c in df.columns if ("turn" in c.lower() or "tov" in c.lower() or c.lower() == "to") and "team" not in c.lower()]
    if not candidates:
        raise KeyError(f"No turnover-like field found. Columns={list(df.columns)}")
    return candidates[0]


def load_team_games() -> tuple[pd.DataFrame, dict]:
    frames, raw_counts = [], {}
    for sy in base.SEASONS:
        d = pd.read_parquet(base.RAW + f"mbb/team_box/parquet/ncaa_mbb_team_box_{sy}.parquet")
        raw_counts[str(sy)] = int(len(d)); d["season"] = sy; frames.append(d)
    g = pd.concat(frames, ignore_index=True)
    sched = pd.read_parquet(base.RAW + "mbb/ncaa_mbb_schedule_master.parquet")
    gid = base.pick(g,["contest_id","game_id"]); team=base.pick(g,["team","team_name","team_display_name"])
    pts=base.pick(g,["pts","points"]); opos=base.pick(g,["o_poss","offensive_possessions"]); dpos=base.pick(g,["d_poss","defensive_possessions"])
    fga=base.pick(g,["fga","field_goal_attempts"]); orb=base.pick(g,["orb","offensive_rebounds"]); tov=_turnover_column(g); fta=base.pick(g,["fta","free_throw_attempts"])
    print("TEAM_BOX_FIELDS",{"gid":gid,"team":team,"pts":pts,"opos":opos,"dpos":dpos,"fga":fga,"orb":orb,"tov":tov,"fta":fta})
    sgid=base.pick(sched,["contest_id","game_id"]); sdate=base.pick(sched,["date","game_date","start_date"])
    g=g.rename(columns={gid:"game_id",team:"team",pts:"pts",opos:"o_poss_raw",dpos:"d_poss_raw",fga:"fga",orb:"orb",tov:"tov",fta:"fta"})
    for c in ["pts","o_poss_raw","d_poss_raw","fga","orb","tov","fta"]: g[c]=base.safe_numeric(g[c])
    g["poss_44"]=g.fga-g.orb+g.tov+0.44*g.fta
    s=sched[[sgid,sdate]].drop_duplicates(sgid).rename(columns={sgid:"game_id",sdate:"date"}); g=g.merge(s,on="game_id",how="left")
    g["date"]=pd.to_datetime(g.date,errors="coerce"); g=g.dropna(subset=["date","pts","o_poss_raw","d_poss_raw","poss_44","team"])
    g=g.sort_values(["date","game_id","team"],kind="mergesort").reset_index(drop=True)
    cnt=g.groupby("game_id").size(); g=g[g.game_id.isin(cnt[cnt==2].index)].copy(); g["ix"]=g.groupby("game_id").cumcount()
    opp=g[["game_id","ix","team","pts"]].copy(); opp["ix"]=1-opp.ix; opp=opp.rename(columns={"team":"opp","pts":"opp_pts"}); g=g.merge(opp,on=["game_id","ix"],validate="one_to_one")
    g["ortg_raw"]=100*g.pts/g.o_poss_raw; g["ortg_44"]=100*g.pts/g.poss_44; g["gposs_raw"]=g.groupby("game_id").o_poss_raw.transform("mean"); g["gposs_44"]=g.groupby("game_id").poss_44.transform("mean")
    return g,{"raw_counts":raw_counts,"post_date_two_team_rows":int(len(g)),"games":int(g.game_id.nunique()),"duplicate_game_team_keys":int(g.duplicated(["game_id","team"]).sum()),"missing_dates":int(g.date.isna().sum()),"bad_game_row_counts":int((g.groupby("game_id").size()!=2).sum()),"turnover_source_column":tov,"mean_abs_team_poss_44_minus_raw":float((g.poss_44-g.o_poss_raw).abs().mean()),"mean_abs_game_poss_44_minus_raw":float((g.gposs_44-g.gposs_raw).abs().mean()),"max_abs_game_poss_44_minus_raw":float((g.gposs_44-g.gposs_raw).abs().max())}


def team_season_table(g: pd.DataFrame) -> pd.DataFrame:
    return g.groupby(["season","team"],as_index=False).agg(pts=("pts","sum"),opp_pts=("opp_pts","sum"),op=("o_poss_raw","sum"),dp=("d_poss_raw","sum")).assign(ortg=lambda x:100*x.pts/x.op,drtg=lambda x:100*x.opp_pts/x.dp)


def exact_frozen_continuity() -> pd.DataFrame:
    ps=[]
    for sy in base.SEASONS:
        d=pd.read_parquet(base.RAW+f"mbb/player_box/parquet/ncaa_mbb_player_box_{sy}.parquet")
        tc=base.pick(d,["team","team_name","team_display_name"]); nc=base.pick(d,["athlete","player","name","athlete_display_name"]); mc=base.pick(d,["mins","minutes"])
        z=d[[tc,nc,mc]].copy(); z["pk"]=z[nc].astype(str).str.lower().str.replace(r"[^a-z0-9]","",regex=True); z[mc]=pd.to_numeric(z[mc],errors="coerce").fillna(0)
        z=z.groupby([tc,"pk"],dropna=False)[mc].sum().reset_index().rename(columns={tc:"team",mc:"min"}); z["season"]=sy; ps.append(z)
    p=pd.concat(ps,ignore_index=True); cr=[]; trs=[]
    for sy in range(2022,2026):
        a=p[p.season==sy-1]; c=p[p.season==sy]
        same=a.merge(c[["team","pk"]].drop_duplicates(),on=["team","pk"],how="left",indicator=True)
        for tm,z in same.groupby("team"):
            den=z["min"].sum(); cr.append({"season":sy,"team":tm,"rp":z.loc[z._merge=="both","min"].sum()/den if den else np.nan})
        au=a.groupby("pk").filter(lambda z:z.team.nunique()==1).sort_values("min").drop_duplicates("pk",keep="last")
        cu=c.groupby("pk").filter(lambda z:z.team.nunique()==1).sort_values("min").drop_duplicates("pk",keep="last")
        m=au.merge(cu[["pk","team"]],on="pk",suffixes=("_prev","_cur")); m=m[m.team_prev!=m.team_cur].copy(); m["season"]=sy; trs.append(m)
    c=pd.DataFrame(cr); t=pd.concat(trs,ignore_index=True); tf=t.groupby(["season","team_cur"]).agg(tr_min=("min","sum")).reset_index().rename(columns={"team_cur":"team"})
    c=c.merge(tf,on=["season","team"],how="left"); c["tr_min"]=c.tr_min.fillna(0); c["ztr"]=c.groupby("season").tr_min.transform(lambda x:(x-x.mean())/(x.std()+1e-9))
    print("EXACT_CONTINUITY",{"rows":int(len(c)),"mean_rp":float(c.rp.mean()),"transfer_rows":int(len(t))})
    return c


def load_player_features(team_season: pd.DataFrame) -> pd.DataFrame:
    p1=ORIG_P1_FEATURES(team_season)
    exact=exact_frozen_continuity()[["season","team","rp","ztr"]]
    p1=p1.drop(columns=[c for c in ["rp","ztr"] if c in p1.columns]).merge(exact,on=["season","team"],how="left")
    return p1


base.load_team_games=load_team_games
base.team_season_table=team_season_table
base.load_player_features=load_player_features

if __name__ == "__main__":
    base.main()
