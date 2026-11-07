#!/usr/bin/env python3
"""Prospective-information-only Four Factors residual ablation.
HISTORICAL EXPLORATORY only. Never read sealed 2025-26 data.
Train 2023-24 opening 14 days from 2022-23 priors; test 2024-25
opening 14 days from 2023-24 priors. No current-game feature except Y.
F0/F1/F2/F3 keep identical base strength & pace features and
identical Ridge alpha 120, separate margin and total heads.
"""
from __future__ import annotations
import hashlib,json
from pathlib import Path
import numpy as np,pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
ROOT=Path("/tmp/cbb")
OUT=ROOT/"multimodel_ff_controlled"
OUT.mkdir(exist_ok=True)
YEARS=(2023,2024,2025)
STATS=["team_score","field_goals_made","field_goals_attempted",
    "three_point_field_goals_made","three_point_field_goals_attempted",
    "free_throws_made","free_throws_attempted","offensive_rebounds",
    "defensive_rebounds","total_turnovers","steals","fouls"]
COLS=["game_id","game_date","team_id","opponent_team_id","team_home_away",
    "team_score","opponent_team_score"]+[x for x in STATS if x!="team_score"]
def div(a,b):
    return np.asarray(a,dtype=float)/np.maximum(np.asarray(b,dtype=float),1e-6)
def getbox(year):
    z=pd.read_parquet(ROOT/f"eb{year}.parquet",columns=COLS)
    z["game_date"]=pd.to_datetime(z.game_date,errors="coerce")
    z["game_id"]=z.game_id.astype(str)
    z["team_id"]=pd.to_numeric(z.team_id,errors="coerce").astype("Int64").astype(str)
    z["opponent_team_id"]=pd.to_numeric(z.opponent_team_id,errors="coerce").astype("Int64").astype(str)
    for c in STATS+["opponent_team_score"]:z[c]=pd.to_numeric(z[c],errors="coerce")
    z["poss"]=z.field_goals_attempted-z.offensive_rebounds+z.total_turnovers+.475*z.free_throws_attempted
    z=z[z.poss.between(38,115)&z.team_score.notna()&z.opponent_team_score.notna()&z.game_date.notna()]
    return z.drop_duplicates(["game_id","team_id"]).copy()
def pairs(z):
    h=z[z.team_home_away=="home"].add_prefix("h_")
    a=z[z.team_home_away=="away"].add_prefix("a_")
    m=h.merge(a,left_on=["h_game_id","h_team_id","h_opponent_team_id"],
        right_on=["a_game_id","a_opponent_team_id","a_team_id"],validate="one_to_one")
    return m.assign(game_id=m.h_game_id,game_date=m.h_game_date,
      home_id=m.h_team_id,away_id=m.a_team_id,
      home_score=m.h_team_score,away_score=m.a_team_score,
      actual_margin=m.h_team_score-m.a_team_score,actual_total=m.h_team_score+m.a_team_score)
def rates(z):
    opp=z[["game_id","team_id"]+STATS].rename(columns={"team_id":"opp_id",**{c:"allow_"+c for c in STATS}})
    d=z.merge(opp,left_on=["game_id","opponent_team_id"],right_on=["game_id","opp_id"],how="inner",validate="one_to_one")
    cols={"games":("game_id","nunique"),"pace":("poss","mean"),"poss":("poss","sum")}
    for prefix in ["","allow_"]:
        for c in STATS:cols[prefix+c]=(prefix+c,"sum")
    agg=d.groupby("team_id").agg(**cols).reset_index()
    agg=agg[agg.games>=15].copy()
    for prefix in ["","allow_"]:
        zc=lambda c:agg[prefix+c]
        agg[prefix+"off"]=100*div(zc("team_score"),agg.poss)
        agg[prefix+"efg"]=div(zc("field_goals_made")+.5*zc("three_point_field_goals_made"),zc("field_goals_attempted"))
        agg[prefix+"tov"]=div(zc("total_turnovers"),agg.poss)
        # ORB% requires opponent defensive rebounds; use team ORB share of all its available rebounds as approximation.
        opp_def="allow_" if not prefix else ""
        agg[prefix+"orb"]=div(zc("offensive_rebounds"),zc("offensive_rebounds")+agg[opp_def+"defensive_rebounds"])
        agg[prefix+"ftr"]=div(zc("free_throws_attempted"),zc("field_goals_attempted"))
        agg[prefix+"three_share"]=div(zc("three_point_field_goals_attempted"),zc("field_goals_attempted"))
        agg[prefix+"three_pct"]=div(zc("three_point_field_goals_made"),zc("three_point_field_goals_attempted"))
        agg[prefix+"two_pct"]=div(zc("field_goals_made")-zc("three_point_field_goals_made"),
                                  zc("field_goals_attempted")-zc("three_point_field_goals_attempted"))
        agg[prefix+"ft_pct"]=div(zc("free_throws_made"),zc("free_throws_attempted"))
        agg[prefix+"steals"]=div(zc("steals"),agg.poss)
        agg[prefix+"fouls"]=div(zc("fouls"),agg.poss)
        agg[prefix+"ftmade_poss"]=div(zc("free_throws_made"),agg.poss)
        agg[prefix+"ts"]=div(zc("team_score"),2*(zc("field_goals_attempted")+.44*zc("free_throws_attempted")))
    lg=float(100*div(z.team_score.sum(),z.poss.sum()))
    return agg,lg
def build(year,now,prev):
    early=now[now.game_date<now.game_date.min()+pd.Timedelta(days=14)]
    g=pairs(early)
    prior,lg=rates(prev)
    for side,col in [("h","home_id"),("a","away_id")]:
        g=g.merge(prior.add_prefix(side+"_").rename(columns={side+"_team_id":col}),on=col,how="inner",validate="many_to_one")
    # A fixed algebraic structural baseline available to every F0-F3 candidate.
    g["pace_prior"]=(g.h_pace+g.a_pace)/2
    g["base_home_pp100"]=g.h_off+g.a_allow_off-lg
    g["base_away_pp100"]=g.a_off+g.h_allow_off-lg
    g["baseline_margin"]=g.pace_prior*(g.base_home_pp100-g.base_away_pp100)/100+2.7
    g["baseline_total"]=g.pace_prior*(g.base_home_pp100+g.base_away_pp100)/100
    g["strength_diff"]=g.h_off-g.a_off+g.a_allow_off-g.h_allow_off
    g["strength_sum"]=g.h_off+g.a_off+g.h_allow_off+g.a_allow_off
    for k in ["efg","tov","orb","ftr","three_share","three_pct","two_pct","ft_pct","steals","fouls","ftmade_poss","ts"]:
        g["px_home_"+k]=(g["h_"+k]+g["a_allow_"+k])/2
        g["px_away_"+k]=(g["a_"+k]+g["h_allow_"+k])/2
        g["diff_"+k]=g["px_home_"+k]-g["px_away_"+k]
        g["sum_"+k]=g["px_home_"+k]+g["px_away_"+k]
    g["season"]=year
    return g
box={y:getbox(y) for y in YEARS}
tr=build(2024,box[2024],box[2023])
te=build(2025,box[2025],box[2024])
base=["baseline_margin","baseline_total","pace_prior","strength_diff","strength_sum"]
factor=["efg","tov","orb","ftr"]
shoot=["three_share","three_pct","two_pct","ft_pct"]
event=["steals","fouls","ftmade_poss","ts"]
variants={"F0_BASE":base,
 "F1_FOUR":base+[s+k for k in factor for s in ("diff_","sum_")],
 "F2_SHOOT":base+[s+k for k in factor+shoot for s in ("diff_","sum_")],
 "F3_EVENT":base+[s+k for k in factor+shoot+event for s in ("diff_","sum_")]}
full=sorted(set(sum(variants.values(),[])))
for z in [tr,te]:
    z.replace([np.inf,-np.inf],np.nan,inplace=True)
    z.dropna(subset=full+["actual_margin","actual_total"],inplace=True)
assert len(tr)>300 and len(te)>300,(len(tr),len(te))
scores=[];detail=te[["game_id","game_date","home_id","away_id","actual_margin","actual_total"]].copy()
for v,cols in variants.items():
    preds={}
    for target,baseline in [("margin","baseline_margin"),("total","baseline_total")]:
        ytr=tr["actual_"+target].to_numpy()-tr[baseline].to_numpy()
        mdl=make_pipeline(StandardScaler(),Ridge(alpha=120)).fit(tr[cols],ytr)
        pred=te[baseline].to_numpy()+mdl.predict(te[cols])
        preds[target]=pred
        detail[v+"_"+target]=pred
    errors_m=preds["margin"]-te.actual_margin.to_numpy()
    errors_t=preds["total"]-te.actual_total.to_numpy()
    scores.append({"variant":v,"train_n":len(tr),"screen_n":len(te),"feature_count":len(cols),"margin_mae":float(np.abs(errors_m).mean()),"total_mae":float(np.abs(errors_t).mean()),"margin_bias":float(errors_m.mean()),"total_bias":float(errors_t.mean())})
dates=te.game_date.dt.strftime("%Y-%m-%d").to_numpy()
groups=[np.flatnonzero(dates==d) for d in sorted(set(dates))]
rng=np.random.default_rng(422026);paired=[]
for target in ["margin","total"]:
    act=te["actual_"+target].to_numpy()
    e0=np.abs(detail["F0_BASE_"+target].to_numpy()-act)
    for left,right in [("F0_BASE","F1_FOUR"),("F1_FOUR","F2_SHOOT"),("F2_SHOOT","F3_EVENT"),("F0_BASE","F3_EVENT")]:
        gain=np.abs(detail[left+"_"+target].to_numpy()-act)-np.abs(detail[right+"_"+target].to_numpy()-act)
        vals=np.array([np.concatenate([gain[groups[i]] for i in rng.integers(len(groups),size=len(groups))]).mean() for _ in range(1800)])
        paired.append({"target":target,"base":left,"challenger":right,"mae_gain":float(gain.mean()),"date_cluster_boot_ci95":np.quantile(vals,[.025,.975]).tolist()})
detail.to_csv(OUT/"matched_2024_25_first14d.csv",index=False)
out={"status":"HISTORICAL_DEVELOPMENT_ONLY_NOT_SEALED","train_season":"2023-24","screen_season":"2024-25","train_games":len(tr),"screen_games":len(te),"date_groups":len(groups),"source_sha256":{str(y):hashlib.sha256((ROOT/f"eb{y}.parquet").read_bytes()).hexdigest() for y in YEARS},
 "identical_base_features":base,"variant_feature_sets":variants,"ridge_alpha_all":120,"metrics":scores,"paired_gains":paired,
 "limitations":["reused 2024-25 historical research, not untouched","baseline is simplified team PPP, not sealed champion v14","D1, neutral, OT not fully certified","ORBrate from aggregate rebound source approximate","observational residuals do not show causal effects","no current-season 2025-26 sealed holdout accessed","confidence intervals not multiple-hypothesis adjusted"]}
(OUT/"metrics.json").write_text(json.dumps(out,indent=2))
print("COHORT",len(tr),len(te),"dates",len(groups),flush=True)
print("SCORES",json.dumps(scores),flush=True)
print("PAIRED",json.dumps(paired),flush=True)
