#!/usr/bin/env python3
"""MULTIMODEL historical early-season architecture smoke screen, NOT a production model.

All inputs (one year earlier team box) predate evaluated season, except
M4 within-season Elo, updated only after prior calendar days finish.
Five architecture cards; only M1, M3, M4, M5 executable in this baseline.
M2 player/roster remains BLOCKED by published-before-tipoff roster evidence.
The 2025-26 screen has ALREADY been explored. This is NOT untouched validation.
"""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.ensemble import HistGradientBoostingRegressor
from sklearn.linear_model import Ridge, LinearRegression
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

ROOT=Path("/tmp/cbb")
OUT=ROOT/"multimodel"
OUT.mkdir(parents=True,exist_ok=True)
YEARS=(2024,2025,2026)
N_MIN=15
FGA="field_goals_attempted"
COLS=["game_id","game_date","team_id","opponent_team_id","team_home_away","team_display_name",
      "team_score","opponent_team_score","field_goals_made","field_goals_attempted",
      "three_point_field_goals_made","three_point_field_goals_attempted",
      "free_throws_made","free_throws_attempted","offensive_rebounds",
      "defensive_rebounds","total_turnovers","steals","fouls"]
STATS=["team_score","field_goals_made","field_goals_attempted",
       "three_point_field_goals_made","three_point_field_goals_attempted",
       "free_throws_made","free_throws_attempted","offensive_rebounds",
       "defensive_rebounds","total_turnovers","steals","fouls"]
def safe(a,b):return np.asarray(a,dtype=float)/np.maximum(np.asarray(b,dtype=float),1e-6)
def clamp(a,lo,hi):return np.clip(a,lo,hi)
def box(y):
    d=pd.read_parquet(ROOT/f"eb{y}.parquet",columns=COLS).copy()
    for c in STATS+["opponent_team_score"]:
        d[c]=pd.to_numeric(d[c],errors="coerce")
    d["team_id"]=pd.to_numeric(d.team_id,errors="coerce").astype("Int64").astype(str)
    d["opponent_team_id"]=pd.to_numeric(d.opponent_team_id,errors="coerce").astype("Int64").astype(str)
    d["game_date"]=pd.to_datetime(d.game_date,errors="coerce")
    d["game_id"]=d.game_id.astype(str)
    d["pace"]=d.field_goals_attempted-d.offensive_rebounds+d.total_turnovers+.475*d.free_throws_attempted
    d=d[(d.pace.between(38,115)) & (d.team_score>=0)& d.opponent_team_score.notna()
        & d.game_date.notna()].copy()
    d=d.drop_duplicates(["game_id","team_id"])
    return d

def pairs(d):
    h=d[d.team_home_away=="home"].copy()
    a=d[d.team_home_away=="away"].copy()
    h=h.add_prefix("h_");a=a.add_prefix("a_")
    g=h.merge(a,left_on=["h_game_id","h_team_id","h_opponent_team_id"],
                right_on=["a_game_id","a_opponent_team_id","a_team_id"],validate="one_to_one")
    g["game_id"]=g.h_game_id
    g["game_date"]=g.h_game_date
    g["home_id"]=g.h_team_id
    g["away_id"]=g.a_team_id
    g["home_name"]=g.h_team_display_name
    g["away_name"]=g.a_team_display_name
    g["home_score"]=g.h_team_score
    g["away_score"]=g.a_team_score
    g["actual_margin"]=g.home_score-g.away_score
    g["actual_total"]=g.home_score+g.away_score
    g["actual_pace"]=(g.h_pace+g.a_pace)/2
    return g

def preseason(d):
    p=pairs(d)
    # Two team-game rows, each supplemented by its opponent's actual box (for prior-season defensive rates).
    hh=d.merge(d[["game_id","team_id"]+STATS].rename(columns={"team_id":"opponent_id",**{c:"op_"+c for c in STATS}}),
        left_on=["game_id","opponent_team_id"],right_on=["game_id","opponent_id"],how="inner",validate="one_to_one")
    ag={"games":("game_id","nunique"),"points":("team_score","sum"),
        "op_points":("op_team_score","sum"),"pace_sum":("pace","sum"),"pace":("pace","mean")}
    for c in STATS:
        ag[c+"_sum"]=(c,"sum")
        ag["allow_"+c+"_sum"]=("op_"+c,"sum")
    v=hh.groupby("team_id").agg(**ag).reset_index()
    v=v[v.games>=N_MIN].copy()
    v["off_pp100"]=100*safe(v.points,v.pace_sum)
    v["def_pp100"]=100*safe(v.op_points,v.pace_sum)
    v["points_pg"]=safe(v.points,v.games)
    v["allowed_pg"]=safe(v.op_points,v.games)
    v["margin_pg"]=v.points_pg-v.allowed_pg
    v["win_proxy"]=v.margin_pg
    for pref in ["","allow_"]:
        z=lambda c:v[pref+c+"_sum"].astype(float)
        v[pref+"tov_per_poss"]=safe(z("total_turnovers"),v.pace_sum)
        v[pref+"orb_per_fga"]=safe(z("offensive_rebounds"),z("field_goals_attempted"))
        v[pref+"fta_per_fga"]=safe(z("free_throws_attempted"),z("field_goals_attempted"))
        v[pref+"three_share"]=safe(z("three_point_field_goals_attempted"),z("field_goals_attempted"))
        v[pref+"three_pct"]=safe(z("three_point_field_goals_made"),z("three_point_field_goals_attempted"))
        v[pref+"two_pct"]=safe(z("field_goals_made")-z("three_point_field_goals_made"),
                              z("field_goals_attempted")-z("three_point_field_goals_attempted"))
        v[pref+"ft_pct"]=safe(z("free_throws_made"),z("free_throws_attempted"))
        v[pref+"efg"]=safe(z("field_goals_made")+.5*z("three_point_field_goals_made"),
                            z("field_goals_attempted"))
        v[pref+"steal_per_poss"]=safe(z("steals"),v.pace_sum)
    lg={"pp100":float(100*safe(hh.team_score.sum(),hh.pace.sum())),"pace":float(hh.pace.mean()),
         "three":float(safe(hh.three_point_field_goals_made.sum(),hh.three_point_field_goals_attempted.sum()))}
    return v,lg

def current_elo(today,prior):
    # Prior-season Elo is trained on results in that prior season. Update only after each prior DATE.
    def updates(states,frame):
        for dt,sub in frame.sort_values(["game_date","game_id"]).groupby("game_date",sort=True):
            change={}
            for r in sub.itertuples(index=False):
                h=str(r.home_id);a=str(r.away_id)
                rh=states.get(h,1500.);ra=states.get(a,1500.)
                z=1/(1+10**(-(rh+45-ra)/400))
                outcome=1. if r.actual_margin>0 else .0 if r.actual_margin<0 else .5
                delta=22*(outcome-z)
                change[h]=change.get(h,0.)+delta;change[a]=change.get(a,0.)-delta
            for k,v in change.items():states[k]=states.get(k,1500.)+v
        return states
    states=updates({},pairs(prior))
    states={k:1500+.75*(v-1500) for k,v in states.items()}
    # previous-year Elo is seed, current-season pregame Elo updates from previous dates only.
    frame=pairs(today).sort_values(["game_date","game_id"])
    latest={}
    records=[]
    for dt,sub in frame.groupby("game_date",sort=True):
        to_update=[]
        for r in sub.itertuples(index=False):
            h=str(r.home_id);a=str(r.away_id)
            rest_h=(dt-latest[h]).days if h in latest else 7
            rest_a=(dt-latest[a]).days if a in latest else 7
            records.append({"game_id":r.game_id,"elo_diff":states.get(h,1500.)-states.get(a,1500.),
                            "rest_diff":float(clamp(rest_h-rest_a,-6,6)),
                            "rest_h":float(clamp(rest_h,0,7)),"rest_a":float(clamp(rest_a,0,7))})
            to_update.append(r)
        change={}
        for r in to_update:
            h=str(r.home_id);a=str(r.away_id)
            rh=states.get(h,1500.);ra=states.get(a,1500.)
            prob=1/(1+10**(-(rh+45-ra)/400))
            outcome=1. if r.actual_margin>0 else .0 if r.actual_margin<0 else .5
            delta=22*(outcome-prob)
            change[h]=change.get(h,0.)+delta;change[a]=change.get(a,0.)-delta
        for k,v in change.items():states[k]=states.get(k,1500.)+v
        for r in to_update:latest[str(r.home_id)]=dt;latest[str(r.away_id)]=dt
    return pd.DataFrame(records)

def build_eval(y,d,prior_v,lg,elo):
    start=d.game_date.min()
    first=d[d.game_date.lt(start+pd.Timedelta(days=14))]
    g=pairs(first)
    for side,tgt in [("h","home_id"),("a","away_id")]:
        v=prior_v.add_prefix(side+"_").rename(columns={side+"_team_id":tgt})
        g=g.merge(v,on=tgt,how="inner",validate="many_to_one")
    g=g.merge(elo,on="game_id",validate="one_to_one")
    # Prefer PREVIOUS-SEASON pace features, not the current game box pace fields.
    # Current-game pace only appears in actual_pace, a postgame evaluation target.
    for side in ("h","a"):
        g[side+"_pace"]=g[side+"_pace_y"]
    # M1: simple team-only prior-year opponent-adjusted efficiency factors
    g["m1_pace"]=(g.h_pace+g.a_pace)/2
    g["m1_home"]=(lg["pp100"]+(g.h_off_pp100-lg["pp100"])+(g.a_def_pp100-lg["pp100"]))*g.m1_pace/100+1.35
    g["m1_away"]=(lg["pp100"]+(g.a_off_pp100-lg["pp100"])+(g.h_def_pp100-lg["pp100"]))*g.m1_pace/100-1.35
    # M3: independently constructed possession-scoring process (team box prior only).
    g["m3_pace"]=np.sqrt(g.h_pace*g.a_pace)
    for side,other in [("h","a"),("a","h")]:
        def blend(feat):
            return (g[side+"_"+feat]+g[other+"_allow_"+feat])/2
        T=clamp(blend("tov_per_poss"),.05,.3)
        O=clamp(blend("orb_per_fga"),.025,.3)
        F=clamp(blend("fta_per_fga"),.1,.8)
        R3=clamp(blend("three_share"),.05,.7)
        P2=clamp(blend("two_pct"),.25,.8)
        P3=clamp(blend("three_pct"),.15,.55)
        PFT=clamp(blend("ft_pct"),.45,.9)
        fga_per_poss=(1-T)/(1-O+.475*F)
        score_per_poss=fga_per_poss*(2*(1-R3)*P2+3*R3*P3+F*PFT)
        g[f"m3_{'home' if side=='h' else 'away'}"]=g.m3_pace*score_per_poss+(1.35 if side=='h' else -1.35)
        g[f"x_{side}_efg"]=(1-R3)*P2+1.5*R3*P3
        g[f"x_{side}_tov"]=T
        g[f"x_{side}_orb"]=O
        g[f"x_{side}_ftr"]=F
        g[f"x_{side}_three_share"]=R3
        g[f"x_{side}_three_pct"]=P3
        g[f"x_{side}_two_pct"]=P2
        g[f"x_{side}_ft_pct"]=PFT
    for factor in ["efg","orb","ftr","three_share","three_pct","two_pct","ft_pct"]:
        g["diff_"+factor]=g["x_h_"+factor]-g["x_a_"+factor]
    g["diff_tov"]=g.x_a_tov-g.x_h_tov
    # M5 alternate nonlinear architecture: raw style, prior score, pace/strength contrasts; no M1 outputs.
    g["prior_score_diff"]=g.h_points_pg-g.a_points_pg
    g["prior_def_diff"]=g.h_allowed_pg-g.a_allowed_pg
    g["prior_strength_gap"]=g.h_margin_pg-g.a_margin_pg
    g["prior_pace_gap"]=g.h_pace-g.a_pace
    g["prior_three_gap"]=g.h_three_share-g.a_three_share
    g["prior_tov_gap"]=g.h_tov_per_poss-g.a_tov_per_poss
    g["prior_orb_gap"]=g.h_orb_per_fga-g.a_orb_per_fga
    g["prior_ftr_gap"]=g.h_fta_per_fga-g.a_fta_per_fga
    g["prior_efg_gap"]=g.h_efg-g.a_efg
    g["prior_points_sum"]=g.h_points_pg+g.a_points_pg
    g["prior_allowed_sum"]=g.h_allowed_pg+g.a_allowed_pg
    g["season_eval"]=y
    return g

raw={y:box(y) for y in YEARS}
prior={y:preseason(raw[y]) for y in [2024,2025]}
coh={}
for y in [2025,2026]:
    prior_data,lg=prior[y-1]
    coh[y]=build_eval(y,raw[y],prior_data,lg,current_elo(raw[y],raw[y-1]))
tr=coh[2025].copy();te=coh[2026].copy()
# Same eligible games for every executable model, with all input rows complete.
cols_tree=["h_points_pg","a_points_pg","h_allowed_pg","a_allowed_pg","prior_strength_gap",
           "h_pace","a_pace","prior_pace_gap","prior_three_gap","prior_tov_gap",
           "prior_orb_gap","prior_ftr_gap","prior_efg_gap","h_games","a_games",
           "prior_score_diff","prior_def_diff","prior_points_sum","prior_allowed_sum",
           "rest_diff"]
for d in [tr,te]:
    d.replace([np.inf,-np.inf],np.nan,inplace=True)
    d.dropna(subset=cols_tree+["actual_margin","actual_total","m1_home","m1_away","m3_home","m3_away"],inplace=True)
# M4 independent Elo-only ratings plus situational rest, trained on earlier development games.
def fit_ridge(feats,target,alpha):
    return make_pipeline(StandardScaler(),Ridge(alpha=alpha)).fit(tr[feats],tr[target])
m4margin=fit_ridge(["elo_diff","rest_diff"],"actual_margin",25)
m4margin_norest=fit_ridge(["elo_diff"],"actual_margin",25)
m4total=fit_ridge(["h_points_pg","a_points_pg","h_allowed_pg","a_allowed_pg","rest_diff"],"actual_total",120)
te["m4_margin"]=m4margin.predict(te[["elo_diff","rest_diff"]])
te["m4_total"]=m4total.predict(te[["h_points_pg","a_points_pg","h_allowed_pg","a_allowed_pg","rest_diff"]])
te["m4_no_rest"]=m4margin_norest.predict(te[["elo_diff"]])
# M5 nonlinear boosted tree independently fits home and away scores from raw priors.
m5h=HistGradientBoostingRegressor(max_iter=120,max_depth=2,min_samples_leaf=35,learning_rate=.05,l2_regularization=5,random_state=3601).fit(tr[cols_tree],tr.home_score)
m5a=HistGradientBoostingRegressor(max_iter=120,max_depth=2,min_samples_leaf=35,learning_rate=.05,l2_regularization=5,random_state=3602).fit(tr[cols_tree],tr.away_score)
te["m5_home"]=m5h.predict(te[cols_tree]);te["m5_away"]=m5a.predict(te[cols_tree])
# Four Factors 1.0 fixed factor relative importance as an INDEX (not a scoring equation);
# factor proportions must not naively replace learned point values.
ff=["diff_efg","diff_tov","diff_orb","diff_ftr"]
std=StandardScaler().fit(tr[ff])
z_tr=std.transform(tr[ff]);z_te=std.transform(te[ff])
w=np.array([.4,.25,.2,.15])
ff1=LinearRegression().fit((z_tr*w).sum(axis=1).reshape(-1,1),tr.actual_margin)
te["ff1_margin"]=ff1.predict((z_te*w).sum(axis=1).reshape(-1,1))
ff2=fit_ridge(ff,"actual_margin",80)
te["ff2_margin"]=ff2.predict(te[ff])
ff3cols=ff+["diff_three_share","diff_three_pct","diff_two_pct","diff_ft_pct",
            "prior_strength_gap","prior_pace_gap"]
ff3=fit_ridge(ff3cols,"actual_margin",160)
te["ff3_margin"]=ff3.predict(te[ff3cols])
# M3 simulated points; no false label that the Four Factor weights are causally established.
for model in ["m1","m3","m5"]:
    te[model+"_margin"]=te[model+"_home"]-te[model+"_away"]
    te[model+"_total"]=te[model+"_home"]+te[model+"_away"]
te["m4_home"]=(te.m4_total+te.m4_margin)/2;te["m4_away"]=(te.m4_total-te.m4_margin)/2
for k in ["m1","m3","m4","m5"]:
    if not (np.abs((te[k+"_home"]-te[k+"_away"])-te[k+"_margin"])<1e-6).all():raise ValueError("score margin inconsistency "+k)
    if not (np.abs((te[k+"_home"]+te[k+"_away"])-te[k+"_total"])<1e-6).all():raise ValueError("score total inconsistency "+k)
def metrics(pred,actual):
    a=np.asarray(actual,dtype=float);p=np.asarray(pred,dtype=float)
    return float(np.mean(abs(p-a))),float(np.mean(p-a)),float(np.sqrt(np.mean((p-a)**2)))
metrics_list=[]
for k in ["m1","m3","m4","m5"]:
    mg,mb,mrm=metrics(te[k+"_margin"],te.actual_margin)
    tg,tb,trm=metrics(te[k+"_total"],te.actual_total)
    hs,_,_=metrics(te[k+"_home"],te.home_score)
    az,_,_=metrics(te[k+"_away"],te.away_score)
    pg,pb,_=metrics(te["m1_pace"] if k in ["m1","m4","m5"] else te["m3_pace"],te.actual_pace)
    metrics_list.append(dict(model=k,experiment="MULTIMODEL_PRESEASON_EARLY_SCREEN",games=len(te),margin_mae=mg,margin_bias=mb,margin_rmse=mrm,total_mae=tg,total_bias=tb,total_rmse=trm,team_score_mae=(hs+az)/2,pace_mae=pg,pace_bias=pb,winner_accuracy=float(np.mean(np.sign(te[k+"_margin"])==np.sign(te.actual_margin)))))
# Roster model cannot produce pregame-safe point predictions.
metrics_list.insert(1,dict(model="m2",status="BLOCKED_NO_CERTIFIED_HISTORICAL_ASOF_ROSTER",games=0))
ff_metrics=[dict(version=v,n=len(te),margin_mae=metrics(te[v+"_margin"],te.actual_margin)[0],bias=metrics(te[v+"_margin"],te.actual_margin)[1]) for v in ["ff1","ff2","ff3"]]
rest_metrics={"with_rest_mae":metrics(te.m4_margin,te.actual_margin)[0],
              "without_rest_mae":metrics(te.m4_no_rest,te.actual_margin)[0],
              "effect_on_margin_mae":metrics(te.m4_no_rest,te.actual_margin)[0]-metrics(te.m4_margin,te.actual_margin)[0]}
# paired date-block bootstrap, historical and exploratory
dates=te.game_date.dt.strftime("%Y-%m-%d").to_numpy();groups=[np.where(dates==d)[0] for d in np.unique(dates)]
rng=np.random.default_rng(3611)
ci={}
base=np.abs((te.m1_margin-te.actual_margin).to_numpy())
for k in ["m3","m4","m5"]:
    delta=base-np.abs((te[k+"_margin"]-te.actual_margin).to_numpy())
    vals=np.array([np.concatenate([delta[groups[i]] for i in rng.integers(0,len(groups),len(groups))]).mean() for _ in range(1800)])
    ci[k]=dict(gain=float(delta.mean()),day_block_ci=np.quantile(vals,[.025,.975]).tolist())
# cross-model discordance and errors; descriptive only
preds=np.stack([te[k+"_margin"].to_numpy() for k in ["m1","m3","m4","m5"]],axis=1)
disc={"mean_margin_range":float(np.mean(np.ptp(preds,axis=1))),
      "p90_margin_range":float(np.quantile(np.ptp(preds,axis=1),.90)),
      "winner_disagreement_fraction":float(np.mean(np.any(preds>0,axis=1)&np.any(preds<0,axis=1))),
      "pairwise_pred_corr":pd.DataFrame(preds,columns=["m1","m3","m4","m5"]).corr().round(4).to_dict(),
      "pairwise_error_corr":pd.DataFrame(preds-te.actual_margin.to_numpy()[:,None],columns=["m1","m3","m4","m5"]).corr().round(4).to_dict()}
outcols=["game_id","game_date","home_id","away_id","home_name","away_name","home_score","away_score","actual_margin","actual_total","actual_pace","rest_diff"]
for k in ["m1","m3","m4","m5"]:outcols +=[k+"_home",k+"_away",k+"_margin",k+"_total"]
outcols+=["ff1_margin","ff2_margin","ff3_margin","m4_no_rest"]
tofile=te[outcols].sort_values(["game_date","game_id"])
tofile.to_csv(OUT/"matched_games_2025_26.csv",index=False)
result={"status":"HISTORICAL_EXPLORATORY_REUSED_2025_26_SCREEN",
        "training_games":len(tr),"test_games":len(te),"test_dates":len(groups),
        "source_sha256":{str(y):hashlib.sha256((ROOT/f"eb{y}.parquet").read_bytes()).hexdigest() for y in YEARS},
        "model_metrics":metrics_list,"four_factors_ablation":ff_metrics,
        "situational_rest_ablation":rest_metrics,
        "disagreement_four_executable":disc,"paired_margin_mae_gain_vs_m1":ci,
        "limitations":["2025-26 reused historically; no untouched holdout","D1 season eligibility unverified",
                       "ESPN home label not independently neutral-site certified",
                       "some OT games not isolated, source box possessions P475 approximate",
                       "M4 uses only previous-date same-season results, no current-date updates",
                       "M2 pregame roster date source blocked","M5 tree has 2024-25 early-season development only",
                       "M1 is not exact sealed v14 champion; reconstructed lightweight comparator",
                       "Models share input source and are not statistically independent",
                       "No calibrated win probability and no gambling ROI established"]}
(OUT/"metrics.json").write_text(json.dumps(result,indent=2,default=str))
print("COHORT",len(tr),len(te),"days",len(groups),flush=True)
print("METRICS",json.dumps(metrics_list),flush=True)
print("FF",json.dumps(ff_metrics),"REST",json.dumps(rest_metrics),flush=True)
print("DIVERSITY",json.dumps({k:v for k,v in disc.items() if not k.startswith("pairwise")}),flush=True)
print("CI",json.dumps(ci),flush=True)
