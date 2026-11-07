#!/usr/bin/env python3
"""Independent hypothesis tests for CBB Research OS. Historical exploratory only.

Modes: favorite, pace, factors, players. Runs on distinct independent processes.
No betting lines, market inputs, or sealed 2025-26 outcomes in model training.
Basketball team-box sources: 2022-23, 2023-24, 2024-25.
Team model protocol: 2023-24 14-day dev / 2024-25 14-day historical screen.
Player model: 2022-23->2023-24 dev / 2023-24->2024-25 screen.
The source mix is not season-D1 or OT certified; never promote from this run.
"""
import argparse, hashlib, json, warnings
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from pandas.errors import PerformanceWarning

warnings.filterwarnings("ignore", category=PerformanceWarning)
warnings.filterwarnings("ignore", category=FutureWarning)
ROOT=Path("/tmp/cbb")
DIR=ROOT/"parallel_hypothesis_lab"
DIR.mkdir(exist_ok=True)
SEED=20261106
def safe(a,b): return np.asarray(a,dtype=float)/np.maximum(np.asarray(b,dtype=float),1e-9)
def sha(path):return hashlib.sha256(Path(path).read_bytes()).hexdigest()
def finite(df,columns):
    return df.replace([np.inf,-np.inf],np.nan).dropna(subset=columns).copy()
def mae(a,b):return float(np.mean(np.abs(np.asarray(a,dtype=float)-np.asarray(b,dtype=float))))
def date_ci(df,delta,reps=1200):
    date=pd.to_datetime(df.game_date).dt.strftime("%Y-%m-%d").to_numpy()
    groups=[np.flatnonzero(date==d) for d in sorted(set(date))]
    rng=np.random.default_rng(SEED)
    delta=np.asarray(delta,dtype=float)
    res=np.array([np.concatenate([delta[groups[i]] for i in rng.integers(len(groups),size=len(groups))]).mean() for _ in range(reps)])
    return dict(gain=float(delta.mean()),ci95=np.quantile(res,[.025,.975]).round(5).tolist(),date_clusters=len(groups))
def save(track,payload,df):
    payload["track"]=track
    payload["status"]="HISTORICAL_EXPLORATORY_REUSED_NO_PRODUCTION"
    payload["no_market_data"]=True
    payload["created_by"]="scripts/cbb_parallel_hypothesis_lab_20261106.py"
    out=DIR/(track+".json")
    out.write_text(json.dumps(payload,indent=2,default=str),encoding="utf-8")
    if df is not None:df.to_csv(DIR/(track+"_game_level.csv"),index=False,float_format="%.5f")
    print("RESULT",track,json.dumps({k:v for k,v in payload.items() if k not in ["rows","source_hashes","diagnostics"]},default=str),flush=True)
    print("SAVED",out,flush=True)
def frames():
    helper=Path("/tmp/cbb_multimodel_fourfactors_controlled_20261106.py")
    if not helper.exists():
        raise RuntimeError("Source helper must be staged from pinned project repo")
    code=helper.read_text()
    split="\nbox={y:getbox(y) for y in YEARS}"
    if split not in code:raise RuntimeError("Unknown helper version; abort to protect feature PIT")
    safe_prefix=code.split(split)[0]
    scope={}
    exec(compile(safe_prefix,str(helper),"exec"),scope)
    box={y:scope["getbox"](y) for y in (2023,2024,2025)}
    tr=scope["build"](2024,box[2024],box[2023])
    te=scope["build"](2025,box[2025],box[2024])
    assert all(k in tr.columns and k in te.columns for k in ["p_h_off","p_a_allow_off","p_h_pace","p_a_pace","baseline_margin","actual_total"])
    assert all(pd.to_datetime(tr.game_date)<pd.Timestamp("2024-01-01"))
    assert all(pd.to_datetime(te.game_date)>pd.Timestamp("2024-01-01"))
    meta={"team_source_hashes":{str(y):sha(ROOT/f"eb{y}.parquet") for y in box},
        "helper_sha256":sha(helper),"train_year":"2023-24","screen_year":"2024-25",
        "PIT":"previous season team-box features only, no target-game box in feature columns",
        "source_gates":"Team D1 identity, neutral, OT period lengths and historic publication not independently certified"}
    return tr,te,box,meta
def ridge_residual_fit(train,test,feats,target,baseline,alpha=120):
    for df in [train,test]:
        if not np.isfinite(df[feats+[baseline,target]].to_numpy(dtype=float)).all():
            raise RuntimeError("Non-finite model features")
    m=make_pipeline(StandardScaler(),Ridge(alpha=alpha))
    m.fit(train[feats],train[target]-train[baseline])
    return test[baseline].to_numpy()+m.predict(test[feats])
def large_fav():
    tr,te,box,meta=frames()
    for d in [tr,te]:
        d["pred_home"]=(d.baseline_total+d.baseline_margin)/2
        d["pred_away"]=(d.baseline_total-d.baseline_margin)/2
        d["is_home_favorite"]=d.baseline_margin>=0
        d["fav_sign"]=np.where(d.is_home_favorite,1.,-1.)
        d["fav_pred"]=np.where(d.is_home_favorite,d.pred_home,d.pred_away)
        d["dog_pred"]=np.where(d.is_home_favorite,d.pred_away,d.pred_home)
        d["fav_actual"]=np.where(d.is_home_favorite,d.home_score,d.away_score)
        d["dog_actual"]=np.where(d.is_home_favorite,d.away_score,d.home_score)
        d["fav_prior_off"]=np.where(d.is_home_favorite,d.p_h_off,d.p_a_off)
        d["dog_prior_def"]=np.where(d.is_home_favorite,d.p_a_allow_off,d.p_h_allow_off)
        d["gap"]=np.abs(d.baseline_margin)
        d["off_dev"]=d.fav_prior_off-100
        d["def_dev"]=d.dog_prior_def-100
        d["interaction"]=d.off_dev*d.def_dev/100
        d["favorite_residual"]=d.fav_actual-d.fav_pred
        d["underdog_residual"]=d.dog_actual-d.dog_pred
    tr=finite(tr,["off_dev","def_dev","gap","interaction","fav_pred","dog_pred","fav_actual","dog_actual"])
    te=finite(te,["off_dev","def_dev","gap","interaction","fav_pred","dog_pred","fav_actual","dog_actual"])
    fav_offset=float(tr.favorite_residual.mean())
    dog_offset=float(tr.underdog_residual.mean())
    variants={"ORIGINAL":[],"TWO_SIDE_OFFSET":[],
              "FAV_OFF_ONLY":["off_dev"],"WEAK_DEF_ONLY":["def_dev"],
              "STRENGTH_GAP_ONLY":["gap"],"OFF_DEF_JOINT":["off_dev","def_dev","interaction"]}
    result=[];preds=te[["game_id","game_date","home_id","away_id","actual_margin","actual_total","home_score","away_score","fav_actual","dog_actual","gap","off_dev","def_dev"]].copy()
    for name,feats in variants.items():
        if name=="ORIGINAL":
            fav=te.fav_pred.to_numpy();dog=te.dog_pred.to_numpy()
        elif name=="TWO_SIDE_OFFSET":
            fav=te.fav_pred.to_numpy()+fav_offset;dog=te.dog_pred.to_numpy()+dog_offset
        else:
            m=make_pipeline(StandardScaler(),Ridge(alpha=60))
            m.fit(tr[feats],tr.favorite_residual)
            fav=te.fav_pred.to_numpy()+m.predict(te[feats])
            dog=te.dog_pred.to_numpy()+dog_offset
        home=np.where(te.is_home_favorite,fav,dog)
        away=np.where(te.is_home_favorite,dog,fav)
        mrg=home-away;total=home+away
        preds[name+"_margin"]=mrg;preds[name+"_total"]=total
        for cohort,mask in [("ALL",np.ones(len(te),bool)),("GAP_0_10",te.gap<10),
                            ("GAP_10_20",(te.gap>=10)&(te.gap<20)),("GAP_20_PLUS",te.gap>=20)]:
            if not mask.any():continue
            result.append(dict(model=name,cohort=cohort,n=int(mask.sum()),
                margin_mae=mae(mrg[mask],te.actual_margin.to_numpy()[mask]),
                total_mae=mae(total[mask],te.actual_total.to_numpy()[mask]),
                favorite_points_mae=mae(fav[mask],te.fav_actual.to_numpy()[mask]),
                underdog_points_mae=mae(dog[mask],te.dog_actual.to_numpy()[mask]),
                favorite_pred_minus_actual=float(np.mean(fav[mask]-te.fav_actual.to_numpy()[mask])),
                underdog_pred_minus_actual=float(np.mean(dog[mask]-te.dog_actual.to_numpy()[mask]))))
    baseline=np.abs(preds.ORIGINAL_margin-te.actual_margin.to_numpy())
    cis={}
    for variant in ["TWO_SIDE_OFFSET","FAV_OFF_ONLY","WEAK_DEF_ONLY","STRENGTH_GAP_ONLY","OFF_DEF_JOINT"]:
        delta=baseline-np.abs(preds[variant+"_margin"]-te.actual_margin.to_numpy())
        cis[variant]=date_ci(te,delta)
    payload=dict(meta,train_n=len(tr),screen_n=len(te),models=result,paired_gain_vs_original=cis,
       train_favorite_offset=fav_offset,train_underdog_offset=dog_offset,
       off_def_corr=float(te.off_dev.corr(te.def_dev)),
       limitation="Fav offense and opponent defense terms co-vary; observational coefficients do NOT identify causal origin of favorite scoring bias")
    save("favorite",payload,preds)
def pace():
    tr,te,box,meta=frames()
    def contextual(d,prev):
        prior=prev.groupby("team_id").poss.mean().to_dict()
        temp=prev.assign(opp_pace=prev.opponent_team_id.map(prior))
        oppavg=temp.groupby("team_id").opp_pace.mean().to_dict()
        league=float(prev.poss.mean())
        for side in ("h","a"):
            d[side+"_priorpace"]=d["p_"+side+"_pace"]
            d[side+"_oppavg"]=d[side+"_id" if side+"_id" in d.columns else "home_id" if side=="h" else "away_id"].map(oppavg)
            d[side+"_pace_adjp"]=d[side+"_priorpace"]-.5*(d[side+"_oppavg"]-league)
        d["paced_avg"]=(d.h_priorpace+d.a_priorpace)/2
        d["paced_oppadj"]=(d.h_pace_adjp+d.a_pace_adjp)/2
        d["pace_gap"]=abs(d.h_priorpace-d.a_priorpace)
        d["strength_gap"]=abs(d.baseline_margin)
        d["actual_pace"]=(d.h_poss+d.a_poss)/2
    contextual(tr,box[2023]);contextual(te,box[2024])
    required=["paced_avg","paced_oppadj","pace_gap","strength_gap","actual_pace","baseline_total","pace_prior"]
    tr=finite(tr,required);te=finite(te,required)
    offset=float((tr.actual_pace-tr.paced_avg).mean())
    tr["pace_target_resid"]=tr.actual_pace-tr.paced_avg
    variants={"EQUAL_MEAN":te.paced_avg.to_numpy(),
       "TRAINED_OFFSET":te.paced_avg.to_numpy()+offset,
       "OPP_ADJUSTED":te.paced_oppadj.to_numpy()+offset}
    for name,feat in [("GAP_AND_OFFSET",["pace_gap"]),("STRENGTH_AND_OFFSET",["strength_gap"]),
                      ("PACE_STRENGTH_INTERACTION",["pace_gap","strength_gap"])]:
        model=make_pipeline(StandardScaler(),Ridge(alpha=100))
        model.fit(tr[feat],tr.pace_target_resid)
        variants[name]=te.paced_avg.to_numpy()+model.predict(te[feat])
    out=te[["game_id","game_date","home_id","away_id","baseline_margin","baseline_total","actual_total","actual_pace","pace_gap","strength_gap"]].copy()
    model_results=[];ci={}
    price=te["base_home_pp100"].to_numpy()+te["base_away_pp100"].to_numpy()
    bpace=te.paced_avg.to_numpy();y=te.actual_pace.to_numpy()
    bt=te.actual_total.to_numpy()
    base_abs_p=np.abs(bpace-y)
    base_abs_t=np.abs(price*bpace/100-bt)
    for name,p in variants.items():
        pred_total=price*p/100
        out[name+"_pace"]=p;out[name+"_total"]=pred_total
        d_p=base_abs_p-np.abs(p-y)
        d_t=base_abs_t-np.abs(pred_total-bt)
        model_results.append(dict(name=name,n=len(te),pace_mae=mae(p,y),pace_bias=float(np.mean(p-y)),
                                  total_mae=mae(pred_total,bt),total_bias=float(np.mean(pred_total-bt))))
        if name!="EQUAL_MEAN":
            ci[name]={"pace":date_ci(te,d_p),"total":date_ci(te,d_t)}
    payload=dict(meta,train_n=len(tr),screen_n=len(te),train_global_offset=offset,
         results=model_results,paired_gain_vs_equal=ci,
         note="P475 box possession. Source has overtime and possible non-D1; prior-season opponent adjustment is rough and not certified. Total uses fixed identical scoring PPP.")
    save("pace",payload,out)
def factors():
    tr,te,box,meta=frames()
    base=["baseline_margin","baseline_total","pace_prior","strength_diff","strength_sum"]
    groups={
      "FOUR_EFG":["efg"],"FOUR_TOV":["tov"],"FOUR_ORB":["orb"],"FOUR_FTR":["ftr"],
      "SHOOT_VOLUME":["three_share"],"SHOOT_ACCURACY":["two_pct","three_pct"],
      "FT_ACCURACY":["ft_pct"],"OPP_STEALS":["steals"],"OPP_FOULS":["fouls"],
      "COMPACT_FOUR":["efg","tov","orb","ftr"]}
    allcol=sorted(set(base+sum(([side+k for k in lst for side in ("diff_","sum_")] for lst in groups.values()),[])))
    tr=finite(tr,allcol+["actual_margin","actual_total"]);te=finite(te,allcol+["actual_margin","actual_total"])
    allresults=[];outs=te[["game_id","game_date","home_id","away_id","actual_margin","actual_total","baseline_margin","baseline_total"]].copy()
    for target in ("margin","total"):
        y=te["actual_"+target].to_numpy()
        baseparam="baseline_"+target
        per={}
        for name,gr in {"F0_BASE":[]}.items():
            per[name]=base
        for name,gr in groups.items():per[name]=base+[side+k for k in gr for side in ("diff_","sum_")]
        for name,features in per.items():
            pred=ridge_residual_fit(tr,te,features,"actual_"+target,baseparam)
            outs[name+"_"+target]=pred
            gain=None if name=="F0_BASE" else date_ci(te,np.abs(outs["F0_BASE_"+target]-y)-np.abs(pred-y))
            allresults.append({"target":target,"variant":name,"n":len(te),"factors":len(features),
                  "mae":mae(pred,y),"signed_bias":float(np.mean(pred-y)),
                  "paired_mae_gain_vs_F0":gain})
    payload=dict(meta,train_n=len(tr),screen_n=len(te),comparison=allresults,
      note="All challengers include same F0 structural baseline + individually added prior-only factor group. Match train games/alpha=120/temporal cuts. Multiplicity not corrected; no promotion.")
    save("factors",payload,outs)
def players():
    cols=["athlete_id","team_id","game_id","minutes","points","field_goals_attempted",
           "three_point_field_goals_made","three_point_field_goals_attempted",
           "free_throws_made","free_throws_attempted"]
    players={}
    for year in (2023,2024,2025):
        raw=pd.read_parquet(ROOT/f"ep{year}.parquet",columns=cols)
        raw["athlete_id"]=pd.to_numeric(raw.athlete_id,errors="coerce").astype("Int64").astype(str)
        raw["team_id"]=pd.to_numeric(raw.team_id,errors="coerce").astype("Int64").astype(str)
        raw=raw[~raw.athlete_id.isin(["<NA>","nan"])].copy()
        for col in cols[3:]:raw[col]=pd.to_numeric(raw[col],errors="coerce").fillna(0)
        d=raw.groupby("athlete_id")[cols[3:]].sum().reset_index()
        n=raw.groupby("athlete_id").game_id.nunique().rename("games").reset_index()
        t=(raw.groupby(["athlete_id","team_id"]).minutes.sum().reset_index()
          .sort_values(["athlete_id","minutes"],ascending=[True,False]).drop_duplicates("athlete_id"))
        d=d.merge(n,on="athlete_id",validate="one_to_one").merge(t[["athlete_id","team_id"]],on="athlete_id",validate="one_to_one")
        players[year]=d
    def cohort(prev,nxt):
        old=players[prev].copy(); new=players[nxt].copy()
        old=old[old.three_point_field_goals_attempted>=10]
        merged=old.merge(new,on="athlete_id",how="left",suffixes=("_prior","_next"),indicator=True,validate="one_to_one")
        eligible=(merged._merge=="both") & (merged.three_point_field_goals_attempted_next>=1)
        matched=merged.loc[eligible].copy()
        prior_rate=float(players[prev].three_point_field_goals_made.sum()/max(1,players[prev].three_point_field_goals_attempted.sum()))
        return matched,prior_rate,dict(prior_year=prev,next_year=nxt,prior_n=len(old),
          next_matched_n=int((merged._merge=="both").sum()),next_3pa_eligible_n=len(matched),
          missing_or_zero_shots_n=int(len(merged)-len(matched)))
    dev,dev_p,dev_meta=cohort(2023,2024)
    test,test_p,test_meta=cohort(2024,2025)
    def predict(k,d,p):
        return np.clip((d.three_point_field_goals_made_prior+k*p)/(d.three_point_field_goals_attempted_prior+k),.025,.975)
    def ll(d,pr):
        y=d.three_point_field_goals_made_next.to_numpy();n=d.three_point_field_goals_attempted_next.to_numpy()
        return -float(np.sum(y*np.log(pr)+(n-y)*np.log(1-pr))/np.sum(n))
    kgrid=[0,15,30,50,75,100,150,200,300]
    train_scores={str(k):ll(dev,predict(k,dev,dev_p)) for k in kgrid}
    selected=min(kgrid,key=lambda k:train_scores[str(k)])
    scores=[];results=test[["athlete_id","team_id_prior","team_id_next","games_prior","games_next","three_point_field_goals_attempted_prior","three_point_field_goals_made_prior","three_point_field_goals_attempted_next","three_point_field_goals_made_next","minutes_prior","minutes_next"]].copy()
    results["transfer_observed_next"]=results.team_id_prior!=results.team_id_next
    y=safe(test.three_point_field_goals_made_next,test.three_point_field_goals_attempted_next)
    for k in kgrid:
        pred=predict(k,test,test_p)
        results["prediction_k"+str(k)]=pred
        for group,mask in [("ALL",np.ones(len(test),bool)),
                           ("SAME_PROGRAM",(test.team_id_prior==test.team_id_next).to_numpy()),
                           ("TRANSFER_OBSERVED",(test.team_id_prior!=test.team_id_next).to_numpy())]:
            if not mask.any():continue
            d=test.loc[mask]
            p=pred[mask]
            scores.append({"k":k,"group":group,"n":int(mask.sum()),
                "next_shots":float(d.three_point_field_goals_attempted_next.sum()),
                "future_shot_logloss":ll(d,p),
                "next_3p_percent_mae":float(np.mean(abs(p-y[mask]))),
                "next_3p_percent_weighted_mae":float(np.average(abs(p-y[mask]),weights=d.three_point_field_goals_attempted_next))})
    baseline=predict(0,test,test_p);win=predict(selected,test,test_p)
    group=np.asarray(test.team_id_prior)
    ids=np.unique(group);rows=[np.flatnonzero(group==t) for t in ids]
    # Bootstrap player-school clusters for per-shot log-loss difference.
    e0=-(test.three_point_field_goals_made_next.to_numpy()*np.log(baseline)+(test.three_point_field_goals_attempted_next.to_numpy()-test.three_point_field_goals_made_next.to_numpy())*np.log(1-baseline))
    e1=-(test.three_point_field_goals_made_next.to_numpy()*np.log(win)+(test.three_point_field_goals_attempted_next.to_numpy()-test.three_point_field_goals_made_next.to_numpy())*np.log(1-win))
    weights=test.three_point_field_goals_attempted_next.to_numpy()
    rng=np.random.default_rng(SEED)
    boot=[]
    for _ in range(1400):
        chosen=np.concatenate([rows[i] for i in rng.integers(len(ids),size=len(ids))])
        boot.append(float(np.sum(e0[chosen]-e1[chosen])/np.sum(weights[chosen])))
    payload={"source_sha256":{str(y):sha(ROOT/f"ep{y}.parquet") for y in (2023,2024,2025)},
       "dev":dev_meta,"test":test_meta,"dev_prior_3p":dev_p,"test_prior_3p":test_p,
       "train_logloss":train_scores,"selected_train_only_k":selected,
       "test_logloss_gain_selected_vs_raw":ll(test,baseline)-ll(test,win),
       "test_logloss_gain_ci95_school_bootstrap":np.quantile(boot,[.025,.975]).round(6).tolist(),
       "results":scores,
       "selection_warning":"Test cohort includes only players with >=1 3PA next year; missing next-year players not scored. Transfer flag is retrospective and not pregame legal without roster as-of."}
    save("players",payload,results)
if __name__=="__main__":
    p=argparse.ArgumentParser()
    p.add_argument("--track",required=True,choices=["favorite","pace","factors","players"])
    args=p.parse_args()
    {"favorite":large_fav,"pace":pace,"factors":factors,"players":players}[args.track]()
