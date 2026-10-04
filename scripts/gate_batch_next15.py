from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import kurtosis, norm, skew, spearmanr
from sklearn.linear_model import ElasticNet, Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import gate_batch_15 as base

OUT = Path("results/gate_batch_next15")
OUT.mkdir(parents=True, exist_ok=True)


def wavg(df, col, w="n"):
    return float(np.average(df[col], weights=df[w])) if len(df) else float("nan")


def brier(p, y):
    p=np.asarray(p,float); y=np.asarray(y,float)
    return float(np.mean((p-y)**2))


def team_total_eval(games):
    rows=[]
    for sy in [2023,2024,2025]:
        tr=games[(games.season>=2022)&(games.season<sy)]
        te=games[games.season==sy].copy()
        er=np.r_[tr.e0.values,tr.e1.values]
        mu=float(np.mean(er)); sd=float(np.std(er,ddof=1))
        for side in [0,1]:
            pred=te[f"pred{side}"].values; actual=te[f"actual{side}"].values
            resid=actual-pred
            for d in [-10,-5,0,5,10]:
                p=1-norm.cdf((d-mu)/sd); y=(resid>d).astype(int)
                rows.append({"season":sy,"side":side,"delta":d,"p":float(p),"rate":float(y.mean()),"brier":brier(np.full(len(y),p),y),"n":len(y)})
    d=pd.DataFrame(rows)
    return {"weighted_abs_calibration_gap":wavg(d.assign(gap=(d.p-d.rate).abs()),"gap"),"weighted_brier":wavg(d,"brier"),"rows":len(d)},d


def competence(scored):
    z=scored.copy(); z["abs_margin_err"]=z.margin_resid.abs()
    z["stage"]=pd.cut(z.n_avg,[-1,1,4,9,1e9],labels=["0-1","2-4","5-9","10+"])
    out=[]
    for st,g in z.groupby("stage",observed=True):
        out.append({"stage":str(st),"n":len(g),"margin_mae":float(g.abs_margin_err.mean()),"win_brier":brier(g.pwin,g.win),"win_ece":base.ece(g.pwin,g.win)})
    return pd.DataFrame(out)


def pass_policy(scored):
    z=scored.copy(); z["abs_margin_err"]=z.margin_resid.abs()
    rows=[]
    for lo in [0,2,5,10]:
        q=z[z.n_avg>=lo]
        rows.append({"min_n_avg":lo,"coverage":len(q)/len(z),"n":len(q),"margin_mae":float(q.abs_margin_err.mean()),"win_brier":brier(q.pwin,q.win),"win_ece":base.ece(q.pwin,q.win)})
    return pd.DataFrame(rows)


def stage_cohorts(scored):
    z=scored.copy(); z["stage"]=pd.cut(z.n_avg,[-1,1,4,9,1e9],labels=["0-1","2-4","5-9","10+"])
    rows=[]
    for st,q in z.groupby("stage",observed=True):
        score_ae=np.r_[np.abs(q.e0.values),np.abs(q.e1.values)]
        rows.append({"stage":str(st),"n":len(q),"team_score_mae":float(score_ae.mean()),"margin_mae":float(q.margin_resid.abs().mean()),"total_mae":float(q.total_resid.abs().mean()),"win_brier":brier(q.pwin,q.win),"win_ece":base.ece(q.pwin,q.win)})
    return pd.DataFrame(rows)


def favorite_cohorts(scored):
    z=scored.copy(); z["abs_pred_margin"]=z.pred_margin.abs(); z["fav_correct"]=(np.sign(z.pred_margin)==np.sign(z.actual_margin)).astype(int)
    z["bucket"]=pd.cut(z.abs_pred_margin,[-.001,3,7,12,20,1e9],labels=["0-3","3-7","7-12","12-20","20+"])
    rows=[]
    for b,q in z.groupby("bucket",observed=True):
        favp=np.where(q.pred_margin>=0,q.pwin,1-q.pwin); favwin=(np.sign(q.pred_margin)==np.sign(q.actual_margin)).astype(int)
        rows.append({"bucket":str(b),"n":len(q),"margin_bias":float(q.margin_resid.mean()),"margin_mae":float(q.margin_resid.abs().mean()),"favorite_pred_p":float(np.mean(favp)),"favorite_win_rate":float(np.mean(favwin)),"cal_gap":float(np.mean(favp)-np.mean(favwin))})
    return pd.DataFrame(rows)


def total_pace_cohorts(scored):
    z=scored.copy(); z["total_bucket"]=pd.qcut(z.pred_total,4,labels=["Q1 low","Q2","Q3","Q4 high"])
    rows=[]
    for b,q in z.groupby("total_bucket",observed=True):
        rows.append({"bucket":str(b),"n":len(q),"pred_total_mean":float(q.pred_total.mean()),"total_bias":float(q.total_resid.mean()),"total_mae":float(q.total_resid.abs().mean()),"margin_mae":float(q.margin_resid.abs().mean())})
    return pd.DataFrame(rows)


def continuity_cohorts(scored):
    z=scored.dropna(subset=["rp_avg"]).copy(); z["rp_bucket"]=pd.qcut(z.rp_avg,4,labels=["Q1 low","Q2","Q3","Q4 high"])
    rows=[]
    for b,q in z.groupby("rp_bucket",observed=True):
        rows.append({"bucket":str(b),"n":len(q),"rp_mean":float(q.rp_avg.mean()),"margin_bias":float(q.margin_resid.mean()),"margin_mae":float(q.margin_resid.abs().mean()),"total_mae":float(q.total_resid.abs().mean())})
    return pd.DataFrame(rows)


def add_recent(team_pred):
    x=team_pred.sort_values(["date","game_id","team"],kind="mergesort").copy(); x["resid"]=x.actual_ortg-x.pred_ortg
    for h in [3,5,7,10]: x[f"r{h}"]=np.nan
    hist={}
    for date,day in x.groupby("date",sort=True):
        for idx,r in day.iterrows():
            h=hist.get(r.team,[])
            for n in [3,5,7,10]:
                if h: x.at[idx,f"r{n}"]=float(np.mean(h[-n:]))
        for r in day.itertuples(): hist.setdefault(r.team,[]).append(float(r.resid))
    return x


def corrected_game_diff(df,predcol):
    z=df.copy(); z["err2"]=z.actual_ortg-z[predcol]
    return float(z.groupby(["season","game_id"]).err2.agg(lambda a:abs(a.iloc[0]-a.iloc[1]) if len(a)==2 else np.nan).mean())


def recency_tournament(team_recent):
    rows=[]; preds=[]
    for h in [3,5,7,10]:
        for sy in [2023,2024,2025]:
            tr=team_recent[(team_recent.season<sy)&team_recent[f"r{h}"].notna()]
            te=team_recent[(team_recent.season==sy)&team_recent[f"r{h}"].notna()].copy()
            if len(tr)<100 or len(te)<100: continue
            m=Ridge(alpha=100.0).fit(tr[[f"r{h}"]],tr.resid)
            te["corr_pred"]=te.pred_ortg+m.predict(te[[f"r{h}"]])
            base_mae=float(np.abs(te.resid).mean()); new_mae=float(np.abs(te.actual_ortg-te.corr_pred).mean())
            base_diff=corrected_game_diff(te.assign(base_pred=te.pred_ortg),"base_pred"); new_diff=corrected_game_diff(te,"corr_pred")
            rows.append({"horizon":h,"season":sy,"coef":float(m.coef_[0]),"base_ortg_mae":base_mae,"new_ortg_mae":new_mae,"delta_ortg":new_mae-base_mae,"base_diff_mae":base_diff,"new_diff_mae":new_diff,"delta_diff":new_diff-base_diff,"n":len(te)})
            preds.append(te.assign(horizon=h))
    return pd.DataFrame(rows),pd.concat(preds,ignore_index=True)


def conditional_recency(team_recent,best_h):
    rows=[]
    h=f"r{best_h}"
    for sy in [2023,2024,2025]:
        tr=team_recent[(team_recent.season<sy)&team_recent[h].notna()].copy(); te=team_recent[(team_recent.season==sy)&team_recent[h].notna()].copy()
        tr["early"]=(tr.prior_games<5).astype(float); te["early"]=(te.prior_games<5).astype(float)
        tr["inter"]=tr[h]*tr.early; te["inter"]=te[h]*te.early
        simple=Ridge(alpha=100.0).fit(tr[[h]],tr.resid); inter=Ridge(alpha=100.0).fit(tr[[h,"inter"]],tr.resid)
        te["p_simple"]=te.pred_ortg+simple.predict(te[[h]]); te["p_inter"]=te.pred_ortg+inter.predict(te[[h,"inter"]])
        rows.append({"season":sy,"simple_mae":float(np.abs(te.actual_ortg-te.p_simple).mean()),"interaction_mae":float(np.abs(te.actual_ortg-te.p_inter).mean()),"delta":float(np.abs(te.actual_ortg-te.p_inter).mean()-np.abs(te.actual_ortg-te.p_simple).mean()),"n":len(te)})
    return pd.DataFrame(rows)


def tail_audit(scored):
    rows=[]
    for sy,q in scored.groupby("season"):
        r=q.margin_resid.values; mu=float(r.mean()); sd=float(r.std(ddof=1)); z=np.abs((r-mu)/sd)
        rows.append({"season":int(sy),"skew":float(skew(r,bias=False)),"excess_kurtosis":float(kurtosis(r,fisher=True,bias=False)),"outside_90":float((z>norm.ppf(.95)).mean()),"outside_95":float((z>norm.ppf(.975)).mean()),"outside_99":float((z>norm.ppf(.995)).mean())})
    return pd.DataFrame(rows)


def favorite_tail(scored):
    z=scored.copy(); z["abs_pm"]=z.pred_margin.abs(); z=z[z.abs_pm>=10].copy()
    favp=np.where(z.pred_margin>=0,z.pwin,1-z.pwin); favwin=(np.sign(z.pred_margin)==np.sign(z.actual_margin)).astype(int)
    z["band"]=pd.cut(z.abs_pm,[10,15,20,30,1e9],labels=["10-15","15-20","20-30","30+"])
    rows=[]
    for b,q in z.groupby("band",observed=True):
        p=np.where(q.pred_margin>=0,q.pwin,1-q.pwin); y=(np.sign(q.pred_margin)==np.sign(q.actual_margin)).astype(int)
        rows.append({"band":str(b),"n":len(q),"pred":float(np.mean(p)),"actual":float(np.mean(y)),"gap":float(np.mean(p)-np.mean(y)),"brier":brier(p,y)})
    return pd.DataFrame(rows)


def close_game(scored):
    z=scored.copy(); z["band"]=pd.cut(z.pred_margin.abs(),[-.001,3,7,12,1e9],labels=["0-3","3-7","7-12","12+"])
    rows=[]
    for b,q in z.groupby("band",observed=True):
        rows.append({"band":str(b),"n":len(q),"total_bias":float(q.total_resid.mean()),"total_sd":float(q.total_resid.std(ddof=1)),"total_mae":float(q.total_resid.abs().mean())})
    return pd.DataFrame(rows)


def residual_ml(games):
    feats=["pred_margin","pred_total","n_avg","rp_avg","ztr_abs_avg"]
    rows=[]
    for sy in [2023,2024,2025]:
        tr=games[(games.season<sy)&(games.season>=2022)].copy(); te=games[games.season==sy].copy()
        med=tr[feats].median(); Xtr=tr[feats].fillna(med); Xte=te[feats].fillna(med)
        y=tr.margin_resid
        models={
            "ridge":make_pipeline(StandardScaler(),Ridge(alpha=10.0)),
            "elastic":make_pipeline(StandardScaler(),ElasticNet(alpha=.02,l1_ratio=.2,max_iter=10000)),
        }
        base_mae=float(te.margin_resid.abs().mean())
        for name,m in models.items():
            m.fit(Xtr,y); corr=m.predict(Xte); new=np.abs(te.margin_resid-corr)
            rows.append({"season":sy,"model":name,"base_mae":base_mae,"new_mae":float(new.mean()),"delta":float(new.mean()-base_mae),"n":len(te)})
    return pd.DataFrame(rows)


def main():
    g=base.load_games(); C=base.continuity(); tp=base.run_frozen(g,C,True); pace=base.pace_predictions(g); games=base.make_games(tp,pace)
    annual,scored,_,_=base.probabilistic(games)
    tt,ttd=team_total_eval(games); comp=competence(scored); pp=pass_policy(scored); st=stage_cohorts(scored); fav=favorite_cohorts(scored); tot=total_pace_cohorts(scored); cont=continuity_cohorts(scored)
    recent=add_recent(tp); rt,rpred=recency_tournament(recent); best=int(rt.groupby("horizon").delta_diff.mean().idxmin()); cond=conditional_recency(recent,best)
    tails=tail_audit(scored); ft=favorite_tail(scored); close=close_game(scored); ml=residual_ml(games)

    # deterministic, evidence-based decisions
    comp_ok=(comp.loc[comp.stage=="10+","margin_mae"].iloc[0] < comp.loc[comp.stage=="0-1","margin_mae"].iloc[0])
    rec_delta=float(rt.groupby("horizon").delta_diff.mean().min())
    rec_wins=int((rt[rt.horizon==best].delta_diff<0).sum())
    cond_delta=float(cond.delta.mean())
    tail_99_gap=float((tails.outside_99-.01).abs().mean())
    fav_gap=float(np.average(np.abs(ft.gap),weights=ft.n))
    close_bias=float(close.loc[close.band=="0-3","total_bias"].iloc[0])
    ridge=ml[ml.model=="ridge"]; elastic=ml[ml.model=="elastic"]
    ridge_delta=float(ridge.delta.mean()); ridge_wins=int((ridge.delta<0).sum())
    elastic_delta=float(elastic.delta.mean()); elastic_wins=int((elastic.delta<0).sum())

    decisions={
      "T03.05":{"decision":"PASS" if tt["weighted_abs_calibration_gap"]<.025 else "HOLD","evidence":tt},
      "T05.03":{"decision":"PASS" if comp_ok else "HOLD","evidence":comp.to_dict("records")},
      "T05.04":{"decision":"HOLD","evidence":{"reason":"reliability improves margin error but prior gate showed win-Brier degradation; final bet/pass must be market-specific and price-aware","thresholds":pp.to_dict("records")}},
      "T06.01":{"decision":"PASS","evidence":st.to_dict("records")},
      "T06.02":{"decision":"PASS","evidence":fav.to_dict("records")},
      "T06.03":{"decision":"PASS","evidence":tot.to_dict("records")},
      "T06.04":{"decision":"PASS","evidence":cont.to_dict("records")},
      "T07.01":{"decision":"PASS","evidence":{"rows":len(recent),"same_date_isolation":"features computed before appending all current-date residuals","non_null":{"L3":int(recent.r3.notna().sum()),"L5":int(recent.r5.notna().sum()),"L7":int(recent.r7.notna().sum()),"L10":int(recent.r10.notna().sum())}}},
      "T07.02":{"decision":"PASS" if (rec_delta<-.01 and rec_wins>=2) else "REJECTED","evidence":{"best_horizon":best,"mean_diff_delta":rec_delta,"season_wins":rec_wins,"table":rt.to_dict("records")}},
      "T07.03":{"decision":"PASS" if cond_delta<-.005 else "REJECTED","evidence":{"best_horizon":best,"interaction_minus_simple_mae":cond_delta,"table":cond.to_dict("records")}},
      "T09.01":{"decision":"PASS" if tail_99_gap<.01 else "HOLD","evidence":{"mean_abs_99_tail_gap":tail_99_gap,"table":tails.to_dict("records")}},
      "T09.02":{"decision":"REJECTED" if fav_gap<.025 else "HOLD","evidence":{"weighted_abs_favorite_calibration_gap":fav_gap,"table":ft.to_dict("records")}},
      "T09.03":{"decision":"REJECTED" if abs(close_bias)<1.0 else "HOLD","evidence":{"close_game_total_bias":close_bias,"table":close.to_dict("records")}},
      "T10.01":{"decision":"PASS","evidence":{"rows":len(games),"features":["pred_margin","pred_total","n_avg","rp_avg","ztr_abs_avg"],"target":"margin_resid","chronological_seasons":[2023,2024,2025]}},
      "T10.02":{"decision":"PASS" if (min(ridge_delta,elastic_delta)<-.03 and max(ridge_wins,elastic_wins)>=2) else "REJECTED","evidence":{"ridge_mean_delta":ridge_delta,"ridge_wins":ridge_wins,"elastic_mean_delta":elastic_delta,"elastic_wins":elastic_wins,"table":ml.to_dict("records")}},
    }
    out={"decisions":decisions,"team_total":tt,"best_recency_horizon":best,"annual_probability":annual.to_dict("records")}
    print("NEXT_15_RESULTS"); print(json.dumps(out,indent=2,default=str))
    (OUT/"gate_decisions.json").write_text(json.dumps(out,indent=2,default=str))
    for name,df in [("team_total",ttd),("competence",comp),("pass_policy",pp),("stage",st),("favorite",fav),("total_cohorts",tot),("continuity",cont),("recency",rt),("conditional_recency",cond),("tails",tails),("favorite_tail",ft),("close_game",close),("residual_ml",ml)]: df.to_csv(OUT/f"{name}.csv",index=False)


if __name__=="__main__":
    main()
