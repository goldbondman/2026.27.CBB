from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal, norm, spearmanr
from sklearn.linear_model import LinearRegression

RAW = "https://raw.githubusercontent.com/sportsdataverse/ncaa-mbb-hoops-data/main/"
SEASONS = range(2021, 2026)
OUT = Path("results/gate_batch_15")
OUT.mkdir(parents=True, exist_ok=True)


def pick(df, opts):
    c = next((x for x in opts if x in df.columns), None)
    if c is None:
        raise KeyError(f"missing one of {opts}")
    return c


def load_games():
    xs = []
    for sy in SEASONS:
        d = pd.read_parquet(RAW + f"mbb/team_box/parquet/ncaa_mbb_team_box_{sy}.parquet")
        d["season"] = sy
        xs.append(d)
    g = pd.concat(xs, ignore_index=True)
    s = pd.read_parquet(RAW + "mbb/ncaa_mbb_schedule_master.parquet")
    gid = pick(g, ["contest_id", "game_id"])
    tm = pick(g, ["team", "team_name", "team_display_name"])
    pts = pick(g, ["pts", "points"])
    poss = pick(g, ["o_poss", "offensive_possessions"])
    sg = pick(s, ["contest_id", "game_id"])
    sd = pick(s, ["date", "game_date", "start_date"])
    g = g.rename(columns={gid: "game_id", tm: "team", pts: "pts", poss: "poss"})
    g["pts"] = pd.to_numeric(g.pts, errors="coerce")
    g["poss"] = pd.to_numeric(g.poss, errors="coerce")
    g["ortg"] = 100 * g.pts / g.poss
    ss = s[[sg, sd]].drop_duplicates(sg).rename(columns={sg: "game_id", sd: "date"})
    g = g.merge(ss, on="game_id", how="left")
    g["date"] = pd.to_datetime(g.date, errors="coerce")
    g = g.dropna(subset=["date", "team", "ortg", "poss", "pts"])
    g = g.sort_values(["date", "game_id", "team"], kind="mergesort")
    cnt = g.groupby("game_id").size()
    g = g[g.game_id.isin(cnt[cnt == 2].index)].copy()
    g["ix"] = g.groupby("game_id").cumcount()
    o = g[["game_id", "ix", "team"]].copy()
    o["ix"] = 1 - o.ix
    o = o.rename(columns={"team": "opp"})
    g = g.merge(o, on=["game_id", "ix"], validate="one_to_one")
    g["gposs"] = g.groupby("game_id").poss.transform("mean")
    return g.sort_values(["date", "game_id", "team"], kind="mergesort").reset_index(drop=True)


def continuity():
    ps = []
    for sy in SEASONS:
        d = pd.read_parquet(RAW + f"mbb/player_box/parquet/ncaa_mbb_player_box_{sy}.parquet")
        tc = pick(d, ["team", "team_name", "team_display_name"])
        nc = pick(d, ["athlete", "player", "name", "athlete_display_name"])
        mc = pick(d, ["mins", "minutes"])
        z = d[[tc, nc, mc]].copy()
        z["pk"] = z[nc].astype(str).str.lower().str.replace(r"[^a-z0-9]", "", regex=True)
        z[mc] = pd.to_numeric(z[mc], errors="coerce").fillna(0)
        z = z.groupby([tc, "pk"], dropna=False)[mc].sum().reset_index().rename(columns={tc: "team", mc: "min"})
        z["season"] = sy
        ps.append(z)
    p = pd.concat(ps, ignore_index=True)
    cr, trs = [], []
    for sy in range(2022, 2026):
        a, c = p[p.season == sy - 1], p[p.season == sy]
        same = a.merge(c[["team", "pk"]].drop_duplicates(), on=["team", "pk"], how="left", indicator=True)
        for tm, z in same.groupby("team"):
            den = z["min"].sum()
            cr.append({"season": sy, "team": tm, "rp": z.loc[z._merge == "both", "min"].sum() / den if den else np.nan})
        au = a.groupby("pk").filter(lambda z: z.team.nunique() == 1).sort_values("min").drop_duplicates("pk", keep="last")
        cu = c.groupby("pk").filter(lambda z: z.team.nunique() == 1).sort_values("min").drop_duplicates("pk", keep="last")
        m = au.merge(cu[["pk", "team"]], on="pk", suffixes=("_prev", "_cur"))
        m = m[m.team_prev != m.team_cur].copy(); m["season"] = sy; trs.append(m)
    C = pd.DataFrame(cr); T = pd.concat(trs, ignore_index=True)
    tf = T.groupby(["season", "team_cur"]).agg(tr_min=("min", "sum")).reset_index().rename(columns={"team_cur": "team"})
    C = C.merge(tf, on=["season", "team"], how="left"); C["tr_min"] = C.tr_min.fillna(0)
    C["ztr"] = C.groupby("season").tr_min.transform(lambda x: (x - x.mean()) / (x.std() + 1e-9))
    return C


def run_frozen(g, C, use_r2):
    O, D, n, hist = {}, {}, {}, {}
    prev = None; pred = []
    for date, day in g.groupby("date", sort=True):
        sy = int(day.season.iloc[0])
        if prev is None: prev = sy
        if sy != prev:
            cc = C[C.season == sy].set_index("team")
            for t in set(O) | set(D):
                rp, zt = .375, 0.0
                if t in cc.index:
                    r = cc.loc[t]; rp = r.rp if pd.notna(r.rp) else .375; zt = r.ztr if pd.notna(r.ztr) else 0.0
                fac = np.clip(1.20 + .30 * (rp - .375) + .06 * zt, .55, 1.45)
                O[t] = O.get(t, 0.0) * fac; D[t] = D.get(t, 0.0) * fac
            n, hist, prev = {}, {}, sy
        cc = C[C.season == sy].set_index("team")
        ups = []
        for r in day.itertuples(index=False):
            nn = n.get(r.team, 0)
            p = 100 + O.get(r.team, 0.0) - D.get(r.opp, 0.0)
            w = 1.0
            if use_r2:
                w = (r.gposs / 70.0) * (1 + .20 * (1 - 2 * min(nn / 10, 1)))
                w = np.clip(w, .50, 1.50)
            rp, zt = np.nan, np.nan
            if r.team in cc.index:
                rr = cc.loc[r.team]; rp = rr.rp; zt = rr.ztr
            h = hist.get(r.team, [])
            vol = float(np.std(h[-5:], ddof=1)) if len(h) >= 2 else np.nan
            if sy >= 2022:
                pred.append((sy, date, r.game_id, r.team, r.opp, nn, r.ortg, p, r.gposs, r.pts, rp, zt, vol))
            e = r.ortg - p; ups.append((r.team, r.opp, e, .12 * w))
        for a, b, e, k in ups:
            O[a] = O.get(a, 0.0) + k * e
            D[b] = D.get(b, 0.0) - k * e
            n[a] = n.get(a, 0) + 1
            hist.setdefault(a, []).append(float(e))
    return pd.DataFrame(pred, columns=["season","date","game_id","team","opp","prior_games","actual_ortg","pred_ortg","gposs","pts","rp","ztr","prior_vol"])


def metrics(x):
    z = x.copy(); z["err"] = z.actual_ortg - z.pred_ortg; z["ae"] = z.err.abs()
    gd = z.groupby(["season","game_id"]).err.agg(lambda a: abs(a.iloc[0] - a.iloc[1]) if len(a) == 2 else np.nan)
    out = {"ortg_mae": float(z.ae.mean()), "diff_mae": float(gd.mean())}
    for sy in range(2022, 2026): out[f"diff_{sy}"] = float(gd.loc[sy].mean())
    return out, gd


def pace_predictions(g):
    team_season_pace = g.groupby(["season","team"]).gposs.mean().to_dict()
    season_global = g.groupby("season").gposs.mean().to_dict()
    cur_sum, cur_n, prev = {}, {}, None; out = []
    for date, day in g.groupby("date", sort=True):
        sy = int(day.season.iloc[0])
        if prev != sy:
            cur_sum, cur_n, prev = {}, {}, sy
        ups = []
        for r in day.itertuples(index=False):
            def state(t):
                if cur_n.get(t, 0): return cur_sum[t] / cur_n[t]
                return team_season_pace.get((sy - 1, t), season_global.get(sy - 1, 70.0))
            pp = (state(r.team) + state(r.opp)) / 2
            if sy >= 2022: out.append((r.game_id, r.team, pp))
            ups.append((r.team, r.gposs))
        for t, p in ups:
            cur_sum[t] = cur_sum.get(t, 0.0) + p; cur_n[t] = cur_n.get(t, 0) + 1
    return pd.DataFrame(out, columns=["game_id","team","pred_pace"])


def make_games(team_pred, pace):
    x = team_pred.merge(pace, on=["game_id","team"], how="left", validate="one_to_one")
    x["pred_score"] = x.pred_ortg * x.pred_pace / 100
    rows = []
    for (sy, gid), z in x.sort_values(["season","game_id","team"]).groupby(["season","game_id"], sort=True):
        if len(z) != 2: continue
        a, b = z.iloc[0], z.iloc[1]
        rows.append({
            "season":sy,"date":a.date,"game_id":gid,"team0":a.team,"team1":b.team,
            "pred0":a.pred_score,"pred1":b.pred_score,"actual0":a.pts,"actual1":b.pts,
            "pred_margin":a.pred_score-b.pred_score,"actual_margin":a.pts-b.pts,
            "pred_total":a.pred_score+b.pred_score,"actual_total":a.pts+b.pts,
            "n_avg":(a.prior_games+b.prior_games)/2,
            "rp_avg":np.nanmean([a.rp,b.rp]),"ztr_abs_avg":np.nanmean([abs(a.ztr),abs(b.ztr)]),
            "vol_avg":np.nanmean([a.prior_vol,b.prior_vol]) if (pd.notna(a.prior_vol) or pd.notna(b.prior_vol)) else np.nan,
        })
    q = pd.DataFrame(rows)
    q["margin_resid"] = q.actual_margin - q.pred_margin
    q["total_resid"] = q.actual_total - q.pred_total
    q["e0"] = q.actual0 - q.pred0; q["e1"] = q.actual1 - q.pred1
    return q


def ece(p, y, bins=10):
    d = pd.DataFrame({"p":p,"y":y}).dropna(); d["b"] = np.minimum((d.p*bins).astype(int), bins-1)
    if d.empty: return np.nan
    vals=[]
    for _, z in d.groupby("b"):
        vals.append(len(z)/len(d)*abs(z.p.mean()-z.y.mean()))
    return float(sum(vals))


def logloss(p, y):
    p=np.clip(np.asarray(p,float),1e-9,1-1e-9); y=np.asarray(y,float)
    return float(np.mean(-(y*np.log(p)+(1-y)*np.log(1-p))))


def probabilistic(games):
    annual=[]; scored=[]; spread_rows=[]; total_rows=[]
    for sy in [2023,2024,2025]:
        tr=games[(games.season>=2022)&(games.season<sy)].copy(); te=games[games.season==sy].copy()
        mr=tr.margin_resid; tt=tr.total_resid
        mm, ms=float(mr.mean()),float(mr.std(ddof=1)); tm, ts=float(tt.mean()),float(tt.std(ddof=1))
        er=np.r_[tr.e0.values,tr.e1.values]; em=float(np.mean(er))
        cov=np.cov(tr[["e0","e1"]].values.T)+np.eye(2)*1e-6
        ll=-multivariate_normal(mean=[em,em],cov=cov,allow_singular=True).logpdf(te[["e0","e1"]].values).mean()
        te["pwin"] = norm.cdf((te.pred_margin + mm)/ms)
        te["win"]=(te.actual_margin>0).astype(int)
        te["global_sigma_margin"]=ms; te["margin_mean_resid"]=mm
        annual.append({
            "season":sy,"joint_nll":float(ll),"margin_sd":ms,"total_sd":ts,
            "margin_cov80":float((abs(te.margin_resid-mm)<=norm.ppf(.90)*ms).mean()),
            "margin_cov90":float((abs(te.margin_resid-mm)<=norm.ppf(.95)*ms).mean()),
            "total_cov80":float((abs(te.total_resid-tm)<=norm.ppf(.90)*ts).mean()),
            "total_cov90":float((abs(te.total_resid-tm)<=norm.ppf(.95)*ts).mean()),
            "win_brier":float(np.mean((te.pwin-te.win)**2)),"win_logloss":logloss(te.pwin,te.win),"win_ece":ece(te.pwin,te.win),
        })
        for d in [-10,-7,-5,-3,0,3,5,7,10]:
            p=1-norm.cdf((d-mm)/ms); y=(te.margin_resid>d).astype(int)
            spread_rows.append({"season":sy,"delta":d,"p":p,"rate":float(y.mean()),"brier":float(np.mean((p-y)**2)),"n":len(y)})
        for d in [-20,-15,-10,-5,0,5,10,15,20]:
            p=1-norm.cdf((d-tm)/ts); y=(te.total_resid>d).astype(int)
            total_rows.append({"season":sy,"delta":d,"p":p,"rate":float(y.mean()),"brier":float(np.mean((p-y)**2)),"n":len(y)})
        scored.append(te)
    return pd.DataFrame(annual),pd.concat(scored,ignore_index=True),pd.DataFrame(spread_rows),pd.DataFrame(total_rows)


def gaussian_nll(resid, mean, sigma):
    s=np.clip(np.asarray(sigma,float),1e-6,None); r=np.asarray(resid,float)
    return float(np.mean(.5*np.log(2*np.pi*s*s)+.5*((r-mean)/s)**2))


def stage_bucket(n):
    return pd.cut(n,[-1,1,4,9,1e9],labels=["0-1","2-4","5-9","10+"])


def uncertainty(games):
    all_rows=[]; summaries=[]
    for sy in [2023,2024,2025]:
        tr=games[(games.season>=2022)&(games.season<sy)].copy(); te=games[games.season==sy].copy()
        tr["stage"]=stage_bucket(tr.n_avg); te["stage"]=stage_bucket(te.n_avg)
        mu=float(tr.margin_resid.mean()); global_sd=float(tr.margin_resid.std(ddof=1))
        sds=tr.groupby("stage",observed=False).margin_resid.std().to_dict()
        te["sigma_stage"]=te.stage.map(sds).astype(float).fillna(global_sd).clip(lower=1e-6)
        nll_global=gaussian_nll(te.margin_resid,mu,np.full(len(te),global_sd))
        nll_stage=gaussian_nll(te.margin_resid,mu,te.sigma_stage)
        # roster model predicts absolute error; convert E|N(0,sigma)| to sigma.
        for frame in [tr,te]:
            frame["churn"]=(1-frame.rp_avg).fillna((1-tr.rp_avg).median())
            frame["zload"]=frame.ztr_abs_avg.fillna(tr.ztr_abs_avg.median())
            frame["n10"]=frame.n_avg.clip(upper=10)
        lr=LinearRegression().fit(tr[["n10","churn","zload"]],np.log1p(tr.margin_resid.abs()))
        pred_abs=np.expm1(lr.predict(te[["n10","churn","zload"]])).clip(.1,None)
        te["sigma_roster"]=np.clip(pred_abs/np.sqrt(2/np.pi),1e-3,None)
        nll_roster=gaussian_nll(te.margin_resid,mu,te.sigma_roster)
        # volatility model.
        vol_med=float(tr.vol_avg.median()) if tr.vol_avg.notna().any() else 10.0
        tr["volx"]=tr.vol_avg.fillna(vol_med); te["volx"]=te.vol_avg.fillna(vol_med)
        lv=LinearRegression().fit(tr[["n10","volx"]],np.log1p(tr.margin_resid.abs()))
        pred_abs_v=np.expm1(lv.predict(te[["n10","volx"]])).clip(.1,None)
        te["sigma_vol"]=np.clip(pred_abs_v/np.sqrt(2/np.pi),1e-3,None)
        nll_vol=gaussian_nll(te.margin_resid,mu,te.sigma_vol)
        summaries.append({"season":sy,"nll_global":nll_global,"nll_stage":nll_stage,"nll_roster":nll_roster,"nll_vol":nll_vol})
        te["mu_resid"]=mu; all_rows.append(te)
    return pd.DataFrame(summaries),pd.concat(all_rows,ignore_index=True)


def roster_timestamp_audit():
    cols=[]
    for sy in [2022,2023,2024,2025]:
        d=pd.read_parquet(RAW+f"mbb/roster/parquet/ncaa_mbb_roster_{sy}.parquet")
        cols.extend(list(d.columns))
    uniq=sorted(set(cols)); temporal=[c for c in uniq if any(k in c.lower() for k in ["date","time","effective","start","end","as_of","updated"])]
    return {"columns":uniq,"temporal_columns":temporal,"preseason_snapshot_supported":bool(temporal)}


def main():
    g=load_games(); C=continuity()
    r1=run_frozen(g,C,False); r2=run_frozen(g,C,True)
    m1,gd1=metrics(r1); m2,gd2=metrics(r2)
    h2=hashlib.sha256(pd.util.hash_pandas_object(gd2,index=True).values.tobytes()).hexdigest()
    targets={"r1_ortg":11.720062,"r1_diff":15.618256,"r2_ortg":11.577690,"r2_diff":15.373316,"r2_hash":"7a37d452ba277678ed0d645a1959363cd4a33b24d58f6af74ac15d6937ff8bfb"}
    exact_r1=abs(m1["ortg_mae"]-targets["r1_ortg"])<5e-7 and abs(m1["diff_mae"]-targets["r1_diff"])<5e-7
    exact_r2=abs(m2["ortg_mae"]-targets["r2_ortg"])<5e-7 and abs(m2["diff_mae"]-targets["r2_diff"])<5e-7 and h2==targets["r2_hash"]

    pace=pace_predictions(g); games=make_games(r2,pace)
    # paired efficiency integrity
    pair_counts=r2.groupby("game_id").size(); reciprocal=0
    for _,z in r2.groupby("game_id"):
        if len(z)==2 and z.iloc[0].team==z.iloc[1].opp and z.iloc[1].team==z.iloc[0].opp: reciprocal+=1
    t02={"team_rows":len(r2),"games":int((pair_counts==2).sum()),"bad_pair_counts":int((pair_counts!=2).sum()),"reciprocal_pairs":reciprocal,"ortg_mae":m2["ortg_mae"]}

    prob, scored, spreads, totals=probabilistic(games)
    unc, us=uncertainty(games)

    # T03 decisions
    cov90_err=float(np.mean(np.abs(prob[["margin_cov90","total_cov90"]].values-.90)))
    t0301="GREEN" if np.isfinite(prob.joint_nll).all() and cov90_err<.05 else "HOLD"
    wb=float(np.average(prob.win_brier,weights=[len(games[games.season==s]) for s in prob.season]))
    we=float(np.average(prob.win_ece,weights=[len(games[games.season==s]) for s in prob.season]))
    t0302="GREEN" if wb<.25 and we<.05 else "HOLD"
    spread_gap=float(np.average(abs(spreads.p-spreads.rate),weights=spreads.n)); spread_b=float(np.average(spreads.brier,weights=spreads.n))
    total_gap=float(np.average(abs(totals.p-totals.rate),weights=totals.n)); total_b=float(np.average(totals.brier,weights=totals.n))
    t0303="GREEN" if spread_gap<.05 and spread_b<.25 else "HOLD"
    t0304="GREEN" if total_gap<.05 and total_b<.25 else "HOLD"

    # T04 decisions
    pooled={c:float(unc[c].mean()) for c in ["nll_global","nll_stage","nll_roster","nll_vol"]}
    t0401="GREEN" if pooled["nll_stage"]<pooled["nll_global"] else "REJECTED"
    us["stage"]=stage_bucket(us.n_avg)
    maturity=us.groupby("stage",observed=False).margin_resid.apply(lambda z:z.abs().mean()).to_dict()
    rho=float(spearmanr(us.n_avg,us.margin_resid.abs(),nan_policy="omit").statistic)
    t0402="GREEN" if rho<0 and maturity.get("0-1",0)>maturity.get("10+",0)+1 else "REJECTED"
    roster_wins=int((unc.nll_roster<unc.nll_stage).sum()); vol_wins=int((unc.nll_vol<unc.nll_stage).sum())
    t0403="GREEN" if pooled["nll_roster"]<pooled["nll_stage"] and roster_wins>=2 else "REJECTED"
    t0404="GREEN" if pooled["nll_vol"]<pooled["nll_stage"] and vol_wins>=2 else "REJECTED"

    candidates={"stage":pooled["nll_stage"],"roster":pooled["nll_roster"],"vol":pooled["nll_vol"]}; best=min(candidates,key=candidates.get)
    us["sigma_best"]=us[{"stage":"sigma_stage","roster":"sigma_roster","vol":"sigma_vol"}[best]]
    rel_rho=float(spearmanr(us.sigma_best,us.margin_resid.abs(),nan_policy="omit").statistic)
    us["reliability_decile"]=pd.qcut(us.sigma_best.rank(method="first"),10,labels=False)
    dec=us.groupby("reliability_decile").margin_resid.apply(lambda z:z.abs().mean()).to_dict()
    t0501="GREEN" if rel_rho>0.05 and dec.get(0,999)<dec.get(9,0)-1 else "REJECTED"
    us["pwin_best"]=norm.cdf((us.pred_margin+us.mu_resid)/us.sigma_best); us["win"]=(us.actual_margin>0).astype(int)
    curves=[]
    for frac in [1.0,.75,.50,.30,.20,.10]:
        vals=[]
        for sy,z in us.groupby("season"):
            k=max(1,int(np.floor(len(z)*frac))); zz=z.nsmallest(k,"sigma_best"); vals.append(zz)
        zz=pd.concat(vals)
        curves.append({"coverage":frac,"n":len(zz),"margin_mae":float(zz.margin_resid.abs().mean()),"win_brier":float(np.mean((zz.pwin_best-zz.win)**2))})
    curves=pd.DataFrame(curves); basec=curves[curves.coverage==1].iloc[0]
    c50=curves[curves.coverage==.5].iloc[0]; c30=curves[curves.coverage==.3].iloc[0]
    t0502="GREEN" if c50.margin_mae<basec.margin_mae and c30.margin_mae<basec.margin_mae and c50.win_brier<basec.win_brier else "REJECTED"

    roster_audit=roster_timestamp_audit(); t0102="GREEN" if roster_audit["preseason_snapshot_supported"] else "HOLD"

    gates={
        "T00.07":{"decision":"GREEN" if exact_r1 else "HOLD","evidence":m1},
        "T00.08":{"decision":"GREEN" if exact_r2 else "HOLD","evidence":dict(m2,hash=h2)},
        "T01.02":{"decision":t0102,"evidence":roster_audit},
        "T02.02":{"decision":"GREEN" if t02["bad_pair_counts"]==0 else "HOLD","evidence":t02},
        "T02.03":{"decision":"GREEN" if t02["reciprocal_pairs"]==t02["games"] else "HOLD","evidence":t02},
        "T03.01":{"decision":t0301,"evidence":{"annual":prob.to_dict("records"),"mean_abs_90_coverage_error":cov90_err}},
        "T03.02":{"decision":t0302,"evidence":{"weighted_brier":wb,"weighted_ece":we,"annual":prob[["season","win_brier","win_logloss","win_ece"]].to_dict("records")}},
        "T03.03":{"decision":t0303,"evidence":{"weighted_abs_calibration_gap":spread_gap,"weighted_brier":spread_b}},
        "T03.04":{"decision":t0304,"evidence":{"weighted_abs_calibration_gap":total_gap,"weighted_brier":total_b}},
        "T04.01":{"decision":t0401,"evidence":{"pooled_nll":pooled,"annual":unc.to_dict("records")}},
        "T04.02":{"decision":t0402,"evidence":{"spearman_games_vs_abs_error":rho,"mae_by_stage":{str(k):float(v) for k,v in maturity.items()}}},
        "T04.03":{"decision":t0403,"evidence":{"pooled_nll":pooled,"season_wins_vs_stage":roster_wins}},
        "T04.04":{"decision":t0404,"evidence":{"pooled_nll":pooled,"season_wins_vs_stage":vol_wins}},
        "T05.01":{"decision":t0501,"evidence":{"best_uncertainty_model":best,"spearman_sigma_vs_abs_error":rel_rho,"mae_by_reliability_decile":{str(k):float(v) for k,v in dec.items()}}},
        "T05.02":{"decision":t0502,"evidence":{"best_uncertainty_model":best,"coverage_curves":curves.to_dict("records")}},
    }
    report={"gates":gates,"targets":targets,"r1":m1,"r2":dict(m2,hash=h2),"probability":prob.to_dict("records"),"uncertainty":unc.to_dict("records"),"coverage_curves":curves.to_dict("records")}
    (OUT/"gate_decisions.json").write_text(json.dumps(report,indent=2,default=str)+"\n")
    r2.to_parquet(OUT/"r2_team_predictions.parquet",index=False); games.to_parquet(OUT/"game_predictions.parquet",index=False)
    prob.to_csv(OUT/"probability_summary.csv",index=False); spreads.to_csv(OUT/"spread_thresholds.csv",index=False); totals.to_csv(OUT/"total_thresholds.csv",index=False)
    unc.to_csv(OUT/"uncertainty_summary.csv",index=False); curves.to_csv(OUT/"coverage_curves.csv",index=False)
    print("GATE_BATCH_15_RESULTS")
    print(json.dumps(report,indent=2,default=str))


if __name__ == "__main__": main()
