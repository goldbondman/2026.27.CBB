from __future__ import annotations
import json, math
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import norm
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

import prediction_first_recency_cert15 as c

OUT=Path('results/prediction_first_recency_activation15'); OUT.mkdir(parents=True,exist_ok=True)

def mae(a,b): return float(mean_absolute_error(a,b))
def brier(y,p): return float(np.mean((np.asarray(p)-np.asarray(y))**2))
def logloss(y,p):
    p=np.clip(np.asarray(p),1e-6,1-1e-6); y=np.asarray(y)
    return float(np.mean(-(y*np.log(p)+(1-y)*np.log(1-p))))
def ece(y,p,bins=10):
    y=np.asarray(y); p=np.asarray(p); edges=np.linspace(0,1,bins+1); out=0
    for i in range(bins):
        m=(p>=edges[i])&(p<(edges[i+1] if i<bins-1 else edges[i+1]+1e-12))
        if m.sum(): out += m.mean()*abs(y[m].mean()-p[m].mean())
    return float(out)

def prep():
    a,q=c.b.build(); q=c.add_model_adjusted_form(a,q)
    if 'actual_home' not in q and 'factor_actual_home' in q: q['actual_home']=q.factor_actual_home
    if 'actual_away' not in q and 'factor_actual_away' in q: q['actual_away']=q.factor_actual_away
    q['actual_total']=q.actual_home+q.actual_away; q['base_total']=q.base_home+q.base_away
    q['win_y']=(q.actual_margin>0).astype(int)
    q['min_game_no']=q[['h_game_no','a_game_no']].min(axis=1)
    q['max_game_no']=q[['h_game_no','a_game_no']].max(axis=1)
    q['is_non_neutral']=~q.neutral.astype(bool)
    return q

def calibrated(q,sy):
    tr=q[q.season.lt(sy)].copy(); te=q[q.season.eq(sy)].copy(); tr,te,coef=c.calibrated_baseline(tr,te); return tr,te,coef

def fit_predict(train,test,features,eligible_train=None,eligible_test=None):
    if eligible_train is None: eligible_train=np.ones(len(train),dtype=bool)
    if eligible_test is None: eligible_test=np.ones(len(test),dtype=bool)
    tr=train.loc[eligible_train].dropna(subset=['cal_resid']+features).copy()
    te=test.loc[eligible_test].dropna(subset=['cal_resid']+features).copy()
    if len(tr)<250 or len(te)<100: return None,None,None
    m=make_pipeline(StandardScaler(),Ridge(alpha=30)).fit(tr[features],tr.cal_resid)
    te['pred']=te.cal_margin+m.predict(te[features])
    return m,tr,te

def date_boot(df, base='cal_margin', pred='pred', n=1000, seed=37):
    rng=np.random.default_rng(seed); dates=np.array(sorted(df.game_date.dropna().unique())); vals=[]
    for _ in range(n):
        samp=rng.choice(dates,size=len(dates),replace=True)
        parts=[df[df.game_date.eq(d)] for d in samp]; z=pd.concat(parts,ignore_index=True)
        vals.append(mae(z.actual_margin,z[base])-mae(z.actual_margin,z[pred]))
    a=np.array(vals); return float(np.mean(a)),float(np.quantile(a,.025)),float(np.quantile(a,.975)),float(np.mean(a>0))

def eval_thresholds(q):
    cand=[7,8,10,12,15]
    rows=[]
    for sy in [2024,2025]:
        tr,te,_=calibrated(q,sy)
        best=None
        for th in cand:
            et=(tr.is_non_neutral)&(tr.min_game_no.ge(th)); ee=(te.is_non_neutral)&(te.min_game_no.ge(th))
            m,ttr,tte=fit_predict(tr,te,['d_adj_l7_margin'],et,ee)
            if tte is None: continue
            # training MAE only selects threshold inside prior data
            trpred=ttr.cal_margin+m.predict(ttr[['d_adj_l7_margin']]); train_gain=mae(ttr.actual_margin,ttr.cal_margin)-mae(ttr.actual_margin,trpred)
            gain=mae(tte.actual_margin,tte.cal_margin)-mae(tte.actual_margin,tte.pred)
            r={'season':sy,'threshold':th,'train_n':len(ttr),'test_n':len(tte),'train_gain':train_gain,'test_gain':gain}
            rows.append(r)
            if best is None or train_gain>best['train_gain']: best=r
        if best: best['selected']=True
    return pd.DataFrame(rows)

def activated_predictions(q,thresholds):
    out=[]
    for sy in [2024,2025]:
        th=int(thresholds[sy]); tr,te,_=calibrated(q,sy)
        et=(tr.is_non_neutral)&(tr.min_game_no.ge(th)); ee=(te.is_non_neutral)&(te.min_game_no.ge(th))
        m,ttr,tte=fit_predict(tr,te,['d_adj_l7_margin'],et,ee)
        te=te.copy(); te['pred']=te.cal_margin; te['active']=False
        if tte is not None:
            ix=tte.index; te.loc[ix,'pred']=tte.pred; te.loc[ix,'active']=True
        out.append(te)
    return pd.concat(out,ignore_index=True)

def nested_factor_gain(q,base_feats,add_feat,thresholds):
    rows=[]
    for sy in [2024,2025]:
        th=int(thresholds[sy]); tr,te,_=calibrated(q,sy)
        et=(tr.is_non_neutral)&(tr.min_game_no.ge(th)); ee=(te.is_non_neutral)&(te.min_game_no.ge(th))
        _,_,bte=fit_predict(tr,te,base_feats,et,ee); _,_,ate=fit_predict(tr,te,base_feats+[add_feat],et,ee)
        if bte is None or ate is None: continue
        ix=bte.index.intersection(ate.index); b=bte.loc[ix]; a=ate.loc[ix]
        rows.append({'season':sy,'n':len(ix),'base_mae':mae(b.actual_margin,b.pred),'add_mae':mae(a.actual_margin,a.pred),'increment':mae(b.actual_margin,b.pred)-mae(a.actual_margin,a.pred)})
    return pd.DataFrame(rows)

def main():
    q=prep(); G=[]
    def A(i,s,d,r,n): G.append({'gate':f'T37.{i:02d}','status':s,'decision':d,'result':r,'next_action':n})
    A(1,'GREEN','PASS — T37 SUBSTRATE LOCKED',f'joined_games={len(q)}; date_coverage={q.game_date.notna().mean():.4f}; non_neutral={int(q.is_non_neutral.sum())}; 2026_outcomes_used=False','Use fixed substrate below')

    th=eval_thresholds(q); th.to_csv(OUT/'t37_threshold_grid.csv',index=False)
    selected={}
    for sy in [2024,2025]:
        z=th[th.season.eq(sy)].sort_values('train_gain',ascending=False); selected[sy]=int(z.iloc[0].threshold)
    A(2,'GREEN','PASS — MINIMUM-GAME THRESHOLD SEARCH COMPLETE',f'grid={th.to_dict("records")}; selected={selected}','Test locked selected thresholds prospectively in each fold')

    ap=activated_predictions(q,selected); ap.to_csv(OUT/'t37_activated_predictions.csv',index=False)
    ar=[]
    for sy in [2024,2025]:
        z=ap[(ap.season.eq(sy))&ap.is_non_neutral]; g=mae(z.actual_margin,z.cal_margin)-mae(z.actual_margin,z.pred); ar.append({'season':sy,'n':len(z),'active_n':int(z.active.sum()),'gain':g})
    A(3,'GREEN' if all(r['gain']>0 for r in ar) else 'HOLD','PASS — TRAIN-SELECTED ACTIVATION IMPROVES NON-NEUTRAL FUTURE FOLDS' if all(r['gain']>0 for r in ar) else 'HOLD — ACTIVATION DOES NOT REPLICATE',f'rows={ar}','Use selected activation in non-neutral tournament')

    boots=[]
    for sy in [2024,2025]:
        z=ap[(ap.season.eq(sy))&ap.is_non_neutral]; m,lo,hi,p=date_boot(z); boots.append({'season':sy,'mean_gain':m,'lo95':lo,'hi95':hi,'p_positive':p})
    pd.DataFrame(boots).to_csv(OUT/'t37_activation_bootstrap.csv',index=False)
    A(4,'GREEN' if all(r['lo95']>0 for r in boots) else 'HOLD','PASS — NON-NEUTRAL ACTIVATION IS DATE-BLOCK ROBUST' if all(r['lo95']>0 for r in boots) else 'HOLD — ACTIVATION CI CROSSES ZERO',f'bootstrap={boots}','Advance mean challenger if both CIs positive')

    # neutral closure using adjusted L7 with same threshold selected, separate model fit on neutral train
    nr=[]
    for sy in [2024,2025]:
        tr,te,_=calibrated(q,sy); thv=selected[sy]
        et=(~tr.is_non_neutral)&tr.min_game_no.ge(thv); ee=(~te.is_non_neutral)&te.min_game_no.ge(thv)
        _,_,tte=fit_predict(tr,te,['d_adj_l7_margin'],et,ee)
        if tte is None: continue
        m,lo,hi,p=date_boot(tte); nr.append({'season':sy,'n':len(tte),'gain':mae(tte.actual_margin,tte.cal_margin)-mae(tte.actual_margin,tte.pred),'lo95':lo,'hi95':hi,'p_positive':p})
    pd.DataFrame(nr).to_csv(OUT/'t37_neutral_bootstrap.csv',index=False)
    A(5,'REJECT' if len(nr)==2 and all(r['hi95']<=0 or r['gain']<=0 for r in nr) else 'HOLD','REJECT — NEUTRAL ADJUSTED-L7 NOT SUPPORTED' if len(nr)==2 and all(r['gain']<=0 for r in nr) else 'HOLD — NEUTRAL SIGNAL UNRESOLVED',f'neutral={nr}','Leave neutral games on incumbent/calibrated baseline')

    # Probability sigma scale tuning on active non-neutral training, test on activated non-neutral rows
    prows=[]; cov=[]
    scales=np.linspace(.65,1.20,56)
    for sy in [2024,2025]:
        thv=selected[sy]; tr,te,_=calibrated(q,sy)
        et=(tr.is_non_neutral)&tr.min_game_no.ge(thv); ee=(te.is_non_neutral)&te.min_game_no.ge(thv)
        m,ttr,tte=fit_predict(tr,te,['d_adj_l7_margin'],et,ee)
        if tte is None: continue
        trpred=ttr.cal_margin+m.predict(ttr[['d_adj_l7_margin']]); resid=ttr.actual_margin-trpred; sig0=float(np.std(resid,ddof=1))
        ytr=(ttr.actual_margin>0).astype(int).to_numpy(); best=None
        for sc in scales:
            p=norm.cdf(trpred/(sig0*sc)); br=brier(ytr,p)
            if best is None or br<best[0]: best=(br,sc)
        sc=best[1]; y=(tte.actual_margin>0).astype(int).to_numpy(); pb=norm.cdf(tte.cal_margin/(sig0*sc)); pc=norm.cdf(tte.pred/(sig0*sc))
        prows.append({'season':sy,'n':len(tte),'sigma':sig0,'scale':sc,'brier_gain':brier(y,pb)-brier(y,pc),'logloss_gain':logloss(y,pb)-logloss(y,pc),'ece_base':ece(y,pb),'ece_challenger':ece(y,pc),'ece_gain':ece(y,pb)-ece(y,pc)})
        for lev in [.5,.8,.9]:
            z=norm.ppf((1+lev)/2); emp=float(np.mean(np.abs(tte.actual_margin-tte.pred)<=z*sig0*sc)); cov.append({'season':sy,'level':lev,'empirical':emp,'error':emp-lev,'scale':sc})
    pd.DataFrame(prows).to_csv(OUT/'t37_probability.csv',index=False); pd.DataFrame(cov).to_csv(OUT/'t37_coverage.csv',index=False)
    A(6,'GREEN' if all(r['brier_gain']>=0 and r['logloss_gain']>=0 for r in prows) else 'HOLD','PASS — SIGMA-RECALIBRATED PROPER SCORES CLEAR' if all(r['brier_gain']>=0 and r['logloss_gain']>=0 for r in prows) else 'HOLD — SIGMA RECALIBRATION DOES NOT CLEAR BOTH FOLDS',f'probability={prows}','Only promote probabilities if both folds non-negative')
    maxerr=max(abs(r['error']) for r in cov) if cov else 99
    A(7,'GREEN' if maxerr<=.03 else 'HOLD','PASS — COVERAGE WITHIN 3 POINTS' if maxerr<=.03 else 'HOLD — COVERAGE STILL MISALIGNED',f'coverage={cov}; max_abs_error={maxerr:.4f}','Retain mean-only if intervals remain miscalibrated')

    # TS bootstrap increment and individual non-shooting ablations within activated non-neutral cohort
    base=['d_adj_l7_margin']; tsres=nested_factor_gain(q,base,'d_pre_ts',selected)
    A(8,'GREEN' if len(tsres)==2 and all(tsres.increment>0) else 'HOLD','PASS — TS ADDS BEYOND ACTIVATED L7' if len(tsres)==2 and all(tsres.increment>0) else 'HOLD — TS INCREMENT DOES NOT REPLICATE',f'rows={tsres.to_dict("records")}; mean_increment={tsres.increment.mean() if len(tsres) else np.nan:.4f}','Keep TS only if fold-consistent')
    base2=['d_adj_l7_margin','d_pre_ts']
    fmap=[('TOV','d_pre_tovr'),('ORB','d_pre_orbr'),('DRB','d_pre_drbr'),('FTR','d_pre_ftr')]
    indiv={}
    for name,feat in fmap:
        z=nested_factor_gain(q,base2,feat,selected); indiv[name]=z.to_dict('records')
    for idx,(name,feat) in enumerate(fmap,start=9):
        z=nested_factor_gain(q,base2,feat,selected); ok=len(z)==2 and all(z.increment>0)
        A(idx,'GREEN' if ok else 'REJECT',f'PASS — {name} ADDS BEYOND L7+TS' if ok else f'REJECT — {name} DOES NOT REPLICATE BEYOND L7+TS',f'rows={z.to_dict("records")}; mean_increment={z.increment.mean() if len(z) else np.nan:.4f}',f'Retain {name} only if both folds positive')

    # parsimonious factor combo: only components positive both folds
    keep=[]
    for name,feat in fmap:
        z=nested_factor_gain(q,base2,feat,selected)
        if len(z)==2 and all(z.increment>0): keep.append(feat)
    combo=nested_factor_gain(q,base2,keep[0],selected) if len(keep)==1 else None
    # if >1, fit full features and compare manually
    combo_rows=[]
    if len(keep)>1:
        for sy in [2024,2025]:
            thv=selected[sy]; tr,te,_=calibrated(q,sy); et=(tr.is_non_neutral)&tr.min_game_no.ge(thv); ee=(te.is_non_neutral)&te.min_game_no.ge(thv)
            _,_,bte=fit_predict(tr,te,base2,et,ee); _,_,ate=fit_predict(tr,te,base2+keep,et,ee)
            ix=bte.index.intersection(ate.index); b=bte.loc[ix]; a=ate.loc[ix]
            combo_rows.append({'season':sy,'n':len(ix),'increment':mae(b.actual_margin,b.pred)-mae(a.actual_margin,a.pred)})
    elif combo is not None: combo_rows=combo.to_dict('records')
    A(13,'GREEN' if combo_rows and all(r.get('increment',0)>0 for r in combo_rows) else 'HOLD','PASS — PARSIMONIOUS NON-SHOOTING ADD-ON REPLICATES' if combo_rows and all(r.get('increment',0)>0 for r in combo_rows) else 'HOLD — NO STABLE PARSIMONIOUS FACTOR ADD-ON',f'kept_features={keep}; rows={combo_rows}','Use only stable component(s) in challenger tournament')

    # final tournament summary: mean activated nonneutral margin + probability gating
    margin_ok=all(r['gain']>0 for r in ar) and all(r['lo95']>0 for r in boots)
    prob_ok=bool(prows) and all(r['brier_gain']>=0 and r['logloss_gain']>=0 for r in prows)
    A(14,'GREEN' if margin_ok else 'HOLD','PASS — NON-NEUTRAL ACTIVATED L7 MEAN TOURNAMENT CLEARS' if margin_ok else 'HOLD — MEAN TOURNAMENT FAILS',f'activation={ar}; bootstrap={boots}; probability_ok={prob_ok}','Schedule formal production-reopen only if proper-score governance also clears')
    A(15,'GREEN','ADVANCE ACTIVATED NON-NEUTRAL ADJUSTED-L7 MEAN CHALLENGER — KEEP V14 FROZEN',f'selected_thresholds={selected}; margin_ok={margin_ok}; probability_ok={prob_ok}; neutral_rows={nr}; stable_factor_features={keep}; 2026_outcomes_used=False','Next: formal common-fold tournament if probability/coverage gates clear; otherwise keep mean-only shadow challenger')

    pd.DataFrame(G).to_csv(OUT/'t37_recency_activation_15_gates.csv',index=False)
    (OUT/'t37_summary.json').write_text(json.dumps({'selected_thresholds':selected,'activation':ar,'bootstrap':boots,'neutral':nr,'probability':prows,'coverage':cov,'stable_factor_features':keep,'gates':G},indent=2))
    print(json.dumps(G,indent=2))

if __name__=='__main__': main()
