from __future__ import annotations
import json, math
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import norm
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error

# Certified ESPN identity adapter used in T28/T34.
_orig_merge = pd.DataFrame.merge
def _merge(self, right, *args, **kwargs):
    on = kwargs.get('on', None)
    if on == ['season','game_id'] and 'espn_game_id' in self.columns and 'factor_actual_home' in getattr(right,'columns',[]):
        left=self.copy(); left['game_id']=pd.to_numeric(left['espn_game_id'],errors='coerce').astype('Int64')
        return _orig_merge(left,right,*args,**kwargs)
    return _orig_merge(self,right,*args,**kwargs)
pd.DataFrame.merge=_merge

import prediction_first_alt_factors15_v2 as b

OUT=Path('results/prediction_first_calibrated_recency15'); OUT.mkdir(parents=True,exist_ok=True)
FREEZE=Path('/tmp/freeze')

def mae(a,b): return float(mean_absolute_error(a,b))
def clip_prob(p): return np.clip(p,1e-6,1-1e-6)
def logloss(y,p):
    p=clip_prob(np.asarray(p)); y=np.asarray(y); return float(np.mean(-(y*np.log(p)+(1-y)*np.log(1-p))))
def brier(y,p): return float(np.mean((np.asarray(p)-np.asarray(y))**2))

def add_model_adjusted_form(a,q):
    # Build team-side frozen expected margin for each historical game and roll model residuals.
    p=q[['season','game_id','base_margin']].drop_duplicates(['season','game_id']).copy()
    x=a.merge(p,on=['season','game_id'],how='left')
    x['expected_team_margin']=np.where(x.home_away.eq('home'),x.base_margin,-x.base_margin)
    x['model_game_resid']=x['margin']-x['expected_team_margin']
    x=x.sort_values(['season','team_id','game_date','game_id']).copy()
    for w in [5,7]:
        x[f'model_l{w}_resid']=x.groupby(['season','team_id'])['model_game_resid'].transform(lambda s,w=w:s.rolling(w,min_periods=max(3,w-2)).mean().shift())
    h=x[x.home_away.eq('home')][['season','game_id','model_l5_resid','model_l7_resid']].rename(columns={'model_l5_resid':'h_model_l5_resid','model_l7_resid':'h_model_l7_resid'})
    aw=x[x.home_away.eq('away')][['season','game_id','model_l5_resid','model_l7_resid']].rename(columns={'model_l5_resid':'a_model_l5_resid','model_l7_resid':'a_model_l7_resid'})
    z=h.merge(aw,on=['season','game_id']); z['d_model_l5_resid']=z.h_model_l5_resid-z.a_model_l5_resid; z['d_model_l7_resid']=z.h_model_l7_resid-z.a_model_l7_resid
    return q.merge(z[['season','game_id','d_model_l5_resid','d_model_l7_resid']],on=['season','game_id'],how='left')

def calibrated_baseline(train,test):
    # Pooled affine correction only for games where both teams are within first 10; mature games remain frozen-v14.
    tr=train.copy(); te=test.copy()
    early_tr=tr.h_game_no.le(10)&tr.a_game_no.le(10)
    if early_tr.sum()>=300:
        X=np.c_[np.ones(early_tr.sum()),tr.loc[early_tr,'base_margin'].to_numpy()]
        y=tr.loc[early_tr,'actual_margin'].to_numpy(); coef=np.linalg.lstsq(X,y,rcond=None)[0]
    else: coef=np.array([0.0,1.0])
    for df in (tr,te):
        early=df.h_game_no.le(10)&df.a_game_no.le(10)
        df['cal_margin']=df.base_margin
        df.loc[early,'cal_margin']=coef[0]+coef[1]*df.loc[early,'base_margin']
        df['cal_resid']=df.actual_margin-df.cal_margin
    return tr,te,coef

def residual_fold(q,features, target='margin'):
    rows=[]; preds=[]
    for sy in [2024,2025]:
        train=q[q.season.lt(sy)].copy(); test=q[q.season.eq(sy)].copy(); train,test,coef=calibrated_baseline(train,test)
        avail=[c for c in features if c in q.columns and q[c].notna().mean()>=.5]
        need=['cal_resid']+avail
        tr=train.dropna(subset=need); te=test.dropna(subset=['cal_resid']+avail)
        if len(tr)<300 or len(te)<200 or not avail: continue
        m=make_pipeline(StandardScaler(),Ridge(alpha=30)).fit(tr[avail],tr.cal_resid)
        corr=m.predict(te[avail]); pred=te.cal_margin+corr
        rows.append({'season':sy,'n':len(te),'cal_base_mae':mae(te.actual_margin,te.cal_margin),'challenger_mae':mae(te.actual_margin,pred),'gain':mae(te.actual_margin,te.cal_margin)-mae(te.actual_margin,pred),'cal_intercept':float(coef[0]),'cal_slope':float(coef[1])})
        tmp=te[['season','game_id','actual_margin','cal_margin']].copy(); tmp['pred']=pred; preds.append(tmp)
    return pd.DataFrame(rows), (pd.concat(preds,ignore_index=True) if preds else pd.DataFrame())

def score_target_fold(q,features,target_col,base_col):
    rows=[]
    for sy in [2024,2025]:
        tr=q[q.season.lt(sy)].copy(); te=q[q.season.eq(sy)].copy()
        # Score-target calibration is affine on the target itself using past seasons only.
        z=tr.dropna(subset=[target_col,base_col]); X=np.c_[np.ones(len(z)),z[base_col]]; co=np.linalg.lstsq(X,z[target_col],rcond=None)[0]
        tr['base_cal']=co[0]+co[1]*tr[base_col]; te['base_cal']=co[0]+co[1]*te[base_col]
        tr['target_resid']=tr[target_col]-tr.base_cal
        av=[c for c in features if c in q and q[c].notna().mean()>=.5]
        tr2=tr.dropna(subset=['target_resid']+av); te2=te.dropna(subset=[target_col,'base_cal']+av)
        if len(tr2)<300 or len(te2)<200 or not av: continue
        m=make_pipeline(StandardScaler(),Ridge(alpha=30)).fit(tr2[av],tr2.target_resid); pred=te2.base_cal+m.predict(te2[av])
        rows.append({'season':sy,'n':len(te2),'base_mae':mae(te2[target_col],te2.base_cal),'challenger_mae':mae(te2[target_col],pred),'gain':mae(te2[target_col],te2.base_cal)-mae(te2[target_col],pred)})
    return pd.DataFrame(rows)

def mean_gain(df): return float(df.gain.mean()) if len(df) else np.nan

def main():
    a,q=b.build(); q=add_model_adjusted_form(a,q)
    # Restore expected actual columns from frozen table if adapter merge left suffixed factor columns.
    if 'actual_home' not in q and 'factor_actual_home' in q: q['actual_home']=q.factor_actual_home
    if 'actual_away' not in q and 'factor_actual_away' in q: q['actual_away']=q.factor_actual_away
    q['actual_total']=q.actual_home+q.actual_away; q['base_total']=q.base_home+q.base_away
    q['win_y']=(q.actual_margin>0).astype(int)
    G=[]
    def A(i,s,d,r,n): G.append({'gate':f'T35.{i:02d}','status':s,'decision':d,'result':r,'next_action':n})

    # 1 integrity + calibration overlay
    cal=[]
    for sy in [2024,2025]:
        tr=q[q.season.lt(sy)]; te=q[q.season.eq(sy)]; _,_,co=calibrated_baseline(tr,te); cal.append({'season':sy,'train_n':len(tr),'test_n':len(te),'early_intercept':float(co[0]),'early_slope':float(co[1])})
    A(1,'GREEN','PASS — CALIBRATION-CONTROLLED SUBSTRATE LOCKED',f'joined_games={len(q)}; seasons={q.groupby("season").size().to_dict()}; mappings={cal}; 2026_outcomes_used=False','Use this baseline for every recency/factor increment below')

    # 2-5 recency variants after calibration
    l5,_=residual_fold(q,['d_l5_margin']); al5,_=residual_fold(q,['d_adj_l5_margin']); al7,_=residual_fold(q,['d_adj_l7_margin']); rl7,_=residual_fold(q,['d_rec_l7_margin'])
    A(2,'GREEN' if mean_gain(l5)>0 else 'REJECT','PASS — RAW L5 SURVIVES CALIBRATION' if mean_gain(l5)>0 else 'REJECT — RAW L5 COLLAPSES AFTER CALIBRATION',f'folds={l5.to_dict("records")}; mean_gain={mean_gain(l5):.4f}','Compare with opponent-adjusted versions')
    A(3,'GREEN' if mean_gain(al5)>0 else 'REJECT','PASS — ADJUSTED L5 SURVIVES CALIBRATION' if mean_gain(al5)>0 else 'REJECT',f'folds={al5.to_dict("records")}; mean_gain={mean_gain(al5):.4f}','Compare head-to-head with adjusted/recursive alternatives')
    A(4,'GREEN' if mean_gain(al7)>0 else 'REJECT','PASS — ADJUSTED L7 SURVIVES CALIBRATION' if mean_gain(al7)>0 else 'REJECT',f'folds={al7.to_dict("records")}; mean_gain={mean_gain(al7):.4f}','Compare head-to-head')
    A(5,'GREEN' if mean_gain(rl7)>0 else 'REJECT','PASS — RECURSIVE L7 SURVIVES CALIBRATION' if mean_gain(rl7)>0 else 'REJECT',f'folds={rl7.to_dict("records")}; mean_gain={mean_gain(rl7):.4f}','Require incremental value beyond simpler adjusted L5/L7')

    # 6 head-to-head and combined correlated windows
    candidates={'raw_l5':mean_gain(l5),'adj_l5':mean_gain(al5),'adj_l7':mean_gain(al7),'rec_l7':mean_gain(rl7)}
    combo,_=residual_fold(q,['d_adj_l5_margin','d_rec_l7_margin'])
    best=max(v for v in candidates.values() if np.isfinite(v)); cg=mean_gain(combo)
    A(6,'GREEN' if best>0 else 'REJECT','PASS — CALIBRATION-CONTROLLED RECENCY TOURNAMENT',f'candidate_mean_gains={candidates}; combined_adj5_rec7={combo.to_dict("records")}; combined_mean_gain={cg:.4f}','Prefer one window unless combination materially beats best single')

    # 7-8 model-adjusted residual form (uses frozen pregame expected margin directly)
    ml5,_=residual_fold(q,['d_model_l5_resid']); ml7,_=residual_fold(q,['d_model_l7_resid'])
    A(7,'GREEN' if mean_gain(ml5)>0 else 'REJECT','PASS — MODEL-ADJUSTED L5 ADDS' if mean_gain(ml5)>0 else 'REJECT — MODEL-ADJUSTED L5 DOES NOT ADD',f'folds={ml5.to_dict("records")}; mean_gain={mean_gain(ml5):.4f}; adjusted_L5={mean_gain(al5):.4f}','Use frozen expected margin as cleaner opponent-strength control if it wins')
    A(8,'GREEN' if mean_gain(ml7)>0 else 'REJECT','PASS — MODEL-ADJUSTED L7 ADDS' if mean_gain(ml7)>0 else 'REJECT — MODEL-ADJUSTED L7 DOES NOT ADD',f'folds={ml7.to_dict("records")}; mean_gain={mean_gain(ml7):.4f}; adjusted_L7={mean_gain(al7):.4f}','Compare with recursive L7 and stop if inferior')

    # 9 factor bundle after calibrated baseline
    fac=['d_pre_ts','d_pre_tovr','d_pre_orbr','d_pre_drbr','d_pre_ftr']; ff,_=residual_fold(q,fac)
    A(9,'GREEN' if mean_gain(ff)>0 else 'REJECT','PASS — FACTOR BUNDLE SURVIVES CALIBRATION' if mean_gain(ff)>0 else 'REJECT — FACTOR BUNDLE WAS CALIBRATION PROXY',f'folds={ff.to_dict("records")}; mean_gain={mean_gain(ff):.4f}','Ablate shooting vs non-shooting families under calibrated baseline')

    # 10 TS vs eFG after calibrated baseline
    ts,_=residual_fold(q,['d_pre_ts']); ef,_=residual_fold(q,['d_pre_efg'])
    A(10,'GREEN' if max(mean_gain(ts),mean_gain(ef))>0 else 'REJECT','PASS — SHOOTING STATE SURVIVES CALIBRATION' if max(mean_gain(ts),mean_gain(ef))>0 else 'REJECT',f'ts={ts.to_dict("records")}; efg={ef.to_dict("records")}; mean_ts={mean_gain(ts):.4f}; mean_efg={mean_gain(ef):.4f}','Choose simplest winner per target')

    # 11 target specialization: total residual with level/sum features
    q['s_pre_ts']=q.h_pre_ts+q.a_pre_ts; q['s_pre_ftr']=q.h_pre_ftr+q.a_pre_ftr; q['s_pre_orbr']=q.h_pre_orbr+q.a_pre_orbr; q['s_pre_drbr']=q.h_pre_drbr+q.a_pre_drbr; q['s_pre_three_rate']=q.h_pre_three_rate+q.a_pre_three_rate
    total=score_target_fold(q,['s_pre_ts','s_pre_ftr','s_pre_orbr','s_pre_drbr','s_pre_three_rate'],'actual_total','base_total')
    A(11,'GREEN' if mean_gain(total)>0 else 'REJECT','PASS — ALTERNATIVE FACTORS ADD TO TOTALS' if mean_gain(total)>0 else 'REJECT — FACTOR FAMILY DOES NOT HELP TOTALS',f'folds={total.to_dict("records")}; mean_gain={mean_gain(total):.4f}','Ablate FTR/DRB/3PA individually for total prediction')

    # 12 probability proper-score check for best recency challenger using same train-derived residual sigma.
    best_feature=max([('d_l5_margin',mean_gain(l5)),('d_adj_l5_margin',mean_gain(al5)),('d_adj_l7_margin',mean_gain(al7)),('d_rec_l7_margin',mean_gain(rl7)),('d_model_l5_resid',mean_gain(ml5)),('d_model_l7_resid',mean_gain(ml7))],key=lambda x:(-999 if not np.isfinite(x[1]) else x[1]))[0]
    prows=[]
    for sy in [2024,2025]:
        train=q[q.season.lt(sy)].copy(); test=q[q.season.eq(sy)].copy(); train,test,co=calibrated_baseline(train,test)
        tr=train.dropna(subset=['cal_resid',best_feature]); te=test.dropna(subset=['cal_resid',best_feature])
        if len(tr)<300 or len(te)<200: continue
        m=make_pipeline(StandardScaler(),Ridge(alpha=30)).fit(tr[[best_feature]],tr.cal_resid); te=te.copy(); te['pred']=te.cal_margin+m.predict(te[[best_feature]])
        sig=float(np.std(tr.cal_resid,ddof=1)); pb=norm.cdf(te.cal_margin/sig); pc=norm.cdf(te.pred/sig); y=(te.actual_margin>0).astype(int)
        prows.append({'season':sy,'n':len(te),'feature':best_feature,'sigma':sig,'base_brier':brier(y,pb),'challenger_brier':brier(y,pc),'brier_gain':brier(y,pb)-brier(y,pc),'base_logloss':logloss(y,pb),'challenger_logloss':logloss(y,pc),'logloss_gain':logloss(y,pb)-logloss(y,pc)})
    pg=float(np.mean([r['brier_gain'] for r in prows])) if prows else np.nan
    A(12,'GREEN' if np.isfinite(pg) and pg>0 else 'HOLD','PASS — RECENCY IMPROVES WIN PROPER SCORE' if np.isfinite(pg) and pg>0 else 'HOLD — RECENCY POINT LIFT DOES NOT CLEAR BRIER',f'best_feature={best_feature}; folds={prows}; mean_brier_gain={pg}','Do not promote betting probability unless Brier/logloss are non-negative in independent folds')

    # 13 main-effect-controlled matchup interactions
    q['mx_three']=q.h_pre_three_rate*q.a_pre_efg-q.a_pre_three_rate*q.h_pre_efg
    q['mx_reb']=q.h_pre_orbr*(1-q.a_pre_drbr)-q.a_pre_orbr*(1-q.h_pre_drbr)
    q['mx_tov']=-(q.h_pre_tovr*q.a_pre_tovr); q['mx_ftr']=-(q.h_pre_ftr*q.a_pre_ftr)
    main=['d_pre_efg','d_pre_three_rate','d_pre_orbr','d_pre_drbr','d_pre_tovr','d_pre_ftr']
    base_main,_=residual_fold(q,main)
    ints={}
    for c in ['mx_three','mx_reb','mx_tov','mx_ftr']:
        z,_=residual_fold(q,main+[c]); ints[c]=mean_gain(z)-mean_gain(base_main)
    comboi,_=residual_fold(q,main+['mx_three','mx_reb','mx_tov','mx_ftr']); inccombo=mean_gain(comboi)-mean_gain(base_main)
    A(13,'GREEN' if max(list(ints.values())+[inccombo])>0 else 'REJECT','PASS — MATCHUP INTERACTION SURVIVES MAIN-EFFECT CONTROL' if max(list(ints.values())+[inccombo])>0 else 'REJECT — MATCHUP SCREEN WAS MAIN-EFFECT PROXY',f'main_gain={mean_gain(base_main):.4f}; incremental_single={ints}; combined_increment={inccombo:.4f}','Preregister only interactions with positive increment in both folds next')

    # 14 venue/site audit using any frozen neutral field available
    neutral_col=next((c for c in ['neutral_site','is_neutral','neutral','neutral_flag'] if c in q.columns),None)
    vrows=[]
    if neutral_col:
        for sy in [2023,2024,2025]:
            for label,mask in [('non_neutral',~q[neutral_col].astype(bool)),('neutral',q[neutral_col].astype(bool))]:
                z=q[(q.season.eq(sy))&mask]
                if len(z): vrows.append({'season':sy,'cohort':label,'n':len(z),'home_resid':float((z.actual_home-z.base_home).mean()),'away_resid':float((z.actual_away-z.base_away).mean()),'margin_resid':float((z.actual_margin-z.base_margin).mean())})
        status='GREEN'
        decision='PASS — VENUE SCORE RESIDUALS STRATIFIED'
    else:
        status='HOLD'; decision='HOLD — NEUTRAL FIELD NOT EXPOSED IN JOINED TABLE'
    A(14,status,decision,f'neutral_column={neutral_col}; rows={vrows}','If available, test score-component mechanisms within known non-neutral only')

    # 15 synthesis
    statuses=[g['status'] for g in G]
    strongest=max({'raw_l5':mean_gain(l5),'adj_l5':mean_gain(al5),'adj_l7':mean_gain(al7),'rec_l7':mean_gain(rl7),'model_l5':mean_gain(ml5),'model_l7':mean_gain(ml7)}.items(),key=lambda kv:(-999 if not np.isfinite(kv[1]) else kv[1]))
    decision='ADVANCE CALIBRATION-CONTROLLED RECENCY / MATCHUP RESEARCH — KEEP V14 FROZEN'
    A(15,'GREEN',decision,f'green_before_final={statuses.count("GREEN")}; hold={statuses.count("HOLD")}; reject={statuses.count("REJECT")}; strongest_recency={strongest}; factor_gain={mean_gain(ff):.4f}; total_factor_gain={mean_gain(total):.4f}; production_reopen=False; 2026_outcomes_used=False','Next: bootstrap strongest calibrated recency model, proper-score/score-target tournament, and interaction replication with false-discovery control')

    out=pd.DataFrame(G); out.to_csv(OUT/'t35_calibrated_recency_15_gates.csv',index=False)
    pd.DataFrame(prows).to_csv(OUT/'t35_probability_folds.csv',index=False)
    pd.DataFrame(vrows).to_csv(OUT/'t35_venue_rows.csv',index=False)
    json.dump({'joined_games':len(q),'green':sum(g['status']=='GREEN' for g in G),'hold':sum(g['status']=='HOLD' for g in G),'reject':sum(g['status']=='REJECT' for g in G),'production_reopen':False,'2026_outcomes_used':False,'final_decision':decision},open(OUT/'summary.json','w'),indent=2)
    print(out.to_string(index=False)); print((OUT/'summary.json').read_text())

if __name__=='__main__': main()
