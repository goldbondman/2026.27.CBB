from __future__ import annotations
import math, re
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import multivariate_normal, multivariate_t, norm, t as student_t

import prediction_first_third15 as m

OUT=Path('results/prediction_first_distribution_integration_qc'); OUT.mkdir(parents=True,exist_ok=True)
G15=Path('/tmp/g15'); G10=Path('/tmp/g10'); VEN=Path('/tmp/venue')
RAW='https://raw.githubusercontent.com/sportsdataverse/ncaa-mbb-hoops-data/main/'
ESPN='https://github.com/sportsdataverse/sportsdataverse-data/releases/download/espn_mens_college_basketball_schedules/mbb_schedule_{}.parquet'

def canon(x):
    if pd.isna(x): return np.nan
    try:return str(int(float(x)))
    except:return str(x).strip()

def load_inputs():
    gp=pd.read_parquet(G15/'game_predictions.parquet'); gp['game_id']=gp.game_id.astype(str); gp['date']=pd.to_datetime(gp.date)
    r2=pd.read_parquet(G15/'r2_team_predictions.parquet'); r2['game_id']=r2.game_id.astype(str); r2['date']=pd.to_datetime(r2.date)
    pace=pd.read_parquet(G10/'pace_predictions.parquet'); pace['game_id']=pace.game_id.astype(str)
    if 'pred_pace' not in pace.columns:
        pp=pace[['game_id','pace_pred']].drop_duplicates('game_id').rename(columns={'pace_pred':'pred_pace'})
        pace=r2[['game_id','team']].merge(pp,on='game_id',how='left',validate='many_to_one')
    venue=pd.read_csv(VEN/'identity_match_ledger.csv',dtype={'game_id':str}); venue['neutral_known']=venue.neutral.notna(); venue['neutral']=venue.neutral.map(lambda x:str(x).lower()=='true' if pd.notna(x) else np.nan); venue['espn_game_id']=venue.espn_game_id.map(canon)
    sched=pd.read_parquet(RAW+'mbb/ncaa_mbb_schedule_master.parquet'); gid='game_id' if 'game_id' in sched.columns else 'contest_id'; sched=sched.rename(columns={gid:'game_id'}); sched['game_id']=sched.game_id.astype(str); sched['hk']=sched.home.map(m.nn); sched['ak']=sched.away.map(m.nn)
    return gp,r2,pace,venue,sched

def postmap():
    frames=[]
    for sy in range(2022,2026):
        d=pd.read_parquet(ESPN.format(sy));
        cand=[c for c in ['notes_type','notes_headline','season_type','tournament_id'] if c in d.columns]
        txt=pd.Series('',index=d.index)
        for c in cand: txt=txt+' '+d[c].astype(str)
        post=txt.str.lower().str.contains('postseason|ncaa tournament|conference tournament|tournament',regex=True,na=False)
        frames.append(pd.DataFrame({'espn_game_id':d.game_id.astype(str),'postseason':post.astype(bool)}))
    return pd.concat(frames,ignore_index=True).drop_duplicates('espn_game_id')

def main():
    gp,r2,pace,venue,sched=load_inputs()
    folds,_,_,use_int=m.build_fold_baselines(gp,r2,pace,venue,sched)
    pm=postmap(); rec=[]; postrec=[]
    df=10
    for sy,tr0,te0 in folds:
        tr=tr0.copy(); te=te0.copy()
        for z in [tr,te]:
            z['mu_home']=z.mu_home_int if use_int else z.mu_home0; z['mu_away']=z.mu_away_int if use_int else z.mu_away0
            z['e_home']=z.actual_home-z.mu_home; z['e_away']=z.actual_away-z.mu_away; z['stage']=z.n_avg.map(m.stage_bucket)
        E=tr[['e_home','e_away']].values; gm=E.mean(0); gc=np.cov(E.T)+np.eye(2)*1e-6
        X=te[['e_home','e_away']].values
        base_lp=multivariate_normal(gm,gc,allow_singular=True).logpdf(X)
        tshape=gc*(df-2)/df; tg_lp=multivariate_t(loc=gm,shape=tshape,df=df,allow_singular=True).logpdf(X)
        sg_lp=[]; st_lp=[]; sig_margin_stage=[]
        for r in te.itertuples(index=False):
            q=tr[tr.stage==r.stage]; w=len(q)/(len(q)+500.0)
            lm=q[['e_home','e_away']].mean().values if len(q)>20 else gm
            lc=np.cov(q[['e_home','e_away']].values.T)+np.eye(2)*1e-6 if len(q)>20 else gc
            mm=w*lm+(1-w)*gm; cc=w*lc+(1-w)*gc
            v=np.array([r.e_home,r.e_away])
            sg_lp.append(multivariate_normal(mm,cc,allow_singular=True).logpdf(v))
            st_lp.append(multivariate_t(loc=mm,shape=cc*(df-2)/df,df=df,allow_singular=True).logpdf(v))
            sig_margin_stage.append(math.sqrt(cc[0,0]+cc[1,1]-2*cc[0,1]))
        rec.append({'season':sy,'n':len(te),'gauss_global_nll':float(-np.mean(base_lp)),'gauss_stage_nll':float(-np.mean(sg_lp)),'t10_global_nll':float(-np.mean(tg_lp)),'t10_stage_nll':float(-np.mean(st_lp))})
        # Postseason integration: compare final stage sigma with prior-postseason sigma.
        tr=tr.merge(pm,on='espn_game_id',how='left'); te=te.merge(pm,on='espn_game_id',how='left'); tr['postseason']=tr.postseason.fillna(False); te['postseason']=te.postseason.fillna(False)
        post=te[te.postseason].copy()
        if len(post)>=30:
            # Map test-row stage sigma in original te order.
            sigmap=dict(zip(te0.game_id.astype(str),sig_margin_stage))
            post['stage_sd']=post.game_id.astype(str).map(sigmap)
            mr=post.actual_margin-(post.mu_home-post.mu_away)
            gmarg=float((tr.actual_margin-(tr.mu_home-tr.mu_away)).mean())
            prior_post=tr[tr.postseason]
            if len(prior_post)>=80:
                pmr=prior_post.actual_margin-(prior_post.mu_home-prior_post.mu_away)
                post_sd=float(pmr.std(ddof=1))
            else: post_sd=float((tr.actual_margin-(tr.mu_home-tr.mu_away)).std(ddof=1))
            nll_stage=float(np.mean(.5*np.log(2*np.pi*post.stage_sd**2)+.5*((mr-gmarg)/post.stage_sd)**2))
            nll_post=float(np.mean(.5*np.log(2*np.pi*post_sd**2)+.5*((mr-gmarg)/post_sd)**2))
            postrec.append({'season':sy,'n':len(post),'stage_margin_nll':nll_stage,'postseason_sd_nll':nll_post,'post_vs_stage_gain':nll_stage-nll_post,'postseason_sd':post_sd,'mean_stage_sd':float(post.stage_sd.mean())})
    R=pd.DataFrame(rec); R.to_csv(OUT/'t20_integration_tournament.csv',index=False)
    P=pd.DataFrame(postrec); P.to_csv(OUT/'t21_postseason_vs_stage.csv',index=False)
    print(R.to_string(index=False)); print(P.to_string(index=False))

if __name__=='__main__': main()
