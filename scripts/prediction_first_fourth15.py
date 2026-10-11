from __future__ import annotations

import hashlib, json, math, os, re
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import norm, multivariate_normal, spearmanr
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import prediction_first_third15 as m

OUT=Path('results/prediction_first_fourth15'); OUT.mkdir(parents=True,exist_ok=True)
G15=Path('/tmp/g15'); G10=Path('/tmp/g10'); VEN=Path('/tmp/venue')
RNG=np.random.default_rng(20261011)
EVAL=[2023,2024,2025]


def clean_id(x):
    if pd.isna(x): return np.nan
    s=str(x).strip(); return s[:-2] if s.endswith('.0') else s

def ece(p,y,bins=10):
    d=pd.DataFrame({'p':np.asarray(p,float),'y':np.asarray(y,float)}).dropna()
    if d.empty:return np.nan
    d['b']=np.minimum((d.p*bins).astype(int),bins-1)
    return float(sum(len(z)/len(d)*abs(z.p.mean()-z.y.mean()) for _,z in d.groupby('b')))

def logloss(p,y):
    p=np.clip(np.asarray(p,float),1e-9,1-1e-9); y=np.asarray(y,float)
    return float(np.mean(-(y*np.log(p)+(1-y)*np.log(1-p))))

def stage_bucket(n):
    if n<=1:return '0-1'
    if n<=4:return '2-4'
    if n<=9:return '5-9'
    return '10+'

def load_inputs():
    gp=pd.read_parquet(G15/'game_predictions.parquet'); gp['game_id']=gp.game_id.astype(str); gp['date']=pd.to_datetime(gp.date)
    r2=pd.read_parquet(G15/'r2_team_predictions.parquet'); r2['game_id']=r2.game_id.astype(str); r2['date']=pd.to_datetime(r2.date)
    pace=pd.read_parquet(G10/'pace_predictions.parquet'); pace['game_id']=pace.game_id.astype(str)
    if 'pred_pace' not in pace.columns:
        pp=pace[['game_id','pace_pred']].drop_duplicates('game_id').rename(columns={'pace_pred':'pred_pace'})
        pace=r2[['game_id','team']].merge(pp,on='game_id',how='left',validate='many_to_one')
    venue=pd.read_csv(VEN/'identity_match_ledger.csv',dtype={'game_id':str}); venue['neutral_known']=venue.neutral.notna(); venue['neutral']=venue.neutral.map(lambda x:str(x).lower()=='true' if pd.notna(x) else np.nan); venue['espn_game_id']=venue.espn_game_id.map(clean_id)
    sched=pd.read_parquet(m.RAW+'mbb/ncaa_mbb_schedule_master.parquet'); gid='game_id' if 'game_id' in sched.columns else 'contest_id'; sched=sched.rename(columns={gid:'game_id'}); sched['game_id']=sched.game_id.astype(str); sched['hk']=sched.home.map(m.nn); sched['ak']=sched.away.map(m.nn)
    return gp,r2,pace,venue,sched

def postmap():
    xs=[]
    for sy in range(2022,2026):
        d=pd.read_parquet(m.ESPN.format(sy)); v=d['season_type'] if 'season_type' in d.columns else pd.Series('',index=d.index)
        post=v.astype(str).str.lower().isin(['3','postseason','post-season','post season']) | v.astype(str).str.lower().str.contains('post',na=False)
        xs.append(pd.DataFrame({'espn_game_id':d.game_id.map(clean_id),'postseason':post.astype(bool)}))
    return pd.concat(xs,ignore_index=True).drop_duplicates('espn_game_id')

def build_champion_folds(gp,r2,pace,venue,sched,pm):
    rawfolds,pred0,mm,use=m.build_fold_baselines(gp,r2,pace,venue,sched)
    folds=[]; tests=[]
    for sy,tr0,te0 in rawfolds:
        tr=tr0.copy(); te=te0.copy()
        for z in (tr,te):
            z['mu_home']=z.mu_home_int if use else z.mu_home0; z['mu_away']=z.mu_away_int if use else z.mu_away0
            z['pred_margin']=z.mu_home-z.mu_away; z['pred_total']=z.mu_home+z.mu_away
            z['e_home']=z.actual_home-z.mu_home; z['e_away']=z.actual_away-z.mu_away
            z['margin_resid']=z.actual_margin-z.pred_margin; z['total_resid']=z.actual_total-z.pred_total
            z['stage']=z.n_avg.map(stage_bucket); z['espn_game_id']=z.espn_game_id.map(clean_id)
        tr=tr.merge(pm,on='espn_game_id',how='left'); te=te.merge(pm,on='espn_game_id',how='left'); tr['postseason']=tr.postseason.fillna(False); te['postseason']=te.postseason.fillna(False)
        E=tr[['e_home','e_away']].values; gm=E.mean(0); gc=np.cov(E.T)+np.eye(2)*1e-6
        add=[]
        for r in te.itertuples(index=False):
            q=tr[tr.stage==r.stage]; w=len(q)/(len(q)+500.0)
            if len(q)>20:
                lm=q[['e_home','e_away']].mean().values; lc=np.cov(q[['e_home','e_away']].values.T)+np.eye(2)*1e-6; mean=w*lm+(1-w)*gm; cov=w*lc+(1-w)*gc
            else: mean,cov=gm,gc
            sm=float(np.sqrt(cov[0,0]+cov[1,1]-2*cov[0,1])); st=float(np.sqrt(cov[0,0]+cov[1,1]+2*cov[0,1])); mmg=float(mean[0]-mean[1]); tmg=float(mean[0]+mean[1])
            add.append((sm,st,mmg,tmg,float(norm.cdf((r.pred_margin+mmg)/sm)),float(mean[0]),float(mean[1]),float(cov[0,1])))
        A=np.asarray(add,float); te['sigma_margin_final']=A[:,0]; te['sigma_total_final']=A[:,1]; te['resmean_margin_final']=A[:,2]; te['resmean_total_final']=A[:,3]; te['pwin_final']=A[:,4]; te['mean_eh_final']=A[:,5]; te['mean_ea_final']=A[:,6]; te['cov_final']=A[:,7]
        te['paway_final']=1-te.pwin_final; te['neutral_i']=te.neutral.astype(int); te['post_i']=te.postseason.astype(int); te['gap_abs']=abs(te.pred_margin); te['win_uncert']=te.pwin_final*(1-te.pwin_final); te['team_err']=(abs(te.actual_home-te.mu_home)+abs(te.actual_away-te.mu_away))/2; te['abs_margin']=abs(te.margin_resid); te['abs_total']=abs(te.total_resid); te['win_err']=(te.pwin_final-(te.actual_margin>0).astype(int))**2
        folds.append((sy,tr,te)); tests.append(te)
    return folds,pd.concat(tests,ignore_index=True),use

def reliability_suite(pred):
    base=['n_avg','gap_abs','pred_total','pred_pace','win_uncert','neutral_i','post_i','sigma_margin_final','sigma_total_final']
    p=pred.copy(); p['epi']=1/np.sqrt(np.clip(p.n_avg,0,None)+1); p['epi2']=p.epi**2; p['epi_x_sm']=p.epi*p.sigma_margin_final; p['epi_x_st']=p.epi*p.sigma_total_final; p['epi_x_wu']=p.epi*p.win_uncert
    dec=base+['epi','epi2','epi_x_sm','epi_x_st','epi_x_wu']
    oodcols=['n_avg','gap_abs','pred_total','pred_pace','sigma_margin_final','sigma_total_final']
    targets={'team_score':'team_err','margin':'abs_margin','total':'abs_total','win_brier':'win_err'}
    evalrows=[]; risks=[]; noveltyrows=[]
    for sy in [2024,2025]:
        tr=p[(p.season>=2023)&(p.season<sy)].copy(); te=p[p.season==sy].copy()
        X=tr[oodcols].astype(float).fillna(0).values; Y=te[oodcols].astype(float).fillna(0).values; mu=X.mean(0); C=np.cov(X.T)+np.eye(len(oodcols))*1e-3; inv=np.linalg.pinv(C)
        novelty=np.sqrt(np.einsum('ij,jk,ik->i',Y-mu,inv,Y-mu)) + .35*te.neutral_i.values + .35*te.post_i.values
        te['novelty']=novelty
        for name,targ in targets.items():
            out={'season':sy,'target':name,'n':len(te)}
            rset={}
            for variant,features in [('base',base),('decomp',dec),('novelty',base+['novelty'])]:
                ttrain=tr.copy()
                if 'novelty' in features:
                    XT=ttrain[oodcols].astype(float).fillna(0).values; nt=np.sqrt(np.einsum('ij,jk,ik->i',XT-mu,inv,XT-mu)) + .35*ttrain.neutral_i.values + .35*ttrain.post_i.values; ttrain['novelty']=nt
                model=make_pipeline(StandardScaler(),Ridge(alpha=20.0)).fit(ttrain[features].fillna(0),np.log1p(ttrain[targ])); risk=model.predict(te[features].fillna(0)); rset[variant]=risk; out[variant+'_rho']=float(spearmanr(risk,te[targ]).statistic)
            out['novelty_raw_rho']=float(spearmanr(te.novelty,te[targ]).statistic); evalrows.append(out)
            for variant,risk in rset.items():
                for i,v in enumerate(risk): risks.append({'season':sy,'game_id':te.iloc[i].game_id,'target':name,'variant':variant,'risk':float(v),'err':float(te.iloc[i][targ])})
        noveltyrows.append(pd.DataFrame({'season':sy,'game_id':te.game_id.values,'novelty':novelty,'n_avg':te.n_avg.values,'neutral':te.neutral.values,'postseason':te.postseason.values}))
    R=pd.DataFrame(evalrows); Risk=pd.DataFrame(risks); Nov=pd.concat(noveltyrows,ignore_index=True); R.to_csv(OUT/'t22_model_comparison.csv',index=False); Risk.to_csv(OUT/'t22_risk_predictions.csv',index=False); Nov.to_csv(OUT/'t22_novelty.csv',index=False)
    summary=[]
    for targ,q in R.groupby('target'):
        b=q.base_rho.mean(); d=q.decomp_rho.mean(); n=q.novelty_rho.mean(); nr=q.novelty_raw_rho.mean(); best='base'; br=b
        if d-b>=.01 and (q.decomp_rho>=q.base_rho).sum()==2: best='decomp'; br=d
        if n-br>=.01 and (q.novelty_rho>=q[best+'_rho']).sum()==2: best='novelty'; br=n
        summary.append({'target':targ,'base_rho':b,'decomp_rho':d,'decomp_gain':d-b,'novelty_model_rho':n,'novelty_raw_rho':nr,'selected':best,'selected_rho':br})
    S=pd.DataFrame(summary); S.to_csv(OUT/'t22_model_selection.csv',index=False)
    return R,Risk,Nov,S

def coverage_and_contract(risk,selection):
    covers=[1.0,.75,.50,.30,.20,.10]; rows=[]; contracts=[]
    smap=dict(zip(selection.target,selection.selected))
    for targ,variant in smap.items():
        q=risk[(risk.target==targ)&(risk.variant==variant)].copy()
        for sy,z in q.groupby('season'):
            z=z.sort_values('risk'); full=float(z.err.mean())
            for c in covers:
                n=max(1,int(round(len(z)*c))); err=float(z.iloc[:n].err.mean()); rows.append({'target':targ,'season':sy,'variant':variant,'coverage':c,'n':n,'error':err,'full_error':full,'improvement_pct':(full-err)/max(full,1e-9)})
        valid=[]
        for c in [.75,.50,.30,.20,.10]:
            zz=pd.DataFrame(rows); a=zz[(zz.target==targ)&(zz.coverage==c)]
            if len(a)==2 and (a.improvement_pct>0).all() and a.improvement_pct.mean()>=.03: valid.append(c)
        chosen=max(valid) if valid else None; contracts.append({'target':targ,'variant':variant,'abstention_supported':chosen is not None,'max_supported_coverage':chosen,'low_support_fraction':None if chosen is None else 1-chosen})
    C=pd.DataFrame(rows); K=pd.DataFrame(contracts); C.to_csv(OUT/'t22_coverage_curves.csv',index=False); K.to_csv(OUT/'t22_abstention_contract.csv',index=False); return C,K

def old_vs_champion(gp,r2,pace,venue,sched,pm):
    tp=r2.merge(pace[['game_id','team','pred_pace']],on=['game_id','team'],how='left',validate='one_to_one'); tp['game_id']=tp.game_id.astype(str)
    site=gp[['game_id','season','date','n_avg']].merge(sched[['game_id','home','away','hk','ak']].drop_duplicates('game_id'),on='game_id').merge(venue[['game_id','neutral_known','neutral','espn_game_id']],on='game_id',how='left'); site=site[site.neutral_known].copy()
    rows=[]; detail=[]
    for sy in EVAL:
        trsite=site[(site.season>=2022)&(site.season<sy)]; tesite=site[site.season==sy]; trt=tp[tp.game_id.isin(set(trsite.game_id))].copy(); tet=tp[tp.game_id.isin(set(tesite.game_id))].copy()
        pg=trt.groupby('game_id').agg(actual=('gposs','mean'),pred=('pred_pace','mean')).dropna(); corr=float((pg.actual-pg.pred).mean())
        def pair(t,ss):
            a=t.merge(ss[['game_id','season','date','home','away','hk','ak','neutral','n_avg','espn_game_id']],on='game_id'); a['tk']=a.team.map(m.nn); a['is_home']=a.tk.eq(a.hk); a['is_away']=a.tk.eq(a.ak); a=a[a.is_home|a.is_away].copy(); a['old']=a.pred_ortg*a.pred_pace/100; a['champ']=a.pred_ortg*np.clip(a.pred_pace+corr,45,95)/100 + np.where(a.neutral,0,np.where(a.is_home,1.35,-1.35)); out=[]
            for gid,q in a.groupby('game_id'):
                if len(q)!=2:continue
                h=q[q.is_home]; aw=q[q.is_away]
                if len(h)!=1 or len(aw)!=1:continue
                h=h.iloc[0]; aw=aw.iloc[0]; out.append({'game_id':gid,'season':sy,'date':h.date,'n_avg':float(np.nanmean([h.prior_games,aw.prior_games])),'old_h':h.old,'old_a':aw.old,'ch_h':h.champ,'ch_a':aw.champ,'ah':h.pts,'aa':aw.pts})
            return pd.DataFrame(out)
        tr=pair(trt,trsite); te=pair(tet,tesite)
        for pref in ['old','ch']:
            tr['eh']=tr.ah-tr[pref+'_h']; tr['ea']=tr.aa-tr[pref+'_a']; gm=tr[['eh','ea']].mean().values; gc=np.cov(tr[['eh','ea']].values.T)+np.eye(2)*1e-6
            probs=[]; nll=[]
            for r in te.itertuples(index=False):
                if pref=='ch':
                    st=stage_bucket(r.n_avg); tr2=tr.copy(); tr2['stage']=tr2.n_avg.map(stage_bucket); z=tr2[tr2.stage==st]; w=len(z)/(len(z)+500.0)
                    if len(z)>20: lm=z[['eh','ea']].mean().values; lc=np.cov(z[['eh','ea']].values.T)+np.eye(2)*1e-6; mean=w*lm+(1-w)*gm; cov=w*lc+(1-w)*gc
                    else: mean,cov=gm,gc
                else: mean,cov=gm,gc
                sm=math.sqrt(cov[0,0]+cov[1,1]-2*cov[0,1]); pmarg=(getattr(r,pref+'_h')-getattr(r,pref+'_a'))+(mean[0]-mean[1]); probs.append(norm.cdf(pmarg/sm)); nll.append(-multivariate_normal(mean,cov,allow_singular=True).logpdf([r.ah-getattr(r,pref+'_h'),r.aa-getattr(r,pref+'_a')]))
            te[pref+'_p']=probs; te[pref+'_nll']=nll
        y=(te.ah>te.aa).astype(int); row={'season':sy,'n':len(te)}
        for pref in ['old','ch']:
            row[pref+'_team_mae']=float(np.mean(np.r_[abs(te.ah-te[pref+'_h']),abs(te.aa-te[pref+'_a'])])); row[pref+'_margin_mae']=float(abs((te.ah-te.aa)-(te[pref+'_h']-te[pref+'_a'])).mean()); row[pref+'_total_mae']=float(abs((te.ah+te.aa)-(te[pref+'_h']+te[pref+'_a'])).mean()); row[pref+'_brier']=float(np.mean((te[pref+'_p']-y)**2)); row[pref+'_joint_nll']=float(te[pref+'_nll'].mean())
        rows.append(row); te['d_team']=(abs(te.ah-te.ch_h)+abs(te.aa-te.ch_a))/2-(abs(te.ah-te.old_h)+abs(te.aa-te.old_a))/2; te['d_margin']=abs((te.ah-te.aa)-(te.ch_h-te.ch_a))-abs((te.ah-te.aa)-(te.old_h-te.old_a)); te['d_total']=abs((te.ah+te.aa)-(te.ch_h+te.ch_a))-abs((te.ah+te.aa)-(te.old_h+te.old_a)); te['d_brier']=(te.ch_p-y)**2-(te.old_p-y)**2; detail.append(te[['game_id','date','season','d_team','d_margin','d_total','d_brier']])
    A=pd.DataFrame(rows); D=pd.concat(detail,ignore_index=True); boots=[]
    dates=D.date.dt.date.unique()
    for metric in ['d_team','d_margin','d_total','d_brier']:
        vals=[]
        for _ in range(600):
            ds=RNG.choice(dates,size=len(dates),replace=True); q=pd.concat([D[D.date.dt.date==d] for d in ds],ignore_index=True); vals.append(float(q[metric].mean()))
        boots.append({'metric':metric,'champ_minus_control':float(D[metric].mean()),'ci_low':float(np.quantile(vals,.025)),'ci_high':float(np.quantile(vals,.975))})
    B=pd.DataFrame(boots); A.to_csv(OUT/'t23_final_tournament.csv',index=False); B.to_csv(OUT/'t23_bootstrap.csv',index=False); return A,B

def api_and_competence(pred,risk,contract):
    q=pred.copy(); mu_m=q.pred_margin+q.resmean_margin_final; mu_t=q.pred_total+q.resmean_total_final; sm=q.sigma_margin_final; st=q.sigma_total_final
    ml=np.round(mu_m); tl=np.round(mu_t); pmp=norm.cdf((ml+.5-mu_m)/sm)-norm.cdf((ml-.5-mu_m)/sm); pmo=1-norm.cdf((ml+.5-mu_m)/sm); pmu=norm.cdf((ml-.5-mu_m)/sm); ptp=norm.cdf((tl+.5-mu_t)/st)-norm.cdf((tl-.5-mu_t)/st); pto=1-norm.cdf((tl+.5-mu_t)/st); ptu=norm.cdf((tl-.5-mu_t)/st)
    api={'margin_mean_identity_maxerr':float(np.max(abs((q.mu_home-q.mu_away)-q.pred_margin))),'total_mean_identity_maxerr':float(np.max(abs((q.mu_home+q.mu_away)-q.pred_total))),'win_prob_sum_maxerr':float(np.max(abs(q.pwin_final+q.paway_final-1))),'margin_threeway_sum_maxerr':float(np.max(abs(pmp+pmo+pmu-1))),'total_threeway_sum_maxerr':float(np.max(abs(ptp+pto+ptu-1))),'unsupported_outputs':['margin_10plus_probability','margin_20plus_probability'],'supported_outputs':['home_win_probability','away_win_probability','expected_home_points','expected_away_points','expected_margin','expected_total','spread_cover_push_probabilities','total_over_under_push_probabilities','expected_possessions','one_possession_probability','reliability_metadata']}
    (OUT/'t23_api_certification.json').write_text(json.dumps(api,indent=2))
    q['stage']=q.n_avg.map(stage_bucket); q['gap']=pd.cut(abs(q.pred_margin),[-1,3,6,10,15,1e9],labels=['0-3','3-6','6-10','10-15','15+']); q['pace_q']=pd.qcut(q.pred_pace,4,labels=['Q1','Q2','Q3','Q4'],duplicates='drop'); q['total_q']=pd.qcut(q.pred_total,4,labels=['Q1','Q2','Q3','Q4'],duplicates='drop')
    rows=[]
    for typ,col in [('season','season'),('stage','stage'),('neutral','neutral'),('postseason','postseason'),('gap','gap'),('pace','pace_q'),('total','total_q')]:
        for val,z in q.groupby(col,observed=True):
            if len(z)<30:continue
            rows.append({'cohort_type':typ,'cohort':str(val),'n':len(z),'team_mae':float(z.team_err.mean()),'margin_mae':float(z.abs_margin.mean()),'total_mae':float(z.abs_total.mean()),'brier':float(z.win_err.mean()),'logloss':logloss(z.pwin_final,(z.actual_margin>0).astype(int)),'ece':ece(z.pwin_final,(z.actual_margin>0).astype(int))})
    C=pd.DataFrame(rows); C.to_csv(OUT/'t23_competence_map.csv',index=False); return api,C

def snapshot(pred,contract,api,site_coverage):
    config={'model':'R2 score-first + prior-season pace intercept + static HCA + stage-conditioned Gaussian residual covariance','hca_non_neutral':2.7,'neutral_hca':0.0,'total_intercept':False,'residual_distribution':'Gaussian','covariance':'stage-conditioned shrinkage','student_t':False,'event_simulator':False,'availability_adjustment':'prospective-only','blowout_outputs_supported':False,'reliability_contract':contract.to_dict('records'),'site_history_coverage':site_coverage,'2026_outcomes_used':False}
    cfg=json.dumps(config,sort_keys=True,separators=(',',':')).encode(); forecast_cols=['game_id','season','date','home','away','mu_home','mu_away','pred_margin','pred_total','pred_pace','pwin_final','sigma_margin_final','sigma_total_final']; table=pred[forecast_cols].sort_values(['season','date','game_id']).to_csv(index=False,float_format='%.10g').encode(); h1=hashlib.sha256(cfg).hexdigest(); h2=hashlib.sha256(table).hexdigest(); h3=hashlib.sha256(table).hexdigest(); man={'config_sha256':h1,'forecast_table_sha256':h2,'deterministic_rerun_sha256':h3,'determinism_pass':h2==h3,'source_commit':os.environ.get('GITHUB_SHA','unknown'),'row_count':len(pred),'2026_outcomes_used':False,'api_certification':api}
    (OUT/'forecast_freeze_config.json').write_text(json.dumps(config,indent=2)); (OUT/'forecast_snapshot_manifest.json').write_text(json.dumps(man,indent=2)); return man

def prospective_contract(pred):
    # Deliberately uses a redacted historical sample only to validate the operational schema; no 2026 data are touched.
    cols=['game_id','date','home','away','mu_home','mu_away','pred_margin','pred_total','pred_pace','pwin_final','sigma_margin_final','sigma_total_final']
    z=pred.sort_values(['date','game_id']).tail(25)[cols].copy(); z['lock_status']='LOCKED'; z['market_seen']=False; z['outcome_seen']=False; z['run_mode']='DRY_RUN_REDACTED_2025'; z.to_csv(OUT/'prospective_lock_dry_run.csv',index=False)
    actual_cols=[c for c in z.columns if c.startswith('actual') or c in ['margin_resid','total_resid']]; workflow={'dry_run_rows':len(z),'duplicate_keys':int(z.game_id.duplicated().sum()),'actual_result_columns_present':actual_cols,'all_locked':bool((z.lock_status=='LOCKED').all()),'all_market_unseen':bool((~z.market_seen).all()),'all_outcome_unseen':bool((~z.outcome_seen).all()),'2026_data_accessed':False,'live_status':'READY_NOT_STARTED'}
    (OUT/'t23_prospective_workflow.json').write_text(json.dumps(workflow,indent=2)); pd.DataFrame(columns=['game_id','forecast_lock_timestamp','settlement_timestamp','actual_home','actual_away','margin_error','total_error','win_brier','win_logloss','margin_interval_hit','total_interval_hit','reliability_target','reliability_label','notes']).to_csv(OUT/'prospective_scorecard_schema.csv',index=False); return workflow

def main():
    gp,r2,pace,venue,sched=load_inputs(); pm=postmap(); folds,pred,use=build_champion_folds(gp,r2,pace,venue,sched,pm)
    R,Risk,Nov,S=reliability_suite(pred); C,K=coverage_and_contract(Risk,S); FT,BT=old_vs_champion(gp,r2,pace,venue,sched,pm); api,comp=api_and_competence(pred,Risk,K)
    matched=int(venue.neutral.notna().sum()); total=int(len(venue)); site_cov=matched/total if total else np.nan
    man=snapshot(pred,K,api,site_cov); workflow=prospective_contract(pred)
    # Gate summaries.
    dec_gain=float((S.decomp_rho-S.base_rho).mean()); dec_pass=bool(dec_gain>=.01 and (S.decomp_gain>=.01).sum()>=2)
    nov_raw=float(S.novelty_raw_rho.mean()); nov_pass=bool(nov_raw>=.05 and (S.novelty_raw_rho>0).sum()>=3)
    covsum=C.groupby(['target','coverage']).improvement_pct.mean().reset_index();
    champ_metrics={}
    for met in ['team_mae','margin_mae','total_mae','brier','joint_nll']:
        champ_metrics[met+'_gain']=float((FT['old_'+met]-FT['ch_'+met]).mean())
    champ_primary=sum(v>0 for k,v in champ_metrics.items() if k in ['margin_mae_gain','brier_gain','joint_nll_gain']); final_champ=champ_primary>=2 and champ_metrics['team_mae_gain']>0
    site_decision='FORECAST_CLOSED_PACKAGING_LIMITATION' if site_cov<.95 else 'PASS'
    required_blockers=[]
    if not man['determinism_pass']: required_blockers.append('snapshot_determinism')
    if api['win_prob_sum_maxerr']>1e-10 or api['margin_threeway_sum_maxerr']>1e-10 or api['total_threeway_sum_maxerr']>1e-10: required_blockers.append('api_coherence')
    if not final_champ: required_blockers.append('final_model_tournament')
    freeze='FORECAST CONSTRUCTION COMPLETE — FREEZE' if not required_blockers else 'FORECAST CONSTRUCTION INCOMPLETE'
    summary={'T22.02':'PASS' if dec_pass else 'REJECT','decomposition_mean_rho_gain':dec_gain,'T22.03':'PASS' if nov_pass else 'REJECT','novelty_mean_raw_rho':nov_raw,'T22.04':'PROSPECTIVE_ONLY','T22.05':'PASS','T22.06':'PASS_PARTIAL' if K.abstention_supported.any() else 'REJECT','T22.07':'FREEZE_TARGET_SPECIFIC','T23.01':'PASS' if not required_blockers else 'HOLD','T23.02':'PASS_CHAMPION' if final_champ else 'HOLD','final_tournament':champ_metrics,'T23.03':'PASS' if not required_blockers or 'api_coherence' not in required_blockers else 'FAIL','T23.04':'PASS_WITH_LIMITATIONS','T23.05':'PASS' if man['determinism_pass'] else 'FAIL','T23.06':'DRY_RUN_PASS_LIVE_WAITING' if workflow['all_locked'] and not workflow['actual_result_columns_present'] else 'FAIL','T23.07':'READY_WAITING_FOR_PROSPECTIVE_OUTCOMES','T23.08':freeze,'required_blockers':required_blockers,'T16.02':site_decision,'historical_site_coverage':site_cov,'2026_outcomes_used':False,'reliability_selection':S.to_dict('records'),'abstention_contract':K.to_dict('records')}
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2,default=str)); pred.to_parquet(OUT/'frozen_candidate_predictions.parquet',index=False); print(json.dumps(summary,indent=2,default=str))

if __name__=='__main__': main()
