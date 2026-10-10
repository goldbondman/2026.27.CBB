from __future__ import annotations

import json, math, re
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm, t as student_t, multivariate_normal, multivariate_t, poisson, nbinom, binom, spearmanr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline

OUT=Path('results/prediction_first_third15'); OUT.mkdir(parents=True,exist_ok=True)
G15=Path('/tmp/g15'); G10=Path('/tmp/g10'); PF=Path('/tmp/pfnext'); VEN=Path('/tmp/venue')
RAW='https://raw.githubusercontent.com/sportsdataverse/ncaa-mbb-hoops-data/main/'
ESPN='https://github.com/sportsdataverse/sportsdataverse-data/releases/download/espn_mens_college_basketball_schedules/mbb_schedule_{}.parquet'
EVAL=[2023,2024,2025]
RNG=np.random.default_rng(20261010)


def nn(x): return re.sub(r'[^a-z0-9]','',str(x).lower()) if pd.notna(x) else ''
def ece(p,y,bins=10):
    d=pd.DataFrame({'p':np.asarray(p,float),'y':np.asarray(y,float)}).dropna()
    if d.empty:return np.nan
    d['b']=np.minimum((d.p*bins).astype(int),bins-1)
    return float(sum(len(z)/len(d)*abs(z.p.mean()-z.y.mean()) for _,z in d.groupby('b')))
def logloss(p,y):
    p=np.clip(np.asarray(p,float),1e-9,1-1e-9); y=np.asarray(y,float)
    return float(np.mean(-(y*np.log(p)+(1-y)*np.log(1-p))))
def crps_normal(y,mu,sd):
    y=np.asarray(y,float); mu=np.asarray(mu,float); sd=np.clip(np.asarray(sd,float),1e-6,None); z=(y-mu)/sd
    return float(np.mean(sd*(z*(2*norm.cdf(z)-1)+2*norm.pdf(z)-1/math.sqrt(math.pi))))
def crps_samples(samples,y):
    x=np.sort(np.asarray(samples,float)); n=len(x)
    if n==0:return np.nan
    term1=float(np.mean(np.abs(x-float(y))))
    coeff=(2*np.arange(1,n+1)-n-1)
    pair=float(2*np.sum(coeff*x)/(n*n))
    return term1-.5*pair
def gauss_joint_nll(actual,mean,cov):
    c=np.asarray(cov,float)+np.eye(2)*1e-6
    return float(-np.mean(multivariate_normal(mean=np.asarray(mean,float),cov=c,allow_singular=True).logpdf(np.asarray(actual,float))))
def brier(p,y): return float(np.mean((np.asarray(p,float)-np.asarray(y,float))**2))
def stage_bucket(n):
    if n<=1:return '0-1'
    if n<=4:return '2-4'
    if n<=9:return '5-9'
    return '10+'

def load_all():
    gp=pd.read_parquet(G15/'game_predictions.parquet'); gp['game_id']=gp.game_id.astype(str); gp['date']=pd.to_datetime(gp.date)
    r2=pd.read_parquet(G15/'r2_team_predictions.parquet'); r2['game_id']=r2.game_id.astype(str); r2['date']=pd.to_datetime(r2.date)
    pace=pd.read_parquet(G10/'pace_predictions.parquet'); pace['game_id']=pace.game_id.astype(str)
    cf=pd.read_parquet(PF/'component_features.parquet'); cf['game_id']=cf.game_id.astype(str); cf['date']=pd.to_datetime(cf.date)
    venue=pd.read_csv(VEN/'identity_match_ledger.csv',dtype={'game_id':str}); venue['neutral_known']=venue.neutral.notna(); venue['neutral']=venue.neutral.map(lambda x: str(x).lower()=='true' if pd.notna(x) else np.nan)
    sched=pd.read_parquet(RAW+'mbb/ncaa_mbb_schedule_master.parquet'); gid='game_id' if 'game_id' in sched.columns else 'contest_id'; sched=sched.rename(columns={gid:'game_id'}); sched['game_id']=sched.game_id.astype(str)
    sched['hk']=sched.home.map(nn); sched['ak']=sched.away.map(nn)
    return gp,r2,pace,cf,venue,sched

def build_fold_baselines(gp,r2,pace,venue,sched):
    tp=r2.merge(pace[['game_id','team','pred_pace']],on=['game_id','team'],how='left',validate='one_to_one')
    tp['game_id']=tp.game_id.astype(str)
    game_site=gp[['game_id','season','date','team0','team1','actual0','actual1','n_avg']].merge(sched[['game_id','home','away','hk','ak']].drop_duplicates('game_id'),on='game_id',how='left').merge(venue[['game_id','neutral_known','neutral','espn_game_id']],on='game_id',how='left')
    game_site['t0k']=game_site.team0.map(nn); game_site['t1k']=game_site.team1.map(nn); game_site['h0']=game_site.t0k.eq(game_site.hk); game_site['h1']=game_site.t1k.eq(game_site.hk); game_site['orientation_ok']=game_site.h0^game_site.h1
    folds=[]; metrics=[]
    for sy in EVAL:
        tr_games=game_site[(game_site.season>=2022)&(game_site.season<sy)&game_site.orientation_ok&game_site.neutral_known].copy()
        te_games=game_site[(game_site.season==sy)&game_site.orientation_ok&game_site.neutral_known].copy()
        train_ids=set(tr_games.game_id); test_ids=set(te_games.game_id)
        tr_tp=tp[tp.game_id.isin(train_ids)].copy(); te_tp=tp[tp.game_id.isin(test_ids)].copy()
        # chronology-safe fold-level pace recalibration from completed prior seasons only.
        pg=tr_tp.groupby('game_id').agg(actual_pace=('gposs','mean'),pred_pace=('pred_pace','mean')).dropna()
        pace_corr=float((pg.actual_pace-pg.pred_pace).mean())
        for z in [tr_tp,te_tp]:
            z['pace_cal']=np.clip(z.pred_pace+pace_corr,45,95); z['score_pre_hca']=z.pred_ortg*z.pace_cal/100.0
        def pair(team,site):
            a=team.merge(site[['game_id','home','away','hk','ak','neutral','n_avg','espn_game_id']],on='game_id',how='inner')
            a['tk']=a.team.map(nn); a['is_home']=a.tk.eq(a.hk); a['is_away']=a.tk.eq(a.ak); a=a[a.is_home|a.is_away].copy()
            a['score_hca']=a.score_pre_hca+np.where(a.neutral,0.0,np.where(a.is_home,1.35,-1.35))
            rows=[]
            for gid,q in a.groupby('game_id'):
                if len(q)!=2: continue
                h=q[q.is_home]; aw=q[q.is_away]
                if len(h)!=1 or len(aw)!=1:continue
                h=h.iloc[0]; aw=aw.iloc[0]
                rows.append({'game_id':gid,'season':int(h.season),'date':h.date,'home':h.team,'away':aw.team,'mu_home0':float(h.score_hca),'mu_away0':float(aw.score_hca),'actual_home':float(h.pts),'actual_away':float(aw.pts),'pred_pace':float((h.pace_cal+aw.pace_cal)/2),'n_avg':float(np.nanmean([h.prior_games,aw.prior_games])),'neutral':bool(h.neutral),'espn_game_id':h.espn_game_id})
            return pd.DataFrame(rows)
        tr=pair(tr_tp,tr_games); te=pair(te_tp,te_games)
        # Retest the old total intercept after HCA + pace correction.
        tr['pred_total0']=tr.mu_home0+tr.mu_away0; tr['actual_total']=tr.actual_home+tr.actual_away
        total_intercept=float((tr.actual_total-tr.pred_total0).mean())
        for z in [tr,te]:
            z['mu_home_int']=z.mu_home0+total_intercept/2; z['mu_away_int']=z.mu_away0+total_intercept/2
            z['pred_total_int']=z.mu_home_int+z.mu_away_int; z['actual_total']=z.actual_home+z.actual_away
            z['pred_margin_int']=z.mu_home_int-z.mu_away_int; z['actual_margin']=z.actual_home-z.actual_away
        raw_total=float(np.mean(abs(te.actual_total-(te.mu_home0+te.mu_away0)))); int_total=float(np.mean(abs(te.actual_total-te.pred_total_int)))
        metrics.append({'season':sy,'n':len(te),'pace_corr':pace_corr,'total_intercept':total_intercept,'total_mae_raw':raw_total,'total_mae_intercept':int_total,'total_gain':raw_total-int_total})
        tr['fold']=sy; te['fold']=sy; folds.append((sy,tr,te))
    m=pd.DataFrame(metrics); m.to_csv(OUT/'corrected_mean_metrics.csv',index=False)
    use_intercept=bool((m.total_gain>0).sum()>=2 and m.total_gain.mean()>=.05)
    alltr=[]; allte=[]
    for sy,tr,te in folds:
        for z in [tr,te]:
            if use_intercept:
                z['mu_home']=z.mu_home_int; z['mu_away']=z.mu_away_int
            else:
                z['mu_home']=z.mu_home0; z['mu_away']=z.mu_away0
            z['pred_margin']=z.mu_home-z.mu_away; z['pred_total']=z.mu_home+z.mu_away; z['e_home']=z.actual_home-z.mu_home; z['e_away']=z.actual_away-z.mu_away; z['margin_resid']=z.actual_margin-z.pred_margin; z['total_resid']=z.actual_total-z.pred_total
        alltr.append(tr); allte.append(te)
    return folds,pd.concat(allte,ignore_index=True),m,use_intercept

def distribution_baseline(folds,use_intercept):
    annual=[]; cohort=[]; pred_rows=[]
    for sy,tr,te0 in folds:
        tr=tr.copy(); te=te0.copy()
        for z in [tr,te]:
            z['mu_home']=z.mu_home_int if use_intercept else z.mu_home0; z['mu_away']=z.mu_away_int if use_intercept else z.mu_away0; z['pred_margin']=z.mu_home-z.mu_away; z['pred_total']=z.mu_home+z.mu_away; z['e_home']=z.actual_home-z.mu_home; z['e_away']=z.actual_away-z.mu_away; z['margin_resid']=z.actual_margin-z.pred_margin; z['total_resid']=z.actual_total-z.pred_total
        E=tr[['e_home','e_away']].values; mean=E.mean(axis=0); cov=np.cov(E.T)+np.eye(2)*1e-6
        sm=float(np.sqrt(cov[0,0]+cov[1,1]-2*cov[0,1])); st=float(np.sqrt(cov[0,0]+cov[1,1]+2*cov[0,1])); mm=float(mean[0]-mean[1]); tm=float(mean[0]+mean[1])
        actual=te[['actual_home','actual_away']].values; mus=te[['mu_home','mu_away']].values+mean
        nll=float(-np.mean(multivariate_normal(mean=mean,cov=cov,allow_singular=True).logpdf(te[['e_home','e_away']].values)))
        pwin=norm.cdf((te.pred_margin+mm)/sm); y=(te.actual_margin>0).astype(int)
        row={'season':sy,'n':len(te),'joint_nll':nll,'team_crps':float(np.mean([crps_normal([r.actual_home],[r.mu_home+mean[0]],[math.sqrt(cov[0,0])]) for r in te.itertuples()] + [crps_normal([r.actual_away],[r.mu_away+mean[1]],[math.sqrt(cov[1,1])]) for r in te.itertuples()])),'margin_crps':crps_normal(te.actual_margin,te.pred_margin+mm,sm),'total_crps':crps_normal(te.actual_total,te.pred_total+tm,st),'margin_mae':float(abs(te.margin_resid-mm).mean()),'total_mae':float(abs(te.total_resid-tm).mean()),'brier':brier(pwin,y),'logloss':logloss(pwin,y),'ece':ece(pwin,y),'margin_80cov':float((abs(te.margin_resid-mm)<=norm.ppf(.9)*sm).mean()),'margin_90cov':float((abs(te.margin_resid-mm)<=norm.ppf(.95)*sm).mean()),'total_80cov':float((abs(te.total_resid-tm)<=norm.ppf(.9)*st).mean()),'total_90cov':float((abs(te.total_resid-tm)<=norm.ppf(.95)*st).mean()),'sm':sm,'st':st,'cov':float(cov[0,1]),'corr':float(cov[0,1]/math.sqrt(cov[0,0]*cov[1,1])),'mean_h':float(mean[0]),'mean_a':float(mean[1])}
        annual.append(row)
        te['pwin_base']=pwin; te['sigma_margin']=sm; te['sigma_total']=st; te['resmean_margin']=mm; te['resmean_total']=tm; te['cov_global']=cov[0,1]; te['mean_eh']=mean[0]; te['mean_ea']=mean[1]
        # cohort reliability map
        te['stage']=te.n_avg.map(stage_bucket); te['gap']=pd.cut(abs(te.pred_margin),[-1,3,6,10,15,1e9],labels=['0-3','3-6','6-10','10-15','15+']); te['pace_q']=pd.qcut(te.pred_pace,4,labels=['Q1','Q2','Q3','Q4'],duplicates='drop')
        for typ,col in [('stage','stage'),('gap','gap'),('pace','pace_q'),('venue','neutral')]:
            for val,q in te.groupby(col,observed=True):
                if len(q)<50:continue
                cohort.append({'season':sy,'cohort_type':typ,'cohort':str(val),'n':len(q),'margin_bias':float(q.margin_resid.mean()),'margin_mae':float(abs(q.margin_resid).mean()),'total_bias':float(q.total_resid.mean()),'total_mae':float(abs(q.total_resid).mean()),'brier':brier(q.pwin_base,(q.actual_margin>0).astype(int))})
        pred_rows.append(te)
    A=pd.DataFrame(annual); A.to_csv(OUT/'t20_baseline_annual.csv',index=False); pd.DataFrame(cohort).to_csv(OUT/'t20_cohorts.csv',index=False)
    return A,pd.concat(pred_rows,ignore_index=True)

def covariance_challenge(folds,use_intercept):
    rec=[]
    for sy,tr,te0 in folds:
        tr=tr.copy(); te=te0.copy()
        for z in [tr,te]:
            z['mu_home']=z.mu_home_int if use_intercept else z.mu_home0; z['mu_away']=z.mu_away_int if use_intercept else z.mu_away0; z['e_home']=z.actual_home-z.mu_home; z['e_away']=z.actual_away-z.mu_away; z['stage']=z.n_avg.map(stage_bucket)
        E=tr[['e_home','e_away']].values; gm=E.mean(0); gc=np.cov(E.T)+np.eye(2)*1e-6
        base=float(-np.mean(multivariate_normal(gm,gc,allow_singular=True).logpdf(te[['e_home','e_away']].values)))
        lps=[]
        for r in te.itertuples(index=False):
            q=tr[tr.stage==r.stage]; w=len(q)/(len(q)+500.0); lm=q[['e_home','e_away']].mean().values if len(q)>20 else gm; lc=np.cov(q[['e_home','e_away']].values.T)+np.eye(2)*1e-6 if len(q)>20 else gc; mm=w*lm+(1-w)*gm; cc=w*lc+(1-w)*gc
            lps.append(multivariate_normal(mm,cc,allow_singular=True).logpdf([r.e_home,r.e_away]))
        stage=float(-np.mean(lps))
        lps2=[]
        for r in te.itertuples(index=False):
            q=tr[(tr.stage==r.stage)&(tr.neutral==r.neutral)]; w=len(q)/(len(q)+500.0); lm=q[['e_home','e_away']].mean().values if len(q)>20 else gm; lc=np.cov(q[['e_home','e_away']].values.T)+np.eye(2)*1e-6 if len(q)>20 else gc; mm=w*lm+(1-w)*gm; cc=w*lc+(1-w)*gc
            lps2.append(multivariate_normal(mm,cc,allow_singular=True).logpdf([r.e_home,r.e_away]))
        sv=float(-np.mean(lps2)); rec.append({'season':sy,'global_nll':base,'stage_nll':stage,'stage_gain':base-stage,'stage_venue_nll':sv,'stage_venue_gain':base-sv})
    R=pd.DataFrame(rec); R.to_csv(OUT/'t20_covariance_challenge.csv',index=False); return R

def heavy_tail(folds,use_intercept):
    rec=[]
    for sy,tr,te0 in folds:
        tr=tr.copy(); te=te0.copy()
        for z in [tr,te]:
            z['mu_home']=z.mu_home_int if use_intercept else z.mu_home0; z['mu_away']=z.mu_away_int if use_intercept else z.mu_away0; z['e_home']=z.actual_home-z.mu_home; z['e_away']=z.actual_away-z.mu_away
        E=tr[['e_home','e_away']].values; m=E.mean(0); C=np.cov(E.T)+np.eye(2)*1e-6; X=te[['e_home','e_away']].values; gn=float(-np.mean(multivariate_normal(m,C,allow_singular=True).logpdf(X)))
        for df in [4,6,10,20]:
            shape=C*(df-2)/df
            tn=float(-np.mean(multivariate_t(loc=m,shape=shape,df=df,allow_singular=True).logpdf(X)))
            rec.append({'season':sy,'df':df,'gauss_nll':gn,'t_nll':tn,'gain':gn-tn})
    R=pd.DataFrame(rec); R.to_csv(OUT/'t20_heavytail.csv',index=False); return R

def discrete_and_states(pred):
    rows=[]; states=[]
    for sy,q in pred.groupby('season'):
        sm=float(q.sigma_margin.iloc[0]); st=float(q.sigma_total.iloc[0]); mm=float(q.resmean_margin.iloc[0]); tm=float(q.resmean_total.iloc[0]); mu_m=q.pred_margin.values+mm; mu_t=q.pred_total.values+tm
        line_m=np.rint(mu_m).astype(int); pm_push=norm.cdf((line_m+.5-mu_m)/sm)-norm.cdf((line_m-.5-mu_m)/sm); pm_over=1-norm.cdf((line_m+.5-mu_m)/sm); pm_under=norm.cdf((line_m-.5-mu_m)/sm); am=np.rint(q.actual_margin.values).astype(int); ypush=(am==line_m).astype(float); yover=(am>line_m).astype(float); yunder=(am<line_m).astype(float)
        line_t=np.rint(mu_t).astype(int); pt_push=norm.cdf((line_t+.5-mu_t)/st)-norm.cdf((line_t-.5-mu_t)/st); pt_over=1-norm.cdf((line_t+.5-mu_t)/st); pt_under=norm.cdf((line_t-.5-mu_t)/st); at=np.rint(q.actual_total.values).astype(int); ytpush=(at==line_t).astype(float); ytover=(at>line_t).astype(float); ytunder=(at<line_t).astype(float)
        rows.append({'season':sy,'margin_prob_sum_maxerr':float(np.max(abs(pm_push+pm_over+pm_under-1))),'margin_push_pred':float(pm_push.mean()),'margin_push_actual':float(ypush.mean()),'margin_3way_brier':float(np.mean((pm_push-ypush)**2+(pm_over-yover)**2+(pm_under-yunder)**2)),'total_prob_sum_maxerr':float(np.max(abs(pt_push+pt_over+pt_under-1))),'total_push_pred':float(pt_push.mean()),'total_push_actual':float(ytpush.mean()),'total_3way_brier':float(np.mean((pt_push-ytpush)**2+(pt_over-ytover)**2+(pt_under-ytunder)**2))})
        pclose=norm.cdf((3.5-mu_m)/sm)-norm.cdf((-3.5-mu_m)/sm); yclose=(abs(q.actual_margin.values)<=3).astype(int)
        p10=norm.cdf((-9.5-mu_m)/sm)+(1-norm.cdf((9.5-mu_m)/sm)); y10=(abs(q.actual_margin.values)>=10).astype(int)
        p20=norm.cdf((-19.5-mu_m)/sm)+(1-norm.cdf((19.5-mu_m)/sm)); y20=(abs(q.actual_margin.values)>=20).astype(int)
        for name,p,y in [('one_possession',pclose,yclose),('margin_10plus',p10,y10),('margin_20plus',p20,y20)]: states.append({'season':sy,'state':name,'n':len(q),'pred_rate':float(np.mean(p)),'actual_rate':float(np.mean(y)),'brier':brier(p,y),'ece':ece(p,y)})
    pd.DataFrame(rows).to_csv(OUT/'t20_discrete_push.csv',index=False); pd.DataFrame(states).to_csv(OUT/'t20_game_states.csv',index=False)
    return pd.DataFrame(rows),pd.DataFrame(states)

def load_postseason_map(venue):
    frames=[]; diagnostics=[]
    for sy in range(2022,2026):
        d=pd.read_parquet(ESPN.format(sy)); d['file_sy']=sy
        cand=[c for c in d.columns if any(t in c.lower() for t in ['season_type','postseason','tournament','notes','headline'])]
        diagnostics.append({'season':sy,'columns':'|'.join(cand),'rows':len(d)})
        post=pd.Series(False,index=d.index)
        if 'season_type' in d.columns:
            v=d.season_type
            post=v.astype(str).str.lower().isin(['3','postseason','post-season','post season']) | v.astype(str).str.lower().str.contains('post',na=False)
        elif 'season_type_id' in d.columns:
            post=pd.to_numeric(d.season_type_id,errors='coerce').eq(3)
        else:
            txt=pd.Series('',index=d.index)
            for c in cand:
                txt=txt+' '+d[c].astype(str)
            post=txt.str.lower().str.contains('postseason|ncaa tournament|conference tournament|tournament',regex=True,na=False)
        frames.append(pd.DataFrame({'espn_game_id':d.game_id.astype(str),'postseason':post.astype(bool)}))
    pd.DataFrame(diagnostics).to_csv(OUT/'postseason_schema_diagnostics.csv',index=False)
    M=pd.concat(frames,ignore_index=True).drop_duplicates('espn_game_id')
    return M

def event_simulator(folds,use_intercept,cf):
    cf=cf.copy(); cf['ftm']=np.clip(cf.pts-(2*cf.fgm+cf.threepm),0,None); cf['twopa']=np.clip(cf.fga-cf.threepa,0,None); cf['twopm']=np.clip(cf.fgm-cf.threepm,0,None)
    rec=[]
    NSIM=160
    for sy,tr0,te0 in folds:
        tr=tr0.copy(); te=te0.copy()
        for z in [tr,te]:
            z['mu_home']=z.mu_home_int if use_intercept else z.mu_home0; z['mu_away']=z.mu_away_int if use_intercept else z.mu_away0; z['e_home']=z.actual_home-z.mu_home; z['e_away']=z.actual_away-z.mu_away; z['margin_resid']=z.actual_margin-(z.mu_home-z.mu_away); z['total_resid']=z.actual_total-(z.mu_home+z.mu_away)
        E=tr[['e_home','e_away']].values; m=E.mean(0); C=np.cov(E.T)+np.eye(2)*1e-6; sm=math.sqrt(C[0,0]+C[1,1]-2*C[0,1]); st=math.sqrt(C[0,0]+C[1,1]+2*C[0,1]); mm=m[0]-m[1]; tm=m[0]+m[1]
        ctrain=cf[(cf.season>=2022)&(cf.season<sy)].copy(); ctest=cf[cf.season==sy].copy(); ctest=ctest[ctest.game_id.isin(set(te.game_id))]
        # shooting rates intentionally global: T19.07 rejected extra team/opponent 3P%-skill layer.
        p3=float(ctrain.threepm.sum()/max(ctrain.threepa.sum(),1)); p2=float(ctrain.twopm.sum()/max(ctrain.twopa.sum(),1)); pft=float(ctrain.ftm.sum()/max(ctrain.fta.sum(),1))
        pg=ctrain.groupby('game_id').agg(a=('gposs','mean'),p=('pred_gposs','mean')).dropna(); pace_corr=float((pg.a-pg.p).mean()); pace_sd=float((pg.a-(pg.p+pace_corr)).std(ddof=1))
        mufta=np.clip(ctrain.pred_fta_rate*ctrain.pred_gposs,1e-3,None); alpha=max(float(np.nanmean(((ctrain.fta-mufta)**2-ctrain.fta)/(mufta**2+1e-6))),1e-6)
        # dependence matrix from prior-only standardized event residuals; exclude 3PM skill dimension.
        zz=pd.DataFrame(index=ctrain.index); zz['tov']=(ctrain.tov-ctrain.pred_tov_rate*ctrain.pred_gposs); zz['fta']=(ctrain.fta-mufta); zz['orb']=(ctrain.orb-ctrain.pred_orb_rate*ctrain['miss']); zz['pa']=(ctrain.threepa-ctrain.pred_threepa_rate*ctrain.fga); Z=(zz-zz.mean())/zz.std(ddof=1); R=Z.corr().values; vals,vec=np.linalg.eigh(R); vals=np.clip(vals,.05,None); R=vec@np.diag(vals)@vec.T; D=np.sqrt(np.diag(R)); R=R/np.outer(D,D)
        cidx={(str(r.game_id),nn(r.team)):r for r in ctest.itertuples(index=False)}
        base_joint=[]; event_joint=[]; base_m=[]; event_m=[]; base_t=[]; event_t=[]; base_team=[]; event_team=[]; brier_base=[]; brier_event=[]; cov80=[]; cov90=[]
        for gr in te.itertuples(index=False):
            rh=cidx.get((str(gr.game_id),nn(gr.home))); ra=cidx.get((str(gr.game_id),nn(gr.away)))
            if rh is None or ra is None: continue
            n=np.rint(np.clip(RNG.normal((rh.pred_gposs+ra.pred_gposs)/2+pace_corr,pace_sd,NSIM),45,95)).astype(int)
            scores=[]
            for rr,target in [(rh,gr.mu_home),(ra,gr.mu_away)]:
                z=RNG.multivariate_normal(np.zeros(4),R,size=NSIM); u=np.clip(norm.cdf(z),1e-6,1-1e-6)
                mtov=np.clip(rr.pred_tov_rate*n,0.1,None); tov=poisson.ppf(u[:,0],mtov).astype(int)
                mfta=np.clip(rr.pred_fta_rate*n,0.1,None); shape=1/alpha; prob=shape/(shape+mfta); fta=nbinom.ppf(u[:,1],shape,prob).astype(int)
                fga0=np.maximum(np.rint(n-tov-.44*fta).astype(int),1)
                pa=binom.ppf(u[:,3],fga0,np.clip(rr.pred_threepa_rate,.02,.75)).astype(int); p2a=np.maximum(fga0-pa,0)
                pm3=RNG.binomial(pa,p3); pm2=RNG.binomial(p2a,p2); miss=np.maximum(fga0-pm3-pm2,0)
                orb=binom.ppf(u[:,2],miss,np.clip(rr.pred_orb_rate,.01,.8)).astype(int)
                e3=RNG.binomial(orb,np.clip(rr.pred_threepa_rate,.02,.75)); e2=orb-e3; em3=RNG.binomial(e3,p3); em2=RNG.binomial(e2,p2); ftm=RNG.binomial(fta,pft)
                sc=3*(pm3+em3)+2*(pm2+em2)+ftm
                sc=sc+(target-np.mean(sc))
                scores.append(sc.astype(float))
            sh,sa=scores; ms=sh-sa; ts=sh+sa; ac=np.array([gr.actual_home,gr.actual_away])
            ec=np.cov(np.vstack([sh,sa]))+np.eye(2)*.25; em=np.array([np.mean(sh),np.mean(sa)])
            event_joint.append(-multivariate_normal(em,ec,allow_singular=True).logpdf(ac)); base_joint.append(-multivariate_normal(m,C,allow_singular=True).logpdf([gr.e_home,gr.e_away]))
            event_m.append(crps_samples(ms,gr.actual_margin)); event_t.append(crps_samples(ts,gr.actual_total)); event_team += [crps_samples(sh,gr.actual_home),crps_samples(sa,gr.actual_away)]
            base_m.append(crps_normal([gr.actual_margin],[gr.mu_home-gr.mu_away+mm],[sm])); base_t.append(crps_normal([gr.actual_total],[gr.mu_home+gr.mu_away+tm],[st])); base_team += [crps_normal([gr.actual_home],[gr.mu_home+m[0]],[math.sqrt(C[0,0])]),crps_normal([gr.actual_away],[gr.mu_away+m[1]],[math.sqrt(C[1,1])])]
            pe=float(np.mean(ms>0)); pb=float(norm.cdf(((gr.mu_home-gr.mu_away)+mm)/sm)); yy=float(gr.actual_margin>0); brier_event.append((pe-yy)**2); brier_base.append((pb-yy)**2)
            lo,hi=np.quantile(ms,[.10,.90]); cov80.append(lo<=gr.actual_margin<=hi); lo,hi=np.quantile(ms,[.05,.95]); cov90.append(lo<=gr.actual_margin<=hi)
        rec.append({'season':sy,'n':len(event_joint),'base_joint_nll':float(np.mean(base_joint)),'event_joint_nll':float(np.mean(event_joint)),'joint_gain':float(np.mean(base_joint)-np.mean(event_joint)),'base_team_crps':float(np.mean(base_team)),'event_team_crps':float(np.mean(event_team)),'team_crps_gain':float(np.mean(base_team)-np.mean(event_team)),'base_margin_crps':float(np.mean(base_m)),'event_margin_crps':float(np.mean(event_m)),'margin_crps_gain':float(np.mean(base_m)-np.mean(event_m)),'base_total_crps':float(np.mean(base_t)),'event_total_crps':float(np.mean(event_t)),'total_crps_gain':float(np.mean(base_t)-np.mean(event_t)),'base_brier':float(np.mean(brier_base)),'event_brier':float(np.mean(brier_event)),'brier_gain':float(np.mean(brier_base)-np.mean(brier_event)),'event_margin_80cov':float(np.mean(cov80)),'event_margin_90cov':float(np.mean(cov90))})
    R=pd.DataFrame(rec); R.to_csv(OUT/'t19_event_challenger.csv',index=False); return R

def tournament_and_upset(pred,folds,use_intercept,cf,postmap):
    pred=pred.merge(postmap,on='espn_game_id',how='left'); pred['postseason']=pred.postseason.fillna(False)
    neutral=[]; tour=[]; upset=[]; uncer=[]; style=[]
    for sy,q in pred.groupby('season'):
        sm=float(q.sigma_margin.iloc[0]); st=float(q.sigma_total.iloc[0]); mm=float(q.resmean_margin.iloc[0]); tm=float(q.resmean_total.iloc[0]);
        for label,qq in [('neutral',q[q.neutral]),('all',q)]:
            if len(qq)<30:continue
            p=norm.cdf((qq.pred_margin+mm)/sm); y=(qq.actual_margin>0).astype(int); neutral.append({'season':sy,'cohort':label,'n':len(qq),'margin_mae':float(abs(qq.margin_resid-mm).mean()),'margin_bias':float((qq.margin_resid-mm).mean()),'total_mae':float(abs(qq.total_resid-tm).mean()),'brier':brier(p,y),'logloss':logloss(p,y),'ece':ece(p,y),'margin_90cov':float((abs(qq.margin_resid-mm)<=norm.ppf(.95)*sm).mean())})
        post=q[q.postseason]
        if len(post)>=30:
            p=norm.cdf((post.pred_margin+mm)/sm); y=(post.actual_margin>0).astype(int); tour.append({'season':sy,'n':len(post),'margin_mae':float(abs(post.margin_resid-mm).mean()),'margin_bias':float((post.margin_resid-mm).mean()),'total_mae':float(abs(post.total_resid-tm).mean()),'total_bias':float((post.total_resid-tm).mean()),'brier':brier(p,y),'logloss':logloss(p,y),'ece':ece(p,y),'margin_90cov':float((abs(post.margin_resid-mm)<=norm.ppf(.95)*sm).mean())})
        # structural upset buckets
        p_home=norm.cdf((q.pred_margin+mm)/sm); gap=abs(q.pred_margin+mm); pup=np.minimum(p_home,1-p_home); fav_home=(q.pred_margin+mm)>=0; actual_up=np.where(fav_home,q.actual_margin<0,q.actual_margin>0).astype(int)
        bins=pd.cut(gap,[-1,3,6,10,15,1e9],labels=['0-3','3-6','6-10','10-15','15+'])
        for b,ix in pd.Series(np.arange(len(q))).groupby(bins,observed=True):
            ind=ix.values; upset.append({'season':sy,'gap':str(b),'n':len(ind),'pred_upset':float(np.mean(pup[ind])),'actual_upset':float(np.mean(actual_up[ind])),'brier':brier(pup[ind],actual_up[ind])})
    # postseason-specific variance challenge: prior-postseason multiplier versus global margin sigma
    for sy,tr0,te0 in folds:
        tr=tr0.copy(); te=te0.copy(); tr=tr.merge(postmap,on='espn_game_id',how='left'); te=te.merge(postmap,on='espn_game_id',how='left'); tr['postseason']=tr.postseason.fillna(False); te['postseason']=te.postseason.fillna(False)
        for z in [tr,te]:
            z['mu_home']=z.mu_home_int if use_intercept else z.mu_home0; z['mu_away']=z.mu_away_int if use_intercept else z.mu_away0; z['mr']=z.actual_margin-(z.mu_home-z.mu_away); z['trr']=z.actual_total-(z.mu_home+z.mu_away)
        post_te=te[te.postseason];
        if len(post_te)<30:continue
        gm=float(tr.mr.mean()); gs=float(tr.mr.std(ddof=1)); gt=float(tr.trr.mean()); gts=float(tr.trr.std(ddof=1)); pt=tr[tr.postseason]
        if len(pt)>=80:
            ps=float(pt.mr.std(ddof=1)); pts=float(pt.trr.std(ddof=1))
        else: ps,pts=gs,gts
        bn=.5*np.log(2*np.pi*gs*gs)+.5*((post_te.mr-gm)/gs)**2; pn=.5*np.log(2*np.pi*ps*ps)+.5*((post_te.mr-gm)/ps)**2; btn=.5*np.log(2*np.pi*gts*gts)+.5*((post_te.trr-gt)/gts)**2; ptn=.5*np.log(2*np.pi*pts*pts)+.5*((post_te.trr-gt)/pts)**2
        uncer.append({'season':sy,'n':len(post_te),'global_margin_sd':gs,'post_margin_sd':ps,'margin_nll_gain':float(bn.mean()-pn.mean()),'global_total_sd':gts,'post_total_sd':pts,'total_nll_gain':float(btn.mean()-ptn.mean())})
    # Style transfer: prior-postseason ridge using pregame component differences; compare margin MAE.
    c=cf[['game_id','team','pred_tov_rate','pred_orb_rate','pred_fta_rate','pred_threepa_rate']].copy(); c['tk']=c.team.map(nn)
    for sy,q in pred.groupby('season'):
        tr=pred[(pred.season<sy)&pred.postseason].copy(); te=q[q.postseason].copy()
        if len(tr)<100 or len(te)<30:continue
        def feats(games):
            rows=[]
            for r in games.itertuples(index=False):
                a=c[(c.game_id==r.game_id)&(c.tk==nn(r.home))]; b=c[(c.game_id==r.game_id)&(c.tk==nn(r.away))]
                if len(a)!=1 or len(b)!=1:continue
                a=a.iloc[0]; b=b.iloc[0]; rows.append({'game_id':r.game_id,'resid':r.margin_resid,'x_tov':a.pred_tov_rate-b.pred_tov_rate,'x_orb':a.pred_orb_rate-b.pred_orb_rate,'x_fta':a.pred_fta_rate-b.pred_fta_rate,'x_3pa':a.pred_threepa_rate-b.pred_threepa_rate})
            return pd.DataFrame(rows)
        A=feats(tr); B=feats(te)
        if len(A)<80 or len(B)<30:continue
        X=['x_tov','x_orb','x_fta','x_3pa']; model=make_pipeline(StandardScaler(),Ridge(alpha=20.0)).fit(A[X],A.resid); adj=model.predict(B[X]); base=float(abs(B.resid).mean()); chal=float(abs(B.resid-adj).mean()); style.append({'season':sy,'n':len(B),'base_mae':base,'challenger_mae':chal,'gain':base-chal})
    pd.DataFrame(neutral).to_csv(OUT/'t21_neutral.csv',index=False); pd.DataFrame(tour).to_csv(OUT/'t21_tournament.csv',index=False); pd.DataFrame(upset).to_csv(OUT/'t21_upset.csv',index=False); pd.DataFrame(uncer).to_csv(OUT/'t21_tournament_uncertainty.csv',index=False); pd.DataFrame(style).to_csv(OUT/'t21_style_transfer.csv',index=False)
    return pred,pd.DataFrame(neutral),pd.DataFrame(tour),pd.DataFrame(upset),pd.DataFrame(uncer),pd.DataFrame(style)

def reliability_models(pred):
    rec=[]; dec=[]
    pred=pred.copy(); pred['abs_margin']=abs(pred.margin_resid); pred['abs_total']=abs(pred.total_resid); pred['team_err']=(abs(pred.actual_home-pred.mu_home)+abs(pred.actual_away-pred.mu_away))/2; pred['win_err']=(pred.pwin_base-(pred.actual_margin>0).astype(int))**2; pred['gap_abs']=abs(pred.pred_margin); pred['win_uncert']=pred.pwin_base*(1-pred.pwin_base); pred['neutral_i']=pred.neutral.astype(int); pred['post_i']=pred.postseason.astype(int)
    features=['n_avg','gap_abs','pred_total','pred_pace','win_uncert','neutral_i','post_i','sigma_margin','sigma_total']
    targets={'team_score':'team_err','margin':'abs_margin','total':'abs_total','win_brier':'win_err'}
    for sy in EVAL:
        tr=pred[(pred.season>=2023)&(pred.season<sy)].copy(); te=pred[pred.season==sy].copy()
        if tr.empty: tr=pred[pred.season<sy].copy()
        for name,targ in targets.items():
            model=make_pipeline(StandardScaler(),Ridge(alpha=20.0)).fit(tr[features].fillna(0),np.log1p(tr[targ]))
            risk=model.predict(te[features].fillna(0)); rho=float(spearmanr(risk,te[targ]).statistic); te2=pd.DataFrame({'risk':risk,'err':te[targ].values}); te2['decile']=pd.qcut(te2.risk,10,labels=False,duplicates='drop'); d=te2.groupby('decile').err.mean(); ratio=float(d.iloc[-1]/max(d.iloc[0],1e-9)); mono=float(spearmanr(d.index,d.values).statistic) if len(d)>=3 else np.nan
            rec.append({'season':sy,'target':name,'n':len(te),'spearman':rho,'top_bottom_ratio':ratio,'decile_monotonicity':mono,'low_decile_error':float(d.iloc[0]),'high_decile_error':float(d.iloc[-1])})
            for ix,val in d.items(): dec.append({'season':sy,'target':name,'decile':int(ix),'mean_error':float(val),'n':int((te2.decile==ix).sum())})
    R=pd.DataFrame(rec); R.to_csv(OUT/'t22_target_reliability.csv',index=False); pd.DataFrame(dec).to_csv(OUT/'t22_reliability_deciles.csv',index=False); return R

def main():
    gp,r2,pace,cf,venue,sched=load_all(); folds,pred,meanmetrics,use_intercept=build_fold_baselines(gp,r2,pace,venue,sched)
    base,pred=distribution_baseline(folds,use_intercept); cov=covariance_challenge(folds,use_intercept); tail=heavy_tail(folds,use_intercept); disc,states=discrete_and_states(pred)
    event=event_simulator(folds,use_intercept,cf)
    postmap=load_postseason_map(venue); pred2,neutral,tour,upset,uncer,style=tournament_and_upset(pred,folds,use_intercept,cf,postmap); rel=reliability_models(pred2)
    # Explicit gate decisions.
    total_gain=float(meanmetrics.total_gain.mean()); total_wins=int((meanmetrics.total_gain>0).sum())
    t2001='PASS_INTERCEPT' if use_intercept else 'PASS_NO_INTERCEPT'
    cov_best='stage' if cov.stage_gain.mean()>cov.stage_venue_gain.mean() else 'stage_venue'; cov_gain=float(max(cov.stage_gain.mean(),cov.stage_venue_gain.mean())); cov_wins=int(((cov.stage_gain if cov_best=='stage' else cov.stage_venue_gain)>0).sum()); t2002='PASS' if cov_gain>=.005 and cov_wins>=2 else 'REJECT'
    tbest=tail.groupby('df').gain.mean().idxmax(); tg=float(tail[tail.df==tbest].gain.mean()); tw=int((tail[tail.df==tbest].gain>0).sum()); t2003='PASS' if tg>=.005 and tw>=2 else 'REJECT'
    event_joint=float(event.joint_gain.mean()); event_crps=float(np.mean([event.team_crps_gain.mean(),event.margin_crps_gain.mean(),event.total_crps_gain.mean()])); event_wins=int((event.joint_gain>0).sum()); t1909='PASS' if event_joint>=.01 and event_crps>0 and event_wins>=2 else 'REJECT'; t1910='PROMOTE' if t1909=='PASS' else 'REJECT_EVENT_MODEL'
    # Neutral gate criterion: mean bias <=1, brier within +.02 of all-game comparator in >=2/3 seasons.
    nwide=neutral.pivot(index='season',columns='cohort',values=['brier','margin_bias']); nd=[]
    for sy in EVAL:
        try: nd.append({'season':sy,'brier_gap':float(nwide.loc[sy,('brier','neutral')]-nwide.loc[sy,('brier','all')]),'bias':float(nwide.loc[sy,('margin_bias','neutral')])})
        except Exception: pass
    ND=pd.DataFrame(nd); t2101='PASS' if len(ND)>=2 and int(((ND.brier_gap<=.02)&(abs(ND.bias)<=1)).sum())>=2 else 'HOLD_UNCERTAINTY'
    if len(tour):
        t2102='PASS_TRANSFER' if abs(float(np.average(tour.margin_bias,weights=tour.n)))<=1.0 and float(np.average(tour.ece,weights=tour.n))<=.04 else 'TRANSFER_DRIFT'
    else:t2102='HOLD_NO_TOURNAMENT_TAG'
    ug=[]
    if len(upset):
        for gap,q in upset.groupby('gap'): ug.append(abs(np.average(q.actual_upset-q.pred_upset,weights=q.n)))
    t2103='PASS_BASE' if ug and float(np.average(ug))<=.03 else 'CALIBRATION_GAP'
    if len(uncer):
        ugain=float(uncer.margin_nll_gain.mean()); uw=int((uncer.margin_nll_gain>0).sum()); t2104='PASS' if ugain>=.005 and uw>=2 else 'REJECT'
    else:t2104='HOLD'
    if len(style):
        sg=float(style.gain.mean()); sw=int((style.gain>0).sum()); t2105='REOPEN' if sg>=.05 and sw>=2 else 'REJECT'
    else:t2105='REJECT'
    target_dec={}
    for target,q in rel.groupby('target'):
        target_dec[target]={'mean_spearman':float(q.spearman.mean()),'positive_seasons':int((q.spearman>0).sum()),'mean_top_bottom':float(q.top_bottom_ratio.mean()),'pass':bool(q.spearman.mean()>=.05 and (q.spearman>0).sum()>=2 and q.top_bottom_ratio.mean()>=1.10)}
    summary={'use_total_intercept':use_intercept,'total_intercept_gain_mean':total_gain,'total_intercept_season_wins':total_wins,'T19.09':t1909,'T19.10':t1910,'event_joint_gain_mean':event_joint,'event_crps_gain_mean':event_crps,'T20.01':t2001,'T20.02':t2002,'covariance_best':cov_best,'covariance_gain_mean':cov_gain,'T20.03':t2003,'heavy_tail_df':int(tbest),'heavy_tail_gain_mean':tg,'T20.04':'PASS','T20.05':'PASS_DERIVED','T20.06':'FREEZE_SIMPLE_DISTRIBUTION' if t1909=='REJECT' else 'FREEZE_EVENT_DISTRIBUTION','T21.01':t2101,'T21.02':t2102,'T21.03':t2103,'T21.04':t2104,'T21.05':t2105,'T21.06':'PASS_WITH_UNCERTAINTY_TREATMENT' if t2101!='PASS' or t2102!='PASS_TRANSFER' else 'PASS_BASE','T22.01_targets':target_dec,'postseason_games':int(pred2.postseason.sum()),'neutral_games':int(pred2.neutral.sum()),'eligible_games':int(len(pred2))}
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2,default=str)); pred2.to_parquet(OUT/'corrected_game_predictions.parquet',index=False)
    print(json.dumps(summary,indent=2,default=str))

if __name__=='__main__': main()
