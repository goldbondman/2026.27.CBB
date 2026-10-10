from __future__ import annotations
import json, math, numpy as np, pandas as pd
from scipy.stats import multivariate_normal, multivariate_t, norm
import prediction_first_third15 as m

OUT=m.OUT

def clean_id(x):
    if pd.isna(x): return np.nan
    s=str(x).strip(); return s[:-2] if s.endswith('.0') else s

def load_all():
    gp=pd.read_parquet(m.G15/'game_predictions.parquet'); gp['game_id']=gp.game_id.astype(str); gp['date']=pd.to_datetime(gp.date)
    r2=pd.read_parquet(m.G15/'r2_team_predictions.parquet'); r2['game_id']=r2.game_id.astype(str); r2['date']=pd.to_datetime(r2.date)
    pace=pd.read_parquet(m.G10/'pace_predictions.parquet'); pace['game_id']=pace.game_id.astype(str)
    if 'pred_pace' not in pace.columns:
        p=pace[['game_id','pace_pred']].drop_duplicates('game_id').rename(columns={'pace_pred':'pred_pace'}); pace=r2[['game_id','team']].merge(p,on='game_id',how='left',validate='many_to_one')
    cf=pd.read_parquet(m.PF/'component_features.parquet'); cf['game_id']=cf.game_id.astype(str); cf['date']=pd.to_datetime(cf.date)
    venue=pd.read_csv(m.VEN/'identity_match_ledger.csv',dtype={'game_id':str}); venue['neutral_known']=venue.neutral.notna(); venue['neutral']=venue.neutral.map(lambda x:str(x).lower()=='true' if pd.notna(x) else np.nan); venue['espn_game_id']=venue.espn_game_id.map(clean_id)
    sched=pd.read_parquet(m.RAW+'mbb/ncaa_mbb_schedule_master.parquet'); gid='game_id' if 'game_id' in sched.columns else 'contest_id'; sched=sched.rename(columns={gid:'game_id'}); sched['game_id']=sched.game_id.astype(str); sched['hk']=sched.home.map(m.nn); sched['ak']=sched.away.map(m.nn)
    return gp,r2,pace,cf,venue,sched

def postmap():
    xs=[]
    for sy in range(2022,2026):
        d=pd.read_parquet(m.ESPN.format(sy)); v=d['season_type'] if 'season_type' in d.columns else pd.Series('',index=d.index); post=v.astype(str).str.lower().isin(['3','postseason','post-season','post season']) | v.astype(str).str.lower().str.contains('post',na=False)
        xs.append(pd.DataFrame({'espn_game_id':d.game_id.map(clean_id),'postseason':post.astype(bool)}))
    return pd.concat(xs,ignore_index=True).drop_duplicates('espn_game_id')

def prep(folds,use_intercept,pm):
    out=[]
    for sy,tr0,te0 in folds:
        tr=tr0.copy(); te=te0.copy()
        for z in (tr,te):
            z['mu_home']=z.mu_home_int if use_intercept else z.mu_home0; z['mu_away']=z.mu_away_int if use_intercept else z.mu_away0; z['e_home']=z.actual_home-z.mu_home; z['e_away']=z.actual_away-z.mu_away; z['mr']=z.actual_margin-(z.mu_home-z.mu_away); z['trr']=z.actual_total-(z.mu_home+z.mu_away); z['stage']=z.n_avg.map(m.stage_bucket); z['espn_game_id']=z.espn_game_id.map(clean_id)
        tr=tr.merge(pm,on='espn_game_id',how='left'); te=te.merge(pm,on='espn_game_id',how='left'); tr['postseason']=tr.postseason.fillna(False); te['postseason']=te.postseason.fillna(False); out.append((sy,tr,te))
    return out

def logpdf_context(r,tr,dist='gauss',df=10,context='global'):
    E=tr[['e_home','e_away']].values; gm=E.mean(0); gc=np.cov(E.T)+np.eye(2)*1e-6
    if context=='global': mm,cc=gm,gc
    else:
        if context=='stage': q=tr[tr.stage==r.stage]
        else: q=tr[(tr.stage==r.stage)&(tr.neutral==r.neutral)]
        w=len(q)/(len(q)+500.0)
        if len(q)>20:
            lm=q[['e_home','e_away']].mean().values; lc=np.cov(q[['e_home','e_away']].values.T)+np.eye(2)*1e-6
            mm=w*lm+(1-w)*gm; cc=w*lc+(1-w)*gc
        else:mm,cc=gm,gc
    x=[r.e_home,r.e_away]
    if dist=='gauss': return multivariate_normal(mm,cc,allow_singular=True).logpdf(x)
    shape=cc*(df-2)/df; return multivariate_t(loc=mm,shape=shape,df=df,allow_singular=True).logpdf(x)

def combined_tournament(folds):
    rows=[]
    variants=[('global_gauss','global','gauss',10),('stage_gauss','stage','gauss',10),('stagevenue_gauss','stagevenue','gauss',10),('global_t10','global','t',10),('stagevenue_t10','stagevenue','t',10),('stagevenue_t20','stagevenue','t',20)]
    for sy,tr,te in folds:
        for name,ctx,dist,df in variants:
            nll=-np.mean([logpdf_context(r,tr,dist,df,ctx) for r in te.itertuples(index=False)])
            rows.append({'season':sy,'variant':name,'nll':float(nll)})
    R=pd.DataFrame(rows); R.to_csv(OUT/'t20_combined_tournament.csv',index=False); return R

def postseason_treatment(folds):
    rows=[]
    for sy,tr,te in folds:
        q=te[te.postseason]; pt=tr[tr.postseason]
        if len(q)<30 or len(pt)<50: continue
        gm=float(tr.mr.mean()); gs=float(tr.mr.std(ddof=1)); pm=float(pt.mr.mean()); ps=float(pt.mr.std(ddof=1)); gt=float(tr.trr.mean()); gts=float(tr.trr.std(ddof=1)); ptm=float(pt.trr.mean()); pts=float(pt.trr.std(ddof=1))
        for target,res,base_mean,base_sd,post_mean,post_sd in [('margin',q.mr,gm,gs,pm,ps),('total',q.trr,gt,gts,ptm,pts)]:
            for var,mu,sd in [('global',base_mean,base_sd),('post_mean',post_mean,base_sd),('post_sd',base_mean,post_sd),('post_mean_sd',post_mean,post_sd)]:
                nll=float(np.mean(.5*np.log(2*np.pi*sd*sd)+.5*((res-mu)/sd)**2)); mae=float(np.mean(abs(res-mu))); bias=float(np.mean(res-mu)); rows.append({'season':sy,'target':target,'variant':var,'n':len(q),'train_post_n':len(pt),'train_post_mean':post_mean,'train_post_sd':post_sd,'nll':nll,'mae':mae,'bias':bias})
    R=pd.DataFrame(rows); R.to_csv(OUT/'t21_postseason_treatment_tournament.csv',index=False); return R

def upset_fixed(folds):
    rows=[]
    bins=[(-1,3,'0-3'),(3,6,'3-6'),(6,10,'6-10'),(10,15,'10-15'),(15,1e9,'15+')]
    for sy,tr,q in folds:
        mm=float(tr.mr.mean()); sm=float(tr.mr.std(ddof=1)); mu=q.mu_home-q.mu_away+mm; p_home=norm.cdf(mu/sm); pup=np.minimum(p_home,1-p_home); fav=mu>=0; y=np.where(fav,q.actual_margin.values<0,q.actual_margin.values>0).astype(int); gap=np.abs(mu)
        for lo,hi,label in bins:
            mask=(gap>lo)&(gap<=hi); n=int(mask.sum())
            if n==0:continue
            pp=pup[mask]; yy=y[mask]; rows.append({'season':sy,'gap':label,'n':n,'pred_upset':float(pp.mean()),'actual_upset':float(yy.mean()),'cal_gap':float(yy.mean()-pp.mean()),'brier':float(np.mean((pp-yy)**2))})
    R=pd.DataFrame(rows); R.to_csv(OUT/'t21_upset_fixed.csv',index=False); return R

def main():
    gp,r2,pace,cf,venue,sched=load_all(); rawfolds,pred,meanmetrics,use=m.build_fold_baselines(gp,r2,pace,venue,sched); pm=postmap(); folds=prep(rawfolds,use,pm)
    C=combined_tournament(folds); P=postseason_treatment(folds); U=upset_fixed(folds)
    cs=C.groupby('variant').nll.mean().sort_values(); base=float(cs['global_gauss']); best=cs.index[0]; gains={k:float(base-v) for k,v in cs.items()}; wins={v:int(((C[C.variant==v].set_index('season').nll)<(C[C.variant=='global_gauss'].set_index('season').nll)).sum()) for v in C.variant.unique()}
    ps={}
    for targ,q in P.groupby('target'):
        means=q.groupby('variant').nll.mean().sort_values(); b=float(means['global']); bv=means.index[0]; ps[targ]={'best':bv,'nll_gain':float(b-means.iloc[0]),'season_wins':int((q[q.variant==bv].set_index('season').nll<q[q.variant=='global'].set_index('season').nll).sum()),'mean_bias':float(q[q.variant==bv].bias.mean()),'mean_mae':float(q[q.variant==bv].mae.mean())}
    ug=U.groupby('gap').apply(lambda q: abs(np.average(q.cal_gap,weights=q.n)),include_groups=False); summary={'t20_best':best,'t20_nll_means':{k:float(v) for k,v in cs.items()},'t20_gain_vs_global':gains,'t20_wins_vs_global':wins,'postseason':ps,'upset_weighted_abs_gap_by_bucket':{str(k):float(v) for k,v in ug.items()},'upset_overall_weighted_abs_bucket_gap':float(np.average(ug.values))}
    (OUT/'followup_summary.json').write_text(json.dumps(summary,indent=2)); print(json.dumps(summary,indent=2))
if __name__=='__main__':main()
