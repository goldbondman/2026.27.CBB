from __future__ import annotations

import json, math, re
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.special import gammaln
from scipy.stats import norm, poisson, nbinom, binom

RAW='https://raw.githubusercontent.com/sportsdataverse/ncaa-mbb-hoops-data/main/'
SEASONS=range(2021,2026)
EVAL=[2023,2024,2025]
OUT=Path('results/prediction_first_next15'); OUT.mkdir(parents=True,exist_ok=True)


def pick(df, opts, required=True):
    for c in opts:
        if c in df.columns: return c
    if required: raise KeyError(f'missing {opts}; columns={list(df.columns)}')
    return None

def num(x): return pd.to_numeric(x,errors='coerce')
def nkey(x): return re.sub(r'[^a-z0-9]','',str(x).lower()) if pd.notna(x) else ''
def normal_crps(y,mu,sd):
    y=np.asarray(y,float); mu=np.asarray(mu,float); sd=np.clip(np.asarray(sd,float),1e-6,None); z=(y-mu)/sd
    return float(np.mean(sd*(z*(2*norm.cdf(z)-1)+2*norm.pdf(z)-1/math.sqrt(math.pi))))
def gauss_nll(y,mu,sd):
    y=np.asarray(y,float); mu=np.asarray(mu,float); sd=np.clip(np.asarray(sd,float),1e-6,None)
    return float(np.mean(.5*np.log(2*np.pi*sd*sd)+.5*((y-mu)/sd)**2))
def pois_nll(y,mu):
    y=np.rint(np.asarray(y,float)).astype(int); mu=np.clip(np.asarray(mu,float),1e-6,None)
    return float(-np.mean(poisson.logpmf(y,mu)))
def nb_params(mu,var):
    mu=np.clip(np.asarray(mu,float),1e-6,None); var=max(float(var),float(np.mean(mu))+1e-6)
    m=float(np.mean(mu)); alpha=max((var-m)/(m*m),1e-8); r=1/alpha; p=r/(r+mu); return r,p
def nb_nll(y,mu,var):
    y=np.rint(np.asarray(y,float)).astype(int); r,p=nb_params(mu,var); return float(-np.mean(nbinom.logpmf(y,r,p)))
def binom_nll(k,n,p):
    k=np.rint(np.asarray(k,float)).astype(int); n=np.maximum(np.rint(np.asarray(n,float)).astype(int),0); p=np.clip(np.asarray(p,float),1e-6,1-1e-6)
    ok=n>=k
    return float(-np.mean(binom.logpmf(k[ok],n[ok],p[ok]))) if ok.any() else np.nan

def load_team():
    xs=[]
    for sy in SEASONS:
        d=pd.read_parquet(RAW+f'mbb/team_box/parquet/ncaa_mbb_team_box_{sy}.parquet'); d['season']=sy; xs.append(d)
    x=pd.concat(xs,ignore_index=True)
    cols={
      'game_id':pick(x,['contest_id','game_id']), 'team':pick(x,['team','team_name','team_display_name']),
      'pts':pick(x,['pts','points']), 'poss':pick(x,['o_poss','offensive_possessions']),
      'fga':pick(x,['fga','field_goals_attempted','field_goal_attempts']),
      'fgm':pick(x,['fgm','field_goals_made','field_goal_makes']),
      'tov':pick(x,['turnovers','total_turnovers','totalTurnovers','tov']),
      'orb':pick(x,['offensive_rebounds','off_rebounds','orb']),
      'fta':pick(x,['free_throws_attempted','free_throw_attempts','fta']),
      'threepa':pick(x,['three_point_field_goals_attempted','three_point_attempts','three_point_field_goal_attempts','threepa','3pa']),
      'threepm':pick(x,['three_point_field_goals_made','three_point_made','three_point_field_goal_makes','threepm','3pm'])}
    z=x[[cols[k] for k in cols]+['season']].copy(); z.columns=list(cols.keys())+['season']
    for c in ['pts','poss','fga','fgm','tov','orb','fta','threepa','threepm']: z[c]=num(z[c])
    s=pd.read_parquet(RAW+'mbb/ncaa_mbb_schedule_master.parquet'); sg=pick(s,['contest_id','game_id']); sd=pick(s,['date','game_date','start_date'])
    ss=s[[sg,sd]].drop_duplicates(sg).rename(columns={sg:'game_id',sd:'date'}); z=z.merge(ss,on='game_id',how='left'); z['date']=pd.to_datetime(z.date,errors='coerce')
    z=z.dropna(subset=['game_id','team','date','poss','fga','fgm','tov','orb','fta','threepa','threepm']).copy(); z['game_id']=z.game_id.astype(str)
    cnt=z.groupby('game_id').size(); z=z[z.game_id.isin(cnt[cnt==2].index)].copy().sort_values(['date','game_id','team'],kind='mergesort')
    z['ix']=z.groupby('game_id').cumcount(); opp=z[['game_id','ix','team','tov','orb','fta','threepa','threepm','fga','fgm']].copy(); opp['ix']=1-opp['ix']; opp=opp.rename(columns={c:f'opp_{c}' for c in opp.columns if c not in ['game_id','ix']})
    z=z.merge(opp,on=['game_id','ix'],validate='one_to_one'); z['gposs']=z.groupby('game_id').poss.transform('mean')
    z['miss']=np.maximum(z.fga-z.fgm,0); z['tov_rate']=z.tov/z.gposs; z['fta_rate']=z.fta/z.gposs; z['orb_rate']=np.where(z['miss']>0,z.orb/z['miss'],np.nan); z['threepa_rate']=np.where(z.fga>0,z.threepa/z.fga,np.nan); z['threep_pct']=np.where(z.threepa>0,z.threepm/z.threepa,np.nan)
    return z.reset_index(drop=True)

def preseason_priors(z):
    out={}; gl={}
    vars=['gposs','tov_rate','fta_rate','orb_rate','threepa_rate','threep_pct']
    for sy in SEASONS:
        q=z[z.season==sy]
        gl[sy]={v:float(q[v].mean()) for v in vars}
        for t,g in q.groupby('team'): out[(sy,t)]={v:float(g[v].mean()) for v in vars}
    return out,gl

def compile_features(z, reverse_within_day=False):
    pri,gl=preseason_priors(z); vars=['gposs','tov_rate','fta_rate','orb_rate','threepa_rate','threep_pct']; sums=defaultdict(lambda:defaultdict(float)); ns=defaultdict(lambda:defaultdict(int)); allow=defaultdict(lambda:defaultdict(float)); an=defaultdict(lambda:defaultdict(int)); rows=[]; cur_sy=None
    def state(sy,t,v,defense=False):
        S=allow if defense else sums; N=an if defense else ns
        if N[t][v]>0: return S[t][v]/N[t][v]
        p=pri.get((sy-1,t),{}).get(v,np.nan)
        if pd.notna(p): return p
        return gl.get(sy-1,gl.get(sy,{})).get(v,0.0)
    for date,day in z.groupby('date',sort=True):
        sy=int(day.season.iloc[0]);
        if cur_sy!=sy: sums=defaultdict(lambda:defaultdict(float)); ns=defaultdict(lambda:defaultdict(int)); allow=defaultdict(lambda:defaultdict(float)); an=defaultdict(lambda:defaultdict(int)); cur_sy=sy
        it=day.sort_values(['game_id','team'],ascending=not reverse_within_day).itertuples(index=False); updates=[]
        for r in it:
            opp=r.opp_team; pred={v:(state(sy,r.team,v)+state(sy,opp,v,True))/2 for v in vars}
            rows.append({'season':sy,'date':date,'game_id':r.game_id,'team':r.team,'opp':opp,'prior_games':ns[r.team]['gposs'],**{f'pred_{v}':pred[v] for v in vars},**{v:getattr(r,v) for v in vars},'tov':r.tov,'orb':r.orb,'fta':r.fta,'threepa':r.threepa,'threepm':r.threepm,'fga':r.fga,'fgm':r.fgm,'miss':r.miss,'pts':r.pts})
            updates.append(r)
        for r in updates:
            for v in vars:
                a=getattr(r,v); oa=getattr(r if False else r,v)
                if pd.notna(a): sums[r.team][v]+=float(a); ns[r.team][v]+=1
                # opponent-allowed value equals this team's realized offensive value
                if pd.notna(a): allow[r.opp_team][v]+=float(a); an[r.opp_team][v]+=1
    return pd.DataFrame(rows)

def possession_distribution(f):
    g=f.groupby(['season','game_id'],as_index=False).agg(actual=('gposs','mean'),pred=('pred_gposs','mean'),prior_games=('prior_games','mean'))
    rec=[]
    for sy in EVAL:
        tr=g[(g.season>=2022)&(g.season<sy)]; te=g[g.season==sy].copy(); resid=tr.actual-tr.pred; sd=float(resid.std(ddof=1)); mu=float(resid.mean()); te_mu=te.pred+mu
        rec.append({'season':sy,'n':len(te),'normal_nll':gauss_nll(te.actual,te_mu,sd),'normal_crps':normal_crps(te.actual,te_mu,sd),'normal_80cov':float((abs(te.actual-te_mu)<=norm.ppf(.9)*sd).mean()),'normal_90cov':float((abs(te.actual-te_mu)<=norm.ppf(.95)*sd).mean()),'poisson_nll':pois_nll(te.actual,te_mu),'resid_mean':mu,'resid_sd':sd,'mae':float(np.mean(abs(te.actual-te_mu)))})
    pd.DataFrame(rec).to_csv(OUT/'possession_distribution.csv',index=False)
    return rec

def event_models(f):
    rec=[]; std_rows=[]
    for sy in EVAL:
        tr=f[(f.season>=2022)&(f.season<sy)].copy(); te=f[f.season==sy].copy()
        # turnover counts per possession
        muv=np.clip(te.pred_tov_rate*te.pred_gposs,1e-3,None); tr_mu=np.clip(tr.pred_tov_rate*tr.pred_gposs,1e-3,None); var=float(np.var(tr.tov-tr_mu,ddof=1)+np.mean(tr_mu))
        pn=pois_nll(te.tov,muv); nn=nb_nll(te.tov,muv,var)
        rec.append({'season':sy,'component':'TOV','baseline_nll':pn,'challenger_nll':nn,'gain':pn-nn,'model':'NB_vs_Poisson'})
        # FTA counts per possession
        muv=np.clip(te.pred_fta_rate*te.pred_gposs,1e-3,None); tr_mu=np.clip(tr.pred_fta_rate*tr.pred_gposs,1e-3,None); var=float(np.var(tr.fta-tr_mu,ddof=1)+np.mean(tr_mu)); pn=pois_nll(te.fta,muv); nn=nb_nll(te.fta,muv,var)
        rec.append({'season':sy,'component':'FTA','baseline_nll':pn,'challenger_nll':nn,'gain':pn-nn,'model':'NB_vs_Poisson'})
        # ORB realization conditional on actual misses; team/opponent prior rate
        p=np.clip(te.pred_orb_rate,1e-4,.9999); n=np.rint(te['miss']).astype(int); k=np.rint(te.orb).astype(int); b=binom_nll(k,n,p)
        glob=float(tr.orb.sum()/max(tr['miss'].sum(),1)); bg=binom_nll(k,n,np.repeat(glob,len(te))); rec.append({'season':sy,'component':'ORB','baseline_nll':bg,'challenger_nll':b,'gain':bg-b,'model':'team_opp_binomial_vs_global'})
        # 3PA share conditional on actual FGA
        p=np.clip(te.pred_threepa_rate,1e-4,.9999); n=np.rint(te.fga).astype(int); k=np.rint(te.threepa).astype(int); b=binom_nll(k,n,p); glob=float(tr.threepa.sum()/max(tr.fga.sum(),1)); bg=binom_nll(k,n,np.repeat(glob,len(te))); rec.append({'season':sy,'component':'3PA','baseline_nll':bg,'challenger_nll':b,'gain':bg-b,'model':'team_opp_binomial_vs_global'})
        # 3P makes conditional on attempts
        p=np.clip(te.pred_threep_pct,1e-4,.9999); n=np.rint(te.threepa).astype(int); k=np.rint(te.threepm).astype(int); b=binom_nll(k,n,p); glob=float(tr.threepm.sum()/max(tr.threepa.sum(),1)); bg=binom_nll(k,n,np.repeat(glob,len(te))); rec.append({'season':sy,'component':'3PM','baseline_nll':bg,'challenger_nll':b,'gain':bg-b,'model':'team_opp_binomial_vs_global'})
        # standardized residuals for dependence; use simple variance standardization
        temp=pd.DataFrame({'season':sy,'game_id':te.game_id,'team':te.team})
        temp['z_tov']=(te.tov-te.pred_tov_rate*te.pred_gposs)/max(float((tr.tov-tr.pred_tov_rate*tr.pred_gposs).std(ddof=1)),1e-6)
        temp['z_fta']=(te.fta-te.pred_fta_rate*te.pred_gposs)/max(float((tr.fta-tr.pred_fta_rate*tr.pred_gposs).std(ddof=1)),1e-6)
        temp['z_orb']=(te.orb-te.pred_orb_rate*te['miss'])/max(float((tr.orb-tr.pred_orb_rate*tr['miss']).std(ddof=1)),1e-6)
        temp['z_3pa']=(te.threepa-te.pred_threepa_rate*te.fga)/max(float((tr.threepa-tr.pred_threepa_rate*tr.fga).std(ddof=1)),1e-6)
        temp['z_3pm']=(te.threepm-te.pred_threep_pct*te.threepa)/max(float((tr.threepm-tr.pred_threep_pct*tr.threepa).std(ddof=1)),1e-6)
        temp['score_resid_proxy']=te.pts-(te.pts.mean())
        std_rows.append(temp)
    R=pd.DataFrame(rec); R.to_csv(OUT/'event_model_metrics.csv',index=False); Z=pd.concat(std_rows,ignore_index=True); Z.to_csv(OUT/'event_standardized_residuals.csv',index=False)
    # dependence stability + covariance NLL vs independent
    deps=[]; cols=['z_tov','z_fta','z_orb','z_3pa','z_3pm']
    allcorr=[]
    for sy in EVAL:
        q=Z[Z.season==sy][cols].dropna(); c=q.corr(); allcorr.append(c)
        for i,a in enumerate(cols):
            for b in cols[i+1:]: deps.append({'season':sy,'a':a,'b':b,'corr':float(c.loc[a,b])})
    D=pd.DataFrame(deps); D.to_csv(OUT/'event_dependence_correlations.csv',index=False)
    # sign stability + max magnitudes
    stab=[]
    for (a,b),q in D.groupby(['a','b']):
        vals=q.sort_values('season').corr.values; stab.append({'a':a,'b':b,'mean_corr':float(np.mean(vals)),'max_abs':float(np.max(np.abs(vals))),'same_sign_3of3':bool(np.all(vals>0) or np.all(vals<0))})
    pd.DataFrame(stab).to_csv(OUT/'event_dependence_stability.csv',index=False)
    return R,D

def player_minutes_legality():
    xs=[]
    for sy in SEASONS:
        d=pd.read_parquet(RAW+f'mbb/player_box/parquet/ncaa_mbb_player_box_{sy}.parquet'); d['season']=sy; xs.append(d)
    p=pd.concat(xs,ignore_index=True); gid=pick(p,['contest_id','game_id']); tm=pick(p,['team','team_name','team_display_name']); pl=pick(p,['athlete_id','athlete','player','name','athlete_display_name']); mn=pick(p,['mins','minutes']);
    p=p[[gid,tm,pl,mn,'season']].rename(columns={gid:'game_id',tm:'team',pl:'player',mn:'minutes'}); p['minutes']=num(p.minutes).fillna(0); p['game_id']=p.game_id.astype(str); p['pk']=p.player.map(nkey)
    s=pd.read_parquet(RAW+'mbb/ncaa_mbb_schedule_master.parquet'); sg=pick(s,['contest_id','game_id']); sd=pick(s,['date','game_date','start_date']); ss=s[[sg,sd]].drop_duplicates(sg).rename(columns={sg:'game_id',sd:'date'}); ss['game_id']=ss.game_id.astype(str); p=p.merge(ss,on='game_id',how='left'); p['date']=pd.to_datetime(p.date,errors='coerce'); p=p.dropna(subset=['date','team','pk']).sort_values(['date','game_id','team','pk'])
    hist=defaultdict(lambda:defaultdict(lambda:deque(maxlen=5))); seen=defaultdict(set); rec=[]
    for date,day in p.groupby('date',sort=True):
        # evaluate each team-game from roster known strictly before date
        for (gid,team),g in day.groupby(['game_id','team']):
            roster=sorted(seen[team]); raw={pk:(np.mean(hist[team][pk]) if len(hist[team][pk]) else 0.0) for pk in roster}; den=sum(raw.values()); pred={pk:(200*v/den if den>0 else 0.0) for pk,v in raw.items()}; actual=g.groupby('pk').minutes.sum().to_dict(); keys=set(pred)|set(actual)
            if den>0:
                ae=[abs(pred.get(k,0)-actual.get(k,0)) for k in keys]; rec.append({'season':int(g.season.iloc[0]),'game_id':gid,'team':team,'n_known':len(roster),'n_actual':len(actual),'pred_sum':sum(pred.values()),'actual_sum':sum(actual.values()),'player_min_mae':float(np.mean(ae)),'unseen_actual_minutes':float(sum(v for k,v in actual.items() if k not in pred))})
        # update only after all predictions for date
        for (gid,team),g in day.groupby(['game_id','team']):
            actual=g.groupby('pk').minutes.sum().to_dict(); roster=set(seen[team])|set(actual)
            for pk in roster: hist[team][pk].append(float(actual.get(pk,0.0)))
            seen[team]|=set(actual)
    R=pd.DataFrame(rec); R.to_csv(OUT/'projected_minutes_legality.csv',index=False); summ=R[R.season>=2022].groupby('season').agg(games=('game_id','size'),mean_pred_sum=('pred_sum','mean'),mean_actual_sum=('actual_sum','mean'),player_min_mae=('player_min_mae','mean'),unseen_actual_minutes=('unseen_actual_minutes','mean')).reset_index(); summ.to_csv(OUT/'projected_minutes_summary.csv',index=False); return summ

def main():
    z=load_team(); f=compile_features(z,False); f2=compile_features(z,True)
    keys=['season','date','game_id','team']; cols=[c for c in f.columns if c.startswith('pred_')]; a=f[keys+cols].sort_values(keys).reset_index(drop=True); b=f2[keys+cols].sort_values(keys).reset_index(drop=True); maxdiff=float(np.nanmax(np.abs(a[cols].to_numpy(float)-b[cols].to_numpy(float))))
    integ={'team_rows':len(f),'games':int(f.game_id.nunique()),'same_date_reverse_order_max_abs_diff':maxdiff,'missing_pred_fraction':float(f[cols].isna().mean().mean())}; (OUT/'component_compiler_integrity.json').write_text(json.dumps(integ,indent=2))
    f.to_parquet(OUT/'component_features.parquet',index=False)
    poss=possession_distribution(f); events,deps=event_models(f); mins=player_minutes_legality()
    summary={'integrity':integ,'possession':poss,'event_mean_gain':events.groupby('component').gain.mean().to_dict(),'event_season_wins':events.assign(win=events.gain>0).groupby('component').win.sum().astype(int).to_dict(),'minutes_summary':mins.to_dict(orient='records')}
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2,default=float)); print(json.dumps(summary,indent=2,default=float))
if __name__=='__main__': main()
