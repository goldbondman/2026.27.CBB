from __future__ import annotations

import math
import numpy as np
import pandas as pd
from scipy.stats import norm, multivariate_normal
import prediction_first_fourth15 as m

RNG=np.random.default_rng(20261011)


def old_vs_champion_fast(gp,r2,pace,venue,sched,pm):
    tp=r2.drop(columns=['date'],errors='ignore').merge(pace[['game_id','team','pred_pace']],on=['game_id','team'],how='left',validate='one_to_one'); tp['game_id']=tp.game_id.astype(str)
    site=gp[['game_id','season','date','n_avg']].merge(sched[['game_id','home','away','hk','ak']].drop_duplicates('game_id'),on='game_id').merge(venue[['game_id','neutral_known','neutral','espn_game_id']],on='game_id',how='left')
    site['neutral_known']=site.neutral_known.fillna(False).astype(bool); site=site[site.neutral_known].copy()
    rows=[]; detail=[]
    for sy in m.EVAL:
        trsite=site[(site.season>=2022)&(site.season<sy)]; tesite=site[site.season==sy]; trt=tp[tp.game_id.isin(set(trsite.game_id))].copy(); tet=tp[tp.game_id.isin(set(tesite.game_id))].copy()
        pg=trt.groupby('game_id').agg(actual=('gposs','mean'),pred=('pred_pace','mean')).dropna(); corr=float((pg.actual-pg.pred).mean())
        def pair(t,ss):
            a=t.merge(ss[['game_id','season','date','home','away','hk','ak','neutral','n_avg','espn_game_id']],on='game_id'); a['tk']=a.team.map(m.m.nn); a['is_home']=a.tk.eq(a.hk); a['is_away']=a.tk.eq(a.ak); a=a[a.is_home|a.is_away].copy(); a['old']=a.pred_ortg*a.pred_pace/100; a['champ']=a.pred_ortg*np.clip(a.pred_pace+corr,45,95)/100 + np.where(a.neutral,0,np.where(a.is_home,1.35,-1.35)); out=[]
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
            tr2=tr.copy(); tr2['stage']=tr2.n_avg.map(m.stage_bucket)
            for r in te.itertuples(index=False):
                if pref=='ch':
                    z=tr2[tr2.stage==m.stage_bucket(r.n_avg)]; w=len(z)/(len(z)+500.0)
                    if len(z)>20: lm=z[['eh','ea']].mean().values; lc=np.cov(z[['eh','ea']].values.T)+np.eye(2)*1e-6; mean=w*lm+(1-w)*gm; cov=w*lc+(1-w)*gc
                    else: mean,cov=gm,gc
                else: mean,cov=gm,gc
                sm=math.sqrt(cov[0,0]+cov[1,1]-2*cov[0,1]); pmarg=(getattr(r,pref+'_h')-getattr(r,pref+'_a'))+(mean[0]-mean[1]); probs.append(norm.cdf(pmarg/sm)); nll.append(-multivariate_normal(mean,cov,allow_singular=True).logpdf([r.ah-getattr(r,pref+'_h'),r.aa-getattr(r,pref+'_a')]))
            te[pref+'_p']=probs; te[pref+'_nll']=nll
        y=(te.ah>te.aa).astype(int); row={'season':sy,'n':len(te)}
        for pref in ['old','ch']:
            row[pref+'_team_mae']=float(np.mean(np.r_[abs(te.ah-te[pref+'_h']),abs(te.aa-te[pref+'_a'])])); row[pref+'_margin_mae']=float(abs((te.ah-te.aa)-(te[pref+'_h']-te[pref+'_a'])).mean()); row[pref+'_total_mae']=float(abs((te.ah+te.aa)-(te[pref+'_h']+te[pref+'_a'])).mean()); row[pref+'_brier']=float(np.mean((te[pref+'_p']-y)**2)); row[pref+'_joint_nll']=float(te[pref+'_nll'].mean())
        rows.append(row); te['d_team']=(abs(te.ah-te.ch_h)+abs(te.aa-te.ch_a))/2-(abs(te.ah-te.old_h)+abs(te.aa-te.old_a))/2; te['d_margin']=abs((te.ah-te.aa)-(te.ch_h-te.ch_a))-abs((te.ah-te.aa)-(te.old_h-te.old_a)); te['d_total']=abs((te.ah+te.aa)-(te.ch_h+te.ch_a))-abs((te.ah+te.aa)-(te.old_h+te.old_a)); te['d_brier']=(te.ch_p-y)**2-(te.old_p-y)**2; detail.append(te[['game_id','date','season','d_team','d_margin','d_total','d_brier']])
    A=pd.DataFrame(rows); D=pd.concat(detail,ignore_index=True); D['day']=pd.to_datetime(D.date).dt.date; boots=[]
    days=np.array(sorted(D.day.unique()),dtype=object)
    for metric in ['d_team','d_margin','d_total','d_brier']:
        daily=D.groupby('day')[metric].agg(['sum','count']).reindex(days); sums=daily['sum'].to_numpy(float); counts=daily['count'].to_numpy(float); vals=np.empty(600,float)
        for i in range(600):
            ix=RNG.integers(0,len(days),size=len(days)); vals[i]=sums[ix].sum()/counts[ix].sum()
        boots.append({'metric':metric,'champ_minus_control':float(D[metric].mean()),'ci_low':float(np.quantile(vals,.025)),'ci_high':float(np.quantile(vals,.975))})
    B=pd.DataFrame(boots); A.to_csv(m.OUT/'t23_final_tournament.csv',index=False); B.to_csv(m.OUT/'t23_bootstrap.csv',index=False); return A,B

m.old_vs_champion=old_vs_champion_fast
m.main()
