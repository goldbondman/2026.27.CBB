from pathlib import Path
import pandas as pd,numpy as np,json,hashlib
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
p=Path('/tmp/cbb')
cols=['game_id','game_date','team_id','opponent_team_id','team_home_away','team_score','opponent_team_score','field_goals_attempted','offensive_rebounds','total_turnovers','free_throws_attempted']
data={y:pd.read_parquet(p/f'eb{y}.parquet',columns=cols) for y in [2024,2025,2026]}
for y,d in data.items():
 d['game_date']=pd.to_datetime(d.game_date,errors='coerce')
 d['team_id']=d.team_id.astype(str);d['opponent_team_id']=d.opponent_team_id.astype(str)
 d['pace']=d.field_goals_attempted-d.offensive_rebounds+d.total_turnovers+.475*d.free_throws_attempted
 d['gross_score']=pd.to_numeric(d.team_score,errors='coerce')
 d['valid']=(d.pace.between(40,110))&(d.gross_score>=0)&d.opponent_team_score.notna()
frames={};
for eval_year in [2025,2026]:
 prev=data[eval_year-1];current=data[eval_year]
 pp=prev[prev.valid].groupby('team_id').agg(pace=('pace','mean'),games=('game_id','nunique'),points=('gross_score','sum'),opp_points=('opponent_team_score','sum'),poss=('pace','sum')).reset_index()
 pp=pp[pp.games>=15]
 pp['pp100']=pp.points/pp.poss*100
 pp['def_pp100']=pp.opp_points/pp.poss*100
 league_pp100=float(prev.loc[prev.valid,'gross_score'].sum()/prev.loc[prev.valid,'pace'].sum()*100)
 x=current[current.valid].copy()
 first=x.game_date.min();x=x[(x.game_date>=first)&(x.game_date<first+pd.Timedelta(days=14))]
 hh=x[x.team_home_away=='home'][['game_id','game_date','team_id','opponent_team_id','gross_score','pace']].rename(columns={'team_id':'home_id','opponent_team_id':'away_id','gross_score':'home_score','pace':'home_poss'})
 aa=x[x.team_home_away=='away'][['game_id','team_id','opponent_team_id','gross_score','pace']].rename(columns={'team_id':'away_id','opponent_team_id':'home_id','gross_score':'away_score','pace':'away_poss'})
 g=hh.merge(aa,on=['game_id','home_id','away_id'],validate='one_to_one')
 pr=pp[['team_id','pace','games','pp100','def_pp100']];g=g.merge(pr.add_prefix('h_').rename(columns={'h_team_id':'home_id'}),on='home_id',how='inner')
 g=g.merge(pr.add_prefix('a_').rename(columns={'a_team_id':'away_id'}),on='away_id',how='inner')
 g['actual']=(g.home_poss+g.away_poss)/2
 g['actual_total']=g.home_score+g.away_score
 g['proj_sum_pp100']=(g.h_pp100+g.a_def_pp100-league_pp100)+(g.a_pp100+g.h_def_pp100-league_pp100)
 g=g[g.actual.between(40,110)].copy()
 g['lower']=g[['h_pace','a_pace']].min(axis=1)
 g['higher']=g[['h_pace','a_pace']].max(axis=1)
 g['gap']=g.higher-g.lower
 g['mean']=(g.higher+g.lower)/2
 g['pace_exp']=(g.h_pace*g.a_pace)/max(1.0,prev[prev.valid].pace.mean())
 g['slow60']=.6*g.lower+.4*g.higher
 g['slow70']=.7*g.lower+.3*g.higher
 g['fast60']=.4*g.lower+.6*g.higher
 g['fast70']=.3*g.lower+.7*g.higher
 g['geom']=np.sqrt(g.h_pace*g.a_pace)
 frames[eval_year]=g
 print('COHORT',eval_year,len(g),'dates',g.game_date.nunique(),'mean_actual',g.actual.mean(),'mean_prior_pred',g['mean'].mean(),flush=True)
tr=frames[2025];te=frames[2026]
models=['mean','slow60','slow70','fast60','fast70','geom','pace_exp']
scores=[]
TRAIN_TOTAL_PRED_OFFSET=None
for name in models:
 for yr,g in [(2025,tr),(2026,te)]:
  pred=g[name].values
  err=pred-g.actual.values
  scores.append({'variant':name,'season':yr,'n':len(g),'pace_mae':round(float(np.abs(err).mean()),4),'pace_bias':round(float(err.mean()),4),'pace_rmse':round(float(np.mean(err**2)**.5),4),'total_mae':round(float(np.abs(pred*g.proj_sum_pp100.to_numpy()/100-g.actual_total.to_numpy()).mean()),4),'large_gap_n':int((g.gap>=5).sum()),'large_gap_mae':round(float(np.abs(err[g.gap>=5]).mean()),4) if (g.gap>=5).sum() else None})
# fold-train optimal slow control slope d = target - arithmetic mean ~ b*(pace gap)
X=tr[['gap']].to_numpy(); y=(tr.actual-tr['mean']).to_numpy()
coef=float(np.linalg.lstsq(np.c_[np.ones(len(X)),X],y,rcond=None)[0][1])
intercept=float(np.linalg.lstsq(np.c_[np.ones(len(X)),X],y,rcond=None)[0][0])
def predreg(g): return g['mean'].to_numpy()+intercept+coef*g.gap.to_numpy()
tr_offset=float((tr.actual-tr['mean']).mean())
for yr,g in [(2025,tr),(2026,te)]:
 pred=predreg(g);err=pred-g.actual.to_numpy()
 scores.append({'variant':'trained_gap_interaction','season':yr,'n':len(g),'pace_mae':round(float(np.abs(err).mean()),4),'pace_bias':round(float(err.mean()),4),'pace_rmse':round(float(np.mean(err**2)**.5),4),'total_mae':round(float(np.abs(pred*g.proj_sum_pp100.to_numpy()/100-g.actual_total.to_numpy()).mean()),4),'large_gap_n':int((g.gap>=5).sum()),'large_gap_mae':round(float(np.abs(err[g.gap>=5]).mean()),4) if (g.gap>=5).sum() else None})
# game day clustered holdout bootstrap
for vname, pred_fun in [('train_offset_only',lambda g: g['mean'].to_numpy()+tr_offset),('train_gap_slope_only',lambda g:g['mean'].to_numpy()+coef*g.gap.to_numpy())]:
 for yr,g in [(2025,tr),(2026,te)]:
  pred=pred_fun(g);err=pred-g.actual.to_numpy()
  scores.append({'variant':vname,'season':yr,'n':len(g),'pace_mae':round(float(np.abs(err).mean()),4),'pace_bias':round(float(err.mean()),4),'pace_rmse':round(float(np.mean(err**2)**.5),4),'total_mae':round(float(np.abs(pred*g.proj_sum_pp100.to_numpy()/100-g.actual_total.to_numpy()).mean()),4),'large_gap_n':int((g.gap>=5).sum()),'large_gap_mae':round(float(np.abs(err[g.gap>=5]).mean()),4) if (g.gap>=5).sum() else None})
base=np.abs(te['mean'].to_numpy()-te.actual.to_numpy())
rng=np.random.default_rng(261107)
dt=te.game_date.dt.strftime('%Y-%m-%d').values
dates=np.unique(dt); groups=[np.flatnonzero(dt==d) for d in dates]
paired={}
for name in ['slow60','slow70','fast60','fast70','geom','pace_exp']:
 delta=base-np.abs(te[name].to_numpy()-te.actual.to_numpy())
 boot=np.array([np.concatenate([delta[groups[i]] for i in rng.integers(0,len(groups),len(groups))]).mean() for _ in range(1500)])
 paired[name]={'gain':round(float(delta.mean()),5),'ci95':np.quantile(boot,[.025,.975]).round(5).tolist()}
delta=base-np.abs(predreg(te)-te.actual.to_numpy())
boot=np.array([np.concatenate([delta[groups[i]] for i in rng.integers(0,len(groups),len(groups))]).mean() for _ in range(1500)])
paired['trained_gap_interaction']={'gain':round(float(delta.mean()),5),'ci95':np.quantile(boot,[.025,.975]).round(5).tolist()}
for vname, ppred in [('train_offset_only',te['mean'].to_numpy()+tr_offset),('train_gap_slope_only',te['mean'].to_numpy()+coef*te.gap.to_numpy())]:
 delta=base-np.abs(ppred-te.actual.to_numpy())
 boot=np.array([np.concatenate([delta[groups[i]] for i in rng.integers(0,len(groups),len(groups))]).mean() for _ in range(1500)])
 paired[vname]={'gain':round(float(delta.mean()),5),'ci95':np.quantile(boot,[.025,.975]).round(5).tolist()}
base_total=np.abs(te['mean'].to_numpy()*te.proj_sum_pp100.to_numpy()/100-te.actual_total.to_numpy())
total_paired={}
for vname, ppred in [('fast60',te.fast60.to_numpy()),('fast70',te.fast70.to_numpy()),('slow60',te.slow60.to_numpy()),('trained_gap_interaction',predreg(te)),('train_offset_only',te['mean'].to_numpy()+tr_offset)]:
 delta=base_total-np.abs(ppred*te.proj_sum_pp100.to_numpy()/100-te.actual_total.to_numpy())
 boot=np.array([np.concatenate([delta[groups[i]] for i in rng.integers(0,len(groups),len(groups))]).mean() for _ in range(1500)])
 total_paired[vname]={'gain':round(float(delta.mean()),5),'ci95':np.quantile(boot,[.025,.975]).round(5).tolist()}
result={'design':'2024-25 first 14 days development (2023-24 priors) / 2025-26 first 14 days previously explored historical holdout (2024-25 priors)','note':'simple source ESPN box P475, no official D1 or exact OT eligibility; historical holdout reused; descriptive, not certified independent result','train_coef_gap':coef,'train_intercept':intercept,'train_offset_only':tr_offset,'scores':scores,'holdout_paired_game_day_bootstrap':paired,'holdout_total_paired_game_day_bootstrap':total_paired,'source_files':[str(p/f'eb{y}.parquet') for y in [2024,2025,2026]],'source_sha':{str(y):hashlib.sha256((p/f'eb{y}.parquet').read_bytes()).hexdigest() for y in [2024,2025,2026]}}
(p/'r030_pace_robustness.json').write_text(json.dumps(result,indent=2))
print('SCORES',json.dumps(scores))
print('HOLDOUT_GAIN',json.dumps(paired))
print('GAPFIT',intercept,coef)

