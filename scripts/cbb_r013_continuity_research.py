import pyarrow.parquet as pq,pandas as pd,numpy as np,json,math,hashlib
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from scipy.stats import bootstrap
p=Path('/tmp/cbb')
def read(n,cols): return pq.read_table(p/(n+'.parquet'),columns=cols).to_pandas()
playercols=['athlete_id','team_id','game_id','minutes','points','field_goals_attempted','assists','rebounds']
boxcols=['game_id','game_date','team_id','opponent_team_id','team_home_away','team_score','opponent_team_score','field_goals_attempted','offensive_rebounds','total_turnovers','free_throws_attempted']
raw={}
for y in [2024,2025,2026]:
 raw['p'+str(y)]=read('ep'+str(y),playercols)
 raw['b'+str(y)]=read('eb'+str(y),boxcols)
 if y>=2025:raw['r'+str(y)]=read('er'+str(y),['team_id','athlete_id'])
for y in [2024,2025,2026]:
 d=raw['p'+str(y)]
 d['athlete_id']=pd.to_numeric(d['athlete_id'],errors='coerce').astype('Int64').astype(str)
 d['team_id']=d['team_id'].astype(str)
 for c in ['minutes','points','field_goals_attempted','assists','rebounds']: d[c]=pd.to_numeric(d[c],errors='coerce').fillna(0)
 b=raw['b'+str(y)]
 for c in ['team_id','opponent_team_id']:b[c]=b[c].astype(str)
 b['game_date']=pd.to_datetime(b['game_date'],errors='coerce')
 b['poss']=b.field_goals_attempted-b.offensive_rebounds+b.total_turnovers+.475*b.free_throws_attempted
for y in [2025,2026]:
 d=raw['r'+str(y)];d['athlete_id']=d.athlete_id.astype(str);d['team_id']=d.team_id.astype(str)
summary={}
frames={}
for y in [2025,2026]:
 prev=raw['p'+str(y-1)]
 prior=raw['b'+str(y-1)]
 upcoming=raw['b'+str(y)]
 roster=raw['r'+str(y)]
 prev_pg=prev.groupby(['team_id','athlete_id'],as_index=False)[['minutes','points','field_goals_attempted','assists','rebounds']].sum()
 membership=roster[['team_id','athlete_id']].drop_duplicates()
 ret=prev_pg.merge(membership.assign(returning=1),on=['team_id','athlete_id'],how='left')
 ret['returning']=ret.returning.fillna(0)
 retcols=['minutes','points','field_goals_attempted','assists','rebounds']
 teams=ret.groupby('team_id')
 X=pd.DataFrame({'team_id':sorted(ret.team_id.unique())})
 for col in retcols:
  v=(ret[col]*ret.returning).groupby(ret.team_id).sum()/ret[col].groupby(ret.team_id).sum().replace(0,np.nan)
  X['return_'+col]=X.team_id.map(v).fillna(0)
 X['prior_roster_player_count']=X.team_id.map(teams.size()).fillna(0)
 # previous-season per-team baseline per team game. Exclude games with invalid possession/scores
 b=prior[(prior.poss>40)&(prior.poss<110)&prior.team_score.notna()&prior.opponent_team_score.notna()]
 gg=b.groupby('team_id')
 priorfeat=pd.DataFrame({'team_id':sorted(b.team_id.unique())})
 priorfeat['prior_games']=priorfeat.team_id.map(gg.size())
 priorfeat['prior_off']=priorfeat.team_id.map(gg.apply(lambda g:(g.team_score.sum()/g.poss.sum()*100),include_groups=False))
 priorfeat['prior_def']=priorfeat.team_id.map(gg.apply(lambda g:(g.opponent_team_score.sum()/g.poss.sum()*100),include_groups=False))
 priorfeat['prior_pace']=priorfeat.team_id.map(gg.poss.mean())
 ppp=float(b.team_score.sum()/b.poss.sum()*100)
 pace=float(b.poss.mean())
 t=priorfeat.merge(X,on='team_id',how='left')
 # missing returning info must be eligible only if exact player-season coverage and team 10+ games
 t=t[t.prior_games>=10].copy()
 t=t[t.return_minutes.notna()].copy()
 r=upcoming.copy()
 r=r[(r.team_score.notna())&(r.opponent_team_score.notna())&(r.poss>35)&(r.poss<110)]
 begin=r.game_date.min()
 cutoff=begin+pd.Timedelta(days=14)
 r=r[(r.game_date>=begin)&(r.game_date<cutoff)]
 home=r[r.team_home_away=='home'].copy()
 away=r[r.team_home_away=='away'].copy()
 home=home[['game_id','game_date','team_id','opponent_team_id','team_score']].rename(columns={'team_id':'h_id','opponent_team_id':'a_id','team_score':'h_score'})
 away=away[['game_id','team_id','opponent_team_id','team_score']].rename(columns={'team_id':'a_id','opponent_team_id':'h_id','team_score':'a_score'})
 g=home.merge(away,on=['game_id','h_id','a_id'],validate='one_to_one')
 for key,prefix in [('h_id','h'),('a_id','a')]:
  tmp=t.add_prefix(prefix+'_').rename(columns={prefix+'_team_id':key})
  g=g.merge(tmp,on=key,how='inner',validate='many_to_one')
 g['actual_margin']=g.h_score-g.a_score;g['actual_total']=g.h_score+g.a_score
 g['base_pace']=(g.h_prior_pace+g.a_prior_pace)/2
 g['base_off_home']=ppp+g.h_prior_off-ppp+g.a_prior_def-ppp
 g['base_off_away']=ppp+g.a_prior_off-ppp+g.h_prior_def-ppp
 g['base_margin']=g.base_pace*(g.base_off_home-g.base_off_away)/100+2.7
 g['base_total']=g.base_pace*(g.base_off_home+g.base_off_away)/100
 for col in retcols:
  g['ret_diff_'+col]=g['h_return_'+col]-g['a_return_'+col]
  g['ret_sum_'+col]=g['h_return_'+col]+g['a_return_'+col]
  g['ret_min_'+col]=np.minimum(g['h_return_'+col],g['a_return_'+col])
 g['strength_diff']=g.h_prior_off-g.a_prior_off-(g.h_prior_def-g.a_prior_def)
 g['strength_sum']=g.h_prior_off+g.a_prior_off+g.h_prior_def+g.a_prior_def
 g['prior_maturity']=np.minimum(g.h_prior_games,g.a_prior_games)
 summary[y]={'prior_player_rows':len(prev),'roster_players':len(roster),'eligible_games':len(g),'raw_completed_games':len(home),'prior_teams':len(t),'ret_minutes_mean':float(t.return_minutes.mean()),'ret_minutes_p10':float(t.return_minutes.quantile(.1)),'ret_minutes_p90':float(t.return_minutes.quantile(.9)),'prior_league_pp100':ppp,'prior_league_pace':pace,'date_min':str(begin),'date_end':str(cutoff)}
 frames[y]=g
d=frames[2025];e=frames[2026]
print("COHORT",json.dumps(summary,default=str),flush=True)
print("FEATURES",d.shape,e.shape)
baseFeatures=['base_margin','base_total','base_pace','strength_diff','strength_sum','prior_maturity']
rosterFeatures=['ret_diff_minutes','ret_sum_minutes','ret_diff_points','ret_sum_points','ret_diff_field_goals_attempted','ret_sum_field_goals_attempted','ret_min_minutes']
def calculate(target):
 if target=='margin': y='actual_margin';bf=['base_margin','strength_diff','base_pace','prior_maturity'];rf=['ret_diff_minutes','ret_diff_points','ret_diff_field_goals_attempted','ret_min_minutes']
 else:y='actual_total';bf=['base_total','base_pace','strength_sum','prior_maturity'];rf=['ret_sum_minutes','ret_sum_points','ret_sum_field_goals_attempted','ret_min_minutes']
 models={}
 for tag,cols in [('A',bf),('B',bf+rf)]:
  model=make_pipeline(StandardScaler(),Ridge(alpha=200))
  model.fit(d[cols],d[y])
  pred=model.predict(e[cols])
  err=pred-e[y].to_numpy()
  models[tag]={'pred':pred,'errors':err,'mae':float(np.mean(np.abs(err))),'bias':float(np.mean(err)),'rmse':float(np.sqrt(np.mean(err**2))),'train_mae':float(np.mean(np.abs(model.predict(d[cols])-d[y]))),'features':cols}
 errsA=np.abs(models['A']['errors']);errsB=np.abs(models['B']['errors']);delta=errsA-errsB
 rng=np.random.default_rng(12062026);dates=e.game_date.dt.strftime('%Y-%m-%d').values;ud=np.unique(dates);idx=[np.flatnonzero(dates==dt) for dt in ud]
 boots=np.array([np.mean(np.concatenate([delta[idx[t]] for t in rng.integers(len(idx),size=len(idx))])) for _ in range(1500)])
 # uncertainty models U0/U1: 2025 in sample errors for A: cheap exploratory only
 trainA=make_pipeline(StandardScaler(),Ridge(alpha=200)).fit(d[bf],d[y])
 trErr=np.abs(trainA.predict(d[bf])-d[y])
 # fit risk targets absolute residual; capped and positive
 riskbase=['base_pace','prior_maturity','base_margin','base_total']
 riskplus=riskbase+['ret_min_minutes','ret_sum_minutes','ret_diff_minutes','ret_sum_points']
 risk={}
 for name,cols in [('U0',riskbase),('U1',riskplus)]:
  model=make_pipeline(StandardScaler(),Ridge(alpha=350))
  model.fit(d[cols],trErr)
  predAbs=np.maximum(2.0,model.predict(e[cols]))
  absErr=errsA
  halfWidth=1.604*predAbs
  risk[name]={'abs_error_mae':float(np.mean(absErr-predAbs)),'risk_mae':float(np.mean(np.abs(absErr-predAbs))),'risk_rmse':float(np.sqrt(np.mean((absErr-predAbs)**2))),'coverage80':float(np.mean(absErr<=halfWidth)),'mean_halfwidth':float(np.mean(halfWidth)),'retained60_mae':float(np.mean(absErr[predAbs<=np.quantile(predAbs,.6)])),'rejected40_mae':float(np.mean(absErr[predAbs>np.quantile(predAbs,.6)])),'coverage_frac':float(np.mean(predAbs<=np.quantile(predAbs,.6))),'spearman_error_corr':float(pd.Series(predAbs).corr(pd.Series(absErr),method='spearman'))}
 return {'baseline':{k:v for k,v in models['A'].items() if k not in ('pred','errors')},'player':{k:v for k,v in models['B'].items() if k not in ('pred','errors')},'paired_mae_gain':float(delta.mean()),'bootstrap_ci95':np.percentile(boots,[2.5,97.5]).tolist(),'risk':risk,'n':len(e),'date_clusters':len(ud)}
results={'source_method':'ESPN current season year-specific roster snapshots (retrospective, PIT uncertified)','2025_development_2026_reused_screen':summary,'margin':calculate('margin'),'total':calculate('total')}
print("RESULTS",json.dumps(results,default=str),flush=True)
(p/'continuity_research.json').write_text(json.dumps(results,default=str,indent=2))

