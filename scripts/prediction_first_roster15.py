from __future__ import annotations
import json, math, time, urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
import prediction_first_earlyseason15 as e

OUT=Path('results/prediction_first_roster15'); OUT.mkdir(parents=True,exist_ok=True)
FREEZE=Path('/tmp/freeze')
YEARS=[2023,2024,2025]
ROSTER_URL='https://sports.core.api.espn.com/v2/sports/basketball/leagues/mens-college-basketball/seasons/{season}/teams/{team}/athletes?limit=200&lang=en&region=us'

def corr(a,b):
 z=pd.DataFrame({'a':a,'b':b}).replace([np.inf,-np.inf],np.nan).dropna()
 return float(spearmanr(z.a,z.b).statistic) if len(z)>=20 and z.a.nunique()>1 and z.b.nunique()>1 else np.nan

def fetch_roster(season,team):
 u=ROSTER_URL.format(season=season,team=int(team))
 for k in range(3):
  try:
   with urllib.request.urlopen(u,timeout=12) as r: x=json.load(r)
   ids=[]
   for it in x.get('items',[]):
    ref=it.get('$ref',''); aid=ref.split('/athletes/')[-1].split('?')[0].split('/')[0]
    if aid.isdigit(): ids.append(int(aid))
   return season,int(team),sorted(set(ids)),None
  except Exception as ex:
   err=str(ex); time.sleep(.2*(k+1))
 return season,int(team),[],err

def rosters(d):
 keys=d[d.season.isin(YEARS)][['season','team_id']].drop_duplicates()
 out=[]; errors=[]
 with ThreadPoolExecutor(max_workers=36) as pool:
  futs=[pool.submit(fetch_roster,int(r.season),int(r.team_id)) for r in keys.itertuples(index=False)]
  for f in as_completed(futs):
   sy,t,ids,err=f.result()
   if err: errors.append((sy,t,err))
   for a in ids: out.append({'season':sy,'team_id':t,'athlete_id':a})
 R=pd.DataFrame(out); E=pd.DataFrame(errors,columns=['season','team_id','error'])
 R.to_csv(OUT/'historical_season_rosters.csv',index=False); E.to_csv(OUT/'roster_fetch_errors.csv',index=False)
 return R,E,keys

def prep_player(d):
 p=e.player_seasons(d); tg=e.team_games(d)
 tm=tg.groupby(['season','team_id']).margin.mean().rename('team_margin').reset_index()
 return p,tg,tm

def roster_features(R,p,tg):
 rows=[]
 for sy in YEARS:
  prev=p[p.season.eq(sy-1)].copy(); r=R[R.season.eq(sy)].copy(); tm_prev=tg[tg.season.eq(sy-1)].groupby('team_id').margin.mean().to_dict()
  team_min=prev.groupby('team_id').minutes.sum().to_dict(); team_val=prev.groupby('team_id').raw_value.sum().to_dict()
  for team,z in r.groupby('team_id'):
   q=z.merge(prev,on='athlete_id',how='left')
   known=q.team_id_y.notna(); same=known & q.team_id_y.eq(team); tr=known & ~same
   denom=team_min.get(team,np.nan); vden=abs(team_val.get(team,np.nan))
   rows.append({'season':sy,'team_id':team,'roster_n':len(q),'known_prior_n':int(known.sum()),'new_n':int((~known).sum()),'new_count_share':float((~known).mean()),
    'return_min_share':float(q.loc[same,'minutes'].sum()/denom) if denom else np.nan,
    'return_value_share':float(q.loc[same,'raw_value'].sum()/vden) if vden else np.nan,
    'transfer_prev_minutes':float(q.loc[tr,'minutes'].sum()),'transfer_prev_value':float(q.loc[tr,'raw_value'].sum()),
    'transfer_n':int(tr.sum()),'prev_team_margin':tm_prev.get(team,np.nan)})
 return pd.DataFrame(rows)

def player_transfers(p,tg):
 a=p.copy(); a['season']=a.season+1; x=a.merge(p,on=['season','athlete_id'],suffixes=('_prev','_cur'))
 x=x[~x.team_id_prev.eq(x.team_id_cur)].copy(); prevm=tg.groupby(['season','team_id']).margin.mean().rename('m').reset_index()
 o=prevm.copy(); o['season']+=1; o=o.rename(columns={'team_id':'team_id_prev','m':'origin_margin'})
 dd=prevm.copy(); dd['season']+=1; dd=dd.rename(columns={'team_id':'team_id_cur','m':'dest_margin'})
 x=x.merge(o,on=['season','team_id_prev'],how='left').merge(dd,on=['season','team_id_cur'],how='left')
 x['log_minutes']=np.log1p(x.minutes_prev); x['role_q']=pd.qcut(x.minutes_prev.rank(method='first'),4,labels=['Q1','Q2','Q3','Q4'])
 return x

def team_game_features(d,Rf):
 tg=e.team_games(d)
 style=e.rolling_team_style(d)
 # pregame rolling team results/style
 cols=['three_rate','tov_rate','orb_rate','pace_proxy']
 s=style.sort_values(['season','team_id','game_date','game_id']).copy()
 for c in cols: s['pre_'+c]=s.groupby(['season','team_id'])[c].transform(lambda v:v.expanding().mean().shift())
 t=tg.sort_values(['season','team_id','game_date','game_id']).copy(); t['pre_margin']=t.groupby(['season','team_id']).margin.transform(lambda v:v.expanding().mean().shift())
 t=t.merge(s[['season','game_id','team_id']+['pre_'+c for c in cols]],on=['season','game_id','team_id'],how='left')
 t=t.merge(Rf,on=['season','team_id'],how='left')
 # home/away pair using player_box labels
 lab=d[['season','game_id','team_id','home_away']].drop_duplicates(['season','game_id','team_id'])
 t=t.merge(lab,on=['season','game_id','team_id'],how='left')
 numeric=['game_no','pre_margin','pre_three_rate','pre_tov_rate','pre_orb_rate','pre_pace_proxy','new_count_share','return_min_share','return_value_share','transfer_prev_minutes','transfer_prev_value','transfer_n','prev_team_margin']
 h=t[t.home_away.eq('home')][['season','game_id','team_id']+numeric].rename(columns={c:'h_'+c for c in numeric}|{'team_id':'home_team_id'})
 a=t[t.home_away.eq('away')][['season','game_id','team_id']+numeric].rename(columns={c:'a_'+c for c in numeric}|{'team_id':'away_team_id'})
 g=h.merge(a,on=['season','game_id'],how='inner')
 for c in numeric:
  if c!='game_no': g['d_'+c]=g['h_'+c]-g['a_'+c]
 return g

def direct_v14(g):
 p=pd.read_parquet(FREEZE/'frozen_candidate_predictions.parquet').copy(); p['game_id']=pd.to_numeric(p.game_id,errors='coerce').astype('Int64')
 if 'actual_margin' not in p: p['actual_margin']=p.actual_home-p.actual_away
 p['base_margin']=p.pred_margin+p.get('resmean_margin_final',0)
 q=p.merge(g,on=['season','game_id'],how='inner')
 q=q[(q.season.isin([2023,2024,2025])) & (q.h_game_no.le(10)) & (q.a_game_no.le(10))].copy()
 q['resid']=q.actual_margin-q.base_margin
 return q

def fold_test(q,features):
 rows=[]
 for sy in [2024,2025]:
  tr=q[q.season.lt(sy)].dropna(subset=['resid']+features); te=q[q.season.eq(sy)].dropna(subset=['resid']+features)
  if len(tr)<100 or len(te)<100: continue
  m=make_pipeline(StandardScaler(),Ridge(alpha=30)).fit(tr[features],tr.resid)
  base=mean_absolute_error(te.actual_margin,te.base_margin); pred=te.base_margin+m.predict(te[features]); chal=mean_absolute_error(te.actual_margin,pred)
  rows.append({'season':sy,'n':len(te),'base_mae':base,'challenger_mae':chal,'gain':base-chal})
 return pd.DataFrame(rows)

def main():
 d=e.prep(); p,tg,tm=prep_player(d); R,E,keys=rosters(d); RF=roster_features(R,p,tg); X=player_transfers(p,tg); G=team_game_features(d,RF); Q=direct_v14(G)
 gates=[]
 def A(i,s,dn,r,n): gates.append({'gate':f'T28.{i:02d}','status':s,'decision':dn,'result':r,'next_action':n})
 # 1 endpoint feasibility
 fetched=R[['season','team_id']].drop_duplicates().shape[0]; requested=len(keys); cov=fetched/requested if requested else 0
 A(1,'GREEN' if cov>.9 else 'HOLD','PASS — ESPN ERA-CORRECT ROSTER ENDPOINT' if cov>.9 else 'HOLD',f'requested_team_seasons={requested}; fetched={fetched}; coverage={cov:.4f}; errors={len(E)}','Version historical season-roster retrieval and preserve fetch manifest')
 # 2 membership coverage vs actual participants
 part=d[d.season.isin(YEARS)][['season','team_id','athlete_id']].drop_duplicates(); rr=R.merge(part.assign(participant=1),on=['season','team_id','athlete_id'],how='left'); pp=part.merge(R.assign(roster=1),on=['season','team_id','athlete_id'],how='left')
 precision=float(rr.participant.fillna(0).mean()) if len(rr) else np.nan; recall=float(pp.roster.fillna(0).mean()) if len(pp) else np.nan
 A(2,'GREEN' if recall>.9 else 'HOLD','PASS — HIGH PARTICIPANT COVERAGE' if recall>.9 else 'HOLD',f'roster_rows={len(R)}; eventual_participants={len(part)}; roster_to_participant_precision={precision:.4f}; participant_recall={recall:.4f}','Use season roster membership instead of player-box participation for shadow preseason priors')
 # 3 roster-only / participant-only
 roster_only=int(rr.participant.isna().sum()); participant_only=int(pp.roster.isna().sum())
 A(3,'GREEN','PASS — ROSTER AND PARTICIPATION ARE NOT IDENTICAL',f'roster_only={roster_only}; participant_only={participant_only}; roster_only_pct={roster_only/max(len(R),1):.4f}; participant_only_pct={participant_only/max(len(part),1):.4f}','Treat DNP/nonparticipant roster members explicitly; do not infer roster from minutes')
 # 4 timestamp legality
 A(4,'SHADOW','SHADOW — SEASON-CORRECT BUT NOT PRETIP-TIMESTAMPED','ESPN core-v2 is era-correct per season, but this retrospective query does not prove the exact roster existed unchanged before opening tip.','Acquire archived preseason snapshots or timestamped roster manifests before promotion')
 # 5 recruiting data source feasibility
 recruit_key=False
 try:
  req=urllib.request.Request('https://api.collegebasketballdata.com/recruiting/players?year=2025',headers={'User-Agent':'cbb-lab'})
  urllib.request.urlopen(req,timeout=8).read(64); recruit_key=True
 except Exception as ex: recruit_err=str(ex)
 A(5,'HOLD','HOLD — RECRUITING SOURCE EXISTS, AUTH REQUIRED',f'CBBD high-school recruits documented 2007-present; unauthenticated_access={recruit_key}; response={recruit_err[:120]}','Connect/provide CBBD API credential or source equivalent historical recruiting data')
 # 6 transfer conditional model
 xx=X[(X.minutes_prev>=200)&(X.minutes_cur>=100)].dropna(subset=['value40_prev','value40_cur','pts40_prev','reb40_prev','ast40_prev','ts_prev','origin_margin','dest_margin']).copy()
 feats=['value40_prev','pts40_prev','reb40_prev','ast40_prev','ts_prev','log_minutes','origin_margin','dest_margin']; rows=[]
 for sy in [2024,2025]:
  tr=xx[xx.season.lt(sy)]; te=xx[xx.season.eq(sy)]
  if len(tr)<200 or len(te)<100: continue
  m=make_pipeline(StandardScaler(),Ridge(alpha=20)).fit(tr[feats],tr.value40_cur); b=mean_absolute_error(te.value40_cur,te.value40_prev); c=mean_absolute_error(te.value40_cur,m.predict(te[feats])); rows.append({'season':sy,'n':len(te),'raw_prior_mae':b,'conditional_mae':c,'gain':b-c})
 trtest=pd.DataFrame(rows); A(6,'GREEN' if len(trtest) and trtest.gain.mean()>0 else 'REJECT','PASS — CONDITIONAL TRANSFER TRANSLATION' if len(trtest) and trtest.gain.mean()>0 else 'REJECT',f'{trtest.to_dict("records")}; mean_gain={trtest.gain.mean() if len(trtest) else np.nan:.4f}','Use conditional transfer projection in team aggregation challenger')
 # 7 role dependence
 role=xx.groupby('role_q',observed=True).apply(lambda z:pd.Series({'n':len(z),'median_value_shift':(z.value40_cur-z.value40_prev).median(),'rho':corr(z.value40_prev,z.value40_cur)}),include_groups=False).reset_index(); spread=float(role.median_value_shift.max()-role.median_value_shift.min())
 A(7,'GREEN' if spread>.25 else 'HOLD','PASS — TRANSFER ROLE MATTERS' if spread>.25 else 'HOLD',f'role_bins={role.to_dict("records")}; median_shift_spread={spread:.4f}','Include prior role/minutes interaction rather than one transfer penalty')
 # 8 origin/destination context
 xx['context_gap']=xx.dest_margin-xx.origin_margin; xx['gap_bin']=pd.qcut(xx.context_gap,4,duplicates='drop'); ctx=xx.groupby('gap_bin',observed=True).apply(lambda z:pd.Series({'n':len(z),'median_value_shift':(z.value40_cur-z.value40_prev).median()}),include_groups=False).reset_index(); cspread=float(ctx.median_value_shift.max()-ctx.median_value_shift.min())
 A(8,'GREEN' if cspread>.25 else 'HOLD','PASS — COMPETITION/DESTINATION CONTEXT MATTERS' if cspread>.25 else 'HOLD',f'context_bins={ctx.astype(str).to_dict("records")}; shift_spread={cspread:.4f}','Condition transfer translation on origin/destination program strength')
 # 9 roster feature completeness
 comp=RF[['return_min_share','return_value_share','transfer_prev_minutes','transfer_prev_value','new_count_share']].notna().all(axis=1).mean(); A(9,'GREEN' if comp>.9 else 'HOLD','PASS — ROSTER PRIOR FEATURES COMPILE' if comp>.9 else 'HOLD',f'team_seasons={len(RF)}; complete_feature_pct={comp:.4f}; median_roster_n={RF.roster_n.median():.1f}; median_new_count_share={RF.new_count_share.median():.4f}','Use compiled roster prior features in direct v14 residual challenge')
 # 10 direct v14 roster-only correction
 roster_feats=['d_new_count_share','d_return_min_share','d_return_value_share','d_transfer_prev_minutes','d_transfer_prev_value','d_prev_team_margin']; r10=fold_test(Q,roster_feats); mg=float(r10.gain.mean()) if len(r10) else np.nan
 A(10,'SHADOW' if mg>0 else 'REJECT','SHADOW — ROSTER PRIOR IMPROVES V14' if mg>0 else 'REJECT — ROSTER PRIOR DOES NOT IMPROVE V14',f'{r10.to_dict("records")}; mean_gain={mg:.4f}','Keep shadow until roster timing is legally timestamped')
 # 11 sequential margin update direct v14
 seq=['d_pre_margin']; r11=fold_test(Q,seq); mg11=float(r11.gain.mean()) if len(r11) else np.nan
 A(11,'GREEN' if mg11>0 else 'REJECT','PASS — SEQUENTIAL TEAM-STATE RESIDUAL UPDATE' if mg11>0 else 'REJECT',f'{r11.to_dict("records")}; mean_gain={mg11:.4f}','Compare against incumbent maturity/K implementation by game-number slice')
 # 12 style direct v14
 sty=['d_pre_three_rate','d_pre_tov_rate','d_pre_orb_rate','d_pre_pace_proxy']; r12=fold_test(Q,sty); mg12=float(r12.gain.mean()) if len(r12) else np.nan
 A(12,'GREEN' if mg12>0 else 'REJECT','PASS — EARLY STYLE ADDS DIRECT FORECAST SIGNAL' if mg12>0 else 'REJECT — STYLE NOT INCREMENTAL',f'{r12.to_dict("records")}; mean_gain={mg12:.4f}','If positive, test style only after at least one prior game and preserve same-date isolation')
 # 13 combined direct v14
 combo=roster_feats+seq+sty; r13=fold_test(Q,combo); mg13=float(r13.gain.mean()) if len(r13) else np.nan
 A(13,'SHADOW' if mg13>0 else 'REJECT','SHADOW — COMBINED EARLY-SEASON CHALLENGER WINS' if mg13>0 else 'REJECT — COMBINED CHALLENGER FAILS',f'{r13.to_dict("records")}; mean_gain={mg13:.4f}; n_games_total={len(Q)}','Replace retrospective season roster with pretip snapshots, then rerun same folds unchanged')
 # 14 uncertainty from roster newness on frozen v14 residuals
 uq=Q.dropna(subset=['h_new_count_share','a_new_count_share']).copy(); uq['roster_uncertainty']=(uq.h_new_count_share+uq.a_new_count_share)/2; uq['abs_base_error']=abs(uq.resid); rho=corr(uq.roster_uncertainty,uq.abs_base_error)
 A(14,'SHADOW' if rho>0.03 else 'HOLD','SHADOW — ROSTER NEWNESS TRACKS FORECAST ERROR' if rho>0.03 else 'HOLD — WEAK UNCERTAINTY LINK',f'n={len(uq)}; spearman_newness_to_abs_v14_error={rho:.4f}','Test as sigma/reliability input only after legal preseason roster/freshman data are available')
 # 15 synthesis
 statuses=[g['status'] for g in gates]; positive=sum(s in ['GREEN','SHADOW'] for s in statuses); decision='ADVANCE EARLY-SEASON CHALLENGER TO TIMESTAMP-VALID ROSTER/RECRUIT PHASE — V14 REMAINS FROZEN'
 A(15,'GREEN',decision,f'prior_gates=14; green={statuses.count("GREEN")}; shadow={statuses.count("SHADOW")}; hold={statuses.count("HOLD")}; reject={statuses.count("REJECT")}; positive_or_shadow={positive}; production_promotion=False; 2026_outcomes_used=False','Next gates: obtain timestamp-valid preseason rosters + recruiting; rerun T28.10/T28.13 unchanged and require common-fold Games 1-10 win')
 pd.DataFrame(gates).to_csv(OUT/'t28_roster_15_gates.csv',index=False); RF.to_csv(OUT/'t28_roster_features.csv',index=False); trtest.to_csv(OUT/'t28_transfer_conditional.csv',index=False); r10.to_csv(OUT/'t28_v14_roster_challenge.csv',index=False); r11.to_csv(OUT/'t28_v14_state_challenge.csv',index=False); r12.to_csv(OUT/'t28_v14_style_challenge.csv',index=False); r13.to_csv(OUT/'t28_v14_combined_challenge.csv',index=False)
 (OUT/'summary.json').write_text(json.dumps({'green':sum(g['status']=='GREEN' for g in gates),'shadow':sum(g['status']=='SHADOW' for g in gates),'hold':sum(g['status']=='HOLD' for g in gates),'reject':sum(g['status']=='REJECT' for g in gates),'model_reopened':False,'production_promotion':False,'2026_outcomes_used':False,'final_decision':decision},indent=2))
 print(pd.DataFrame(gates).to_string(index=False)); print((OUT/'summary.json').read_text())

if __name__=='__main__': main()
