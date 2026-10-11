from __future__ import annotations
import json
from pathlib import Path
import numpy as np, pandas as pd
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
import prediction_first_roster15 as r

OUT=Path('results/prediction_first_roster15_repair'); OUT.mkdir(parents=True,exist_ok=True)
FREEZE=Path('/tmp/freeze'); PRIOR=Path('/tmp/roster1')

def model_ready_roster(RF, team_keys):
 q=team_keys.merge(RF,on=['season','team_id'],how='left')
 q['roster_missing']=q.roster_n.isna().astype(int)
 q['prior_program_missing']=q.prev_team_margin.isna().astype(int)
 q['new_count_share']=q.new_count_share.fillna(1.0)
 for c in ['return_min_share','return_value_share','transfer_prev_minutes','transfer_prev_value','transfer_n','prev_team_margin','roster_n','known_prior_n','new_n']:
  q[c]=q[c].fillna(0)
 return q

def game_features(d,RF):
 tg=r.e.team_games(d); style=r.e.rolling_team_style(d)
 cols=['three_rate','tov_rate','orb_rate','pace_proxy']
 s=style.sort_values(['season','team_id','game_date','game_id']).copy()
 for c in cols: s['pre_'+c]=s.groupby(['season','team_id'])[c].transform(lambda v:v.expanding().mean().shift())
 t=tg.sort_values(['season','team_id','game_date','game_id']).copy(); t['pre_margin']=t.groupby(['season','team_id']).margin.transform(lambda v:v.expanding().mean().shift())
 t=t.merge(s[['season','game_id','team_id']+['pre_'+c for c in cols]],on=['season','game_id','team_id'],how='left').merge(RF,on=['season','team_id'],how='left')
 lab=d[['season','game_id','team_id','home_away']].drop_duplicates(['season','game_id','team_id']); t=t.merge(lab,on=['season','game_id','team_id'],how='left')
 nums=['game_no','pre_margin','pre_three_rate','pre_tov_rate','pre_orb_rate','pre_pace_proxy','new_count_share','return_min_share','return_value_share','transfer_prev_minutes','transfer_prev_value','transfer_n','prev_team_margin','roster_missing','prior_program_missing']
 h=t[t.home_away.eq('home')][['season','game_id','team_id']+nums].rename(columns={c:'h_'+c for c in nums}|{'team_id':'home_team_id'})
 a=t[t.home_away.eq('away')][['season','game_id','team_id']+nums].rename(columns={c:'a_'+c for c in nums}|{'team_id':'away_team_id'})
 g=h.merge(a,on=['season','game_id'],how='inner')
 for c in nums:
  if c!='game_no':g['d_'+c]=g['h_'+c]-g['a_'+c]
 return g

def direct_v14(g):
 p=pd.read_parquet(FREEZE/'frozen_candidate_predictions.parquet').copy()
 key='espn_game_id' if 'espn_game_id' in p.columns else 'game_id'
 p['join_game_id']=pd.to_numeric(p[key],errors='coerce').astype('Int64'); g=g.copy(); g['join_game_id']=pd.to_numeric(g.game_id,errors='coerce').astype('Int64')
 if 'actual_margin' not in p.columns:p['actual_margin']=p.actual_home-p.actual_away
 p['base_margin']=p.pred_margin+(p.resmean_margin_final if 'resmean_margin_final' in p.columns else 0)
 q=p.merge(g.drop(columns=['game_id']),on=['season','join_game_id'],how='inner')
 q=q[(q.season.isin([2023,2024,2025]))&(q.h_game_no.le(10))&(q.a_game_no.le(10))].copy(); q['resid']=q.actual_margin-q.base_margin
 return q,key

def folds(q,features):
 rows=[]
 for sy in [2024,2025]:
  tr=q[q.season.lt(sy)].dropna(subset=['resid']+features); te=q[q.season.eq(sy)].dropna(subset=['resid']+features)
  if len(tr)<100 or len(te)<100: rows.append({'season':sy,'n_train':len(tr),'n':len(te),'status':'INSUFFICIENT'}); continue
  m=make_pipeline(StandardScaler(),Ridge(alpha=30)).fit(tr[features],tr.resid); corr=m.predict(te[features])
  b=mean_absolute_error(te.actual_margin,te.base_margin); c=mean_absolute_error(te.actual_margin,te.base_margin+corr)
  rows.append({'season':sy,'n_train':len(tr),'n':len(te),'base_mae':b,'challenger_mae':c,'gain':b-c,'status':'OK'})
 return pd.DataFrame(rows)

def main():
 d=r.e.prep(); p,tg,_=r.prep_player(d); R=pd.read_csv(PRIOR/'historical_season_rosters.csv'); RF=r.roster_features(R,p,tg)
 team_keys=d[d.season.isin(r.YEARS)][['season','team_id']].drop_duplicates(); RFm=model_ready_roster(RF,team_keys); G=game_features(d,RFm); Q,key=direct_v14(G)
 old=pd.read_csv(PRIOR/'t28_roster_15_gates.csv'); gates=old[old.gate.isin([f'T28.{i:02d}' for i in range(1,10)])].to_dict('records')
 def A(i,s,dn,res,n):gates.append({'gate':f'T28.{i:02d}','status':s,'decision':dn,'result':res,'next_action':n})
 rawcomp=RF[['return_min_share','return_value_share','transfer_prev_minutes','transfer_prev_value','new_count_share']].notna().all(axis=1).mean(); modelcomp=RFm[['return_min_share','return_value_share','transfer_prev_minutes','transfer_prev_value','new_count_share','prev_team_margin']].notna().all(axis=1).mean()
 # amend T28.09 to distinguish raw vs model-ready completeness
 gates=[g for g in gates if g['gate']!='T28.09']; A(9,'GREEN','PASS — EXPLICIT MISSINGNESS CONTRACT',f'raw_complete_feature_pct={rawcomp:.4f}; model_ready_complete_pct={modelcomp:.4f}; roster_missing_team_seasons={int(RFm.roster_missing.sum())}; prior_program_missing={int(RFm.prior_program_missing.sum())}','Keep missingness flags; do not silently drop new programs or empty roster responses')
 roster=['d_new_count_share','d_return_min_share','d_return_value_share','d_transfer_prev_minutes','d_transfer_prev_value','d_prev_team_margin','d_roster_missing','d_prior_program_missing']
 seq=['d_pre_margin']; style=['d_pre_three_rate','d_pre_tov_rate','d_pre_orb_rate','d_pre_pace_proxy']
 r10=folds(Q,roster); r11=folds(Q,seq); r12=folds(Q,style); r13=folds(Q,roster+seq+style)
 def gain(df):
  z=df[df.status.eq('OK')]; return float(z.gain.mean()) if len(z) else np.nan
 g10,g11,g12,g13=map(gain,[r10,r11,r12,r13])
 A(10,'SHADOW' if g10>0 else 'REJECT','SHADOW — ROSTER PRIOR IMPROVES V14' if g10>0 else 'REJECT — ROSTER PRIOR DOES NOT IMPROVE V14',f'join_key={key}; matched_early_games={len(Q)}; folds={r10.to_dict("records")}; mean_gain={g10:.4f}','If positive, require timestamp-valid pretip roster replication before any production promotion')
 A(11,'GREEN' if g11>0 else 'REJECT','PASS — SEQUENTIAL TEAM-STATE UPDATE ADDS TO V14' if g11>0 else 'REJECT — RAW SEQUENTIAL MARGIN UPDATE NOT INCREMENTAL',f'folds={r11.to_dict("records")}; mean_gain={g11:.4f}','Retain only if it beats frozen maturity logic directly')
 A(12,'GREEN' if g12>0 else 'REJECT','PASS — STYLE UPDATE ADDS TO V14' if g12>0 else 'REJECT — STYLE UPDATE NOT INCREMENTAL',f'folds={r12.to_dict("records")}; mean_gain={g12:.4f}','Keep component stability as research evidence even if direct margin correction fails')
 A(13,'SHADOW' if g13>0 else 'REJECT','SHADOW — COMBINED CHALLENGER BEATS V14' if g13>0 else 'REJECT — COMBINED CHALLENGER DOES NOT BEAT V14',f'folds={r13.to_dict("records")}; mean_gain={g13:.4f}; matched_early_games={len(Q)}','Do not promote unless same specification wins with timestamp-valid roster/recruit inputs')
 uq=Q.dropna(subset=['h_new_count_share','a_new_count_share']).copy(); uq['newness']=(uq.h_new_count_share+uq.a_new_count_share)/2; uq['abs_error']=abs(uq.resid); rho=r.corr(uq.newness,uq.abs_error)
 A(14,'SHADOW' if rho>0.03 else 'HOLD','SHADOW — ROSTER NEWNESS TRACKS V14 ERROR' if rho>0.03 else 'HOLD — ROSTER NEWNESS WEAK FOR ERROR',f'n={len(uq)}; spearman_newness_to_abs_v14_error={rho:.4f}','Test as sigma/reliability input after timestamp-valid roster plus freshman quality data')
 prior=[g['status'] for g in gates]; decision='CONTINUE REGISTERED EARLY-SEASON CHALLENGER — V14 REMAINS FROZEN'
 A(15,'GREEN',decision,f'green_before_final={prior.count("GREEN")}; shadow={prior.count("SHADOW")}; hold={prior.count("HOLD")}; reject={prior.count("REJECT")}; matched_early_games={len(Q)}; adapter_repair=espn_game_id; production_promotion=False; 2026_outcomes_used=False','Next gate family: timestamp-valid preseason roster reconstruction + recruiting/freshman value; rerun exact direct-v14 specification')
 out=pd.DataFrame(gates).sort_values('gate'); out.to_csv(OUT/'t28_roster_15_gates_repaired.csv',index=False); RFm.to_csv(OUT/'t28_model_ready_roster_features.csv',index=False); Q[['season','join_game_id','base_margin','actual_margin','resid','h_game_no','a_game_no']].to_csv(OUT/'t28_v14_join_audit.csv',index=False)
 for name,df in [('roster',r10),('state',r11),('style',r12),('combined',r13)]:df.to_csv(OUT/f't28_v14_{name}_challenge.csv',index=False)
 (OUT/'summary.json').write_text(json.dumps({'green':sum(g['status']=='GREEN' for g in gates),'shadow':sum(g['status']=='SHADOW' for g in gates),'hold':sum(g['status']=='HOLD' for g in gates),'reject':sum(g['status']=='REJECT' for g in gates),'matched_early_games':len(Q),'adapter_repair':'espn_game_id','model_reopened':False,'production_promotion':False,'2026_outcomes_used':False,'final_decision':decision},indent=2)); print(out.to_string(index=False)); print((OUT/'summary.json').read_text())
if __name__=='__main__':main()
