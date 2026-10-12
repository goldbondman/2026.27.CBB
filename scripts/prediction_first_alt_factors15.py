from __future__ import annotations
import json
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
import prediction_first_earlyseason15 as e

OUT=Path('results/prediction_first_alt_factors15'); OUT.mkdir(parents=True,exist_ok=True)
FREEZE=Path('/tmp/freeze')
RNG=np.random.default_rng(20261011)

def corr(a,b):
 z=pd.DataFrame({'a':a,'b':b}).replace([np.inf,-np.inf],np.nan).dropna()
 return float(spearmanr(z.a,z.b).statistic) if len(z)>=50 and z.a.nunique()>1 and z.b.nunique()>1 else np.nan

def build():
 d=e.prep()
 a=d.groupby(['season','game_id','game_date','team_id','opponent_team_id','home_away'],as_index=False).agg(
  fgm=('field_goals_made','sum'),fga=('field_goals_attempted','sum'),fg3m=('three_point_field_goals_made','sum'),fg3a=('three_point_field_goals_attempted','sum'),
  ftm=('free_throws_made','sum'),fta=('free_throws_attempted','sum'),orb=('offensive_rebounds','sum'),dreb=('defensive_rebounds','sum'),tov=('turnovers','sum'),pts=('points','sum'))
 opp=a[['season','game_id','team_id','orb','dreb','tov','fga','fta','pts']].rename(columns={'team_id':'opponent_team_id','orb':'opp_orb','dreb':'opp_dreb','tov':'opp_tov','fga':'opp_fga','fta':'opp_fta','pts':'opp_pts'})
 a=a.merge(opp,on=['season','game_id','opponent_team_id'],how='left')
 a['poss']=a.fga-a.orb+a.tov+.44*a.fta
 a['efg']=(a.fgm+.5*a.fg3m)/a.fga.replace(0,np.nan)
 a['ts']=a.pts/(2*(a.fga+.44*a.fta)).replace(0,np.nan)
 a['tovr']=a.tov/a.poss.replace(0,np.nan)
 a['orbr']=a.orb/(a.orb+a.opp_dreb).replace(0,np.nan)
 a['drbr']=a.dreb/(a.dreb+a.opp_orb).replace(0,np.nan)
 a['ftr']=a.fta/a.fga.replace(0,np.nan)
 a['three_rate']=a.fg3a/a.fga.replace(0,np.nan)
 a['margin']=a.pts-a.opp_pts
 a=a.sort_values(['season','team_id','game_date','game_id']).copy()
 a['game_no']=a.groupby(['season','team_id']).cumcount()+1
 # rolling pregame season state and recent windows
 mets=['efg','ts','tovr','orbr','drbr','ftr','three_rate','poss','margin']
 for c in mets:
  a['pre_'+c]=a.groupby(['season','team_id'])[c].transform(lambda s:s.expanding().mean().shift())
  for w in [5,7]: a[f'l{w}_{c}']=a.groupby(['season','team_id'])[c].transform(lambda s,w=w:s.rolling(w,min_periods=max(3,w-2)).mean().shift())
 # opponent pregame state as it existed before each played game, for recursive schedule adjustment
 pre=a[['season','game_id','team_id']+['pre_'+c for c in mets]+[f'l{w}_{c}' for w in [5,7] for c in ['margin']]].copy()
 pre=pre.rename(columns={'team_id':'opponent_team_id',**{c:'opp_'+c for c in pre.columns if c not in ['season','game_id','team_id']}})
 a=a.merge(pre,on=['season','game_id','opponent_team_id'],how='left')
 # opponent adjusted recent game performance = observed margin - opponent pregame season mean margin expectation proxy
 a['adj_game_margin']=a['margin'] + a['opp_pre_margin'] # opponent stronger positive pre margin => credit performance more
 for w in [5,7]:
  a[f'adj_l{w}_margin']=a.groupby(['season','team_id'])['adj_game_margin'].transform(lambda s,w=w:s.rolling(w,min_periods=max(3,w-2)).mean().shift())
  # recursive: opponent form at time faced folded into game credit
  a[f'rec_game_{w}']=a['margin'] + a[f'opp_l{w}_margin'].fillna(a['opp_pre_margin'])
  a[f'rec_l{w}_margin']=a.groupby(['season','team_id'])[f'rec_game_{w}'].transform(lambda s,w=w:s.rolling(w,min_periods=max(3,w-2)).mean().shift())
 # pair games
 num=[c for c in a.columns if c.startswith('pre_') or c.startswith('l5_') or c.startswith('l7_') or c.startswith('adj_l') or c.startswith('rec_l')]
 num+=['game_no']
 h=a[a.home_away.eq('home')][['season','game_id','team_id','pts','margin']+num].rename(columns={'team_id':'home_team_id','pts':'actual_home','margin':'actual_margin',**{c:'h_'+c for c in num}})
 aw=a[a.home_away.eq('away')][['season','game_id','team_id','pts']+num].rename(columns={'team_id':'away_team_id','pts':'actual_away',**{c:'a_'+c for c in num}})
 g=h.merge(aw,on=['season','game_id'])
 for c in num:
  if c!='game_no': g['d_'+c]=g['h_'+c]-g['a_'+c]
 p=pd.read_parquet(FREEZE/'frozen_candidate_predictions.parquet').copy(); p['game_id']=pd.to_numeric(p.game_id,errors='coerce').astype('Int64'); g['game_id']=pd.to_numeric(g.game_id,errors='coerce').astype('Int64')
 p['base_home']=p.mu_home+p.mean_eh_final; p['base_away']=p.mu_away+p.mean_ea_final; p['base_margin']=p.base_home-p.base_away
 q=p.merge(g,on=['season','game_id'],how='inner'); q=q[q.season.isin([2023,2024,2025])].copy(); q['resid']=q.actual_margin-q.base_margin
 return a,q

def fold(q,features,target='resid',years=(2024,2025),alpha=30):
 rows=[]
 for sy in years:
  tr=q[q.season.lt(sy)].dropna(subset=[target]+features); te=q[q.season.eq(sy)].dropna(subset=[target]+features)
  if len(tr)<300 or len(te)<200: continue
  m=make_pipeline(StandardScaler(),Ridge(alpha=alpha)).fit(tr[features],tr[target]); pred=m.predict(te[features])
  base=mean_absolute_error(te.actual_margin,te.base_margin); chal=mean_absolute_error(te.actual_margin,te.base_margin+pred)
  rows.append({'season':sy,'n':len(te),'base_mae':base,'challenger_mae':chal,'gain':base-chal})
 return pd.DataFrame(rows)

def mg(x): return float(x.gain.mean()) if len(x) else np.nan

def main():
 a,q=build(); G=[]
 def A(i,s,d,r,n):G.append({'gate':f'T34.{i:02d}','status':s,'decision':d,'result':r,'next_action':n})
 # 1 persistence TS vs eFG: first 5 vs rest same season, enough games
 rows=[]
 for sy in [2023,2024,2025]:
  z=a[a.season.eq(sy)]; early=z[z.game_no.le(5)].groupby('team_id')[['ts','efg']].mean(); late=z[z.game_no.gt(5)].groupby('team_id')[['ts','efg']].mean(); x=early.join(late,lsuffix='_e',rsuffix='_l').dropna()
  rows.append({'season':sy,'n':len(x),'ts_rho':corr(x.ts_e,x.ts_l),'efg_rho':corr(x.efg_e,x.efg_l)})
 A(1,'GREEN','PASS — TS% AND eFG% PERSISTENCE COMPARED',str(rows),'Use direct residual test to determine whether TS adds beyond eFG')
 # 2 direct residual efg vs ts
 efg=fold(q,['d_pre_efg']); ts=fold(q,['d_pre_ts']); both=fold(q,['d_pre_efg','d_pre_ts'])
 A(2,'GREEN' if max(mg(efg),mg(ts),mg(both))>0 else 'REJECT','PASS — SHOOTING FORMULATION TESTED',f'efg={efg.to_dict("records")}; ts={ts.to_dict("records")}; both={both.to_dict("records")}; mean_gains={{"efg":{mg(efg):.4f},"ts":{mg(ts):.4f},"both":{mg(both):.4f}}}','Retain only formulation with repeatable incremental residual lift')
 # 3 ORB vs ORB+DRB
 o=fold(q,['d_pre_orbr']); od=fold(q,['d_pre_orbr','d_pre_drbr']); inc=mg(od)-mg(o)
 A(3,'GREEN' if inc>0 else 'REJECT','PASS — DRB ADDS BEYOND ORB' if inc>0 else 'REJECT — DRB DOES NOT ADD BEYOND ORB',f'orb={o.to_dict("records")}; orb_drb={od.to_dict("records")}; incremental_gain={inc:.4f}','If positive, retain DRB only through nested ablation')
 # 4 turnover offense/defense proxy vs differential: pre tovr diff already offense differential; compare separate home/away vs diff
 sep=fold(q,['h_pre_tovr','a_pre_tovr']); diff=fold(q,['d_pre_tovr']); A(4,'GREEN' if max(mg(sep),mg(diff))>0 else 'REJECT','PASS — TURNOVER REPRESENTATIONS TESTED',f'separate={sep.to_dict("records")}; differential={diff.to_dict("records")}; mean_gains={{"sep":{mg(sep):.4f},"diff":{mg(diff):.4f}}}','Use simpler differential unless separate sides materially win')
 # 5 FTR after TS
 ts0=fold(q,['d_pre_ts']); tsf=fold(q,['d_pre_ts','d_pre_ftr']); inc5=mg(tsf)-mg(ts0)
 A(5,'GREEN' if inc5>0 else 'REJECT','PASS — FTR ADDS AFTER TS' if inc5>0 else 'REJECT — FTR REDUNDANT AFTER TS',f'ts={ts0.to_dict("records")}; ts_ftr={tsf.to_dict("records")}; incremental={inc5:.4f}','Drop redundant factor if incremental lift does not replicate')
 # 6 empirically weighted factors
 fac=['d_pre_ts','d_pre_tovr','d_pre_orbr','d_pre_drbr','d_pre_ftr']; fw=fold(q,fac); A(6,'GREEN' if mg(fw)>0 else 'REJECT','PASS — EMPIRICAL ALTERNATIVE FACTOR WEIGHTS IMPROVE V14' if mg(fw)>0 else 'REJECT — FACTOR WEIGHTS DO NOT IMPROVE V14',f'folds={fw.to_dict("records")}; mean_gain={mg(fw):.4f}','Audit fold-specific coefficients and stability')
 # 7 coefficient stability
 coeff=[]
 for sy in [2024,2025]:
  tr=q[q.season.lt(sy)].dropna(subset=['resid']+fac)
  m=make_pipeline(StandardScaler(),Ridge(alpha=30)).fit(tr[fac],tr.resid); co=m[-1].coef_; coeff.append({'season':sy,**{f:float(v) for f,v in zip(fac,co)}})
 cdf=pd.DataFrame(coeff); signs={f:bool(np.sign(cdf[f]).nunique()==1) for f in fac}; stable=sum(signs.values())
 A(7,'GREEN' if stable>=3 else 'HOLD','PASS — MOST FACTOR WEIGHTS SIGN-STABLE' if stable>=3 else 'HOLD — FACTOR WEIGHTS UNSTABLE',f'coefficients={coeff}; sign_stability={signs}','Prefer stable factors; do not interpret unstable weights structurally')
 # 8 combined alternative factor challenger
 comb=['d_pre_ts','d_pre_tovr','d_pre_orbr','d_pre_drbr','d_pre_ftr','d_pre_three_rate','d_pre_poss']; fc=fold(q,comb); A(8,'GREEN' if mg(fc)>0 else 'REJECT','PASS — ALTERNATIVE-FACTOR COMBINED CHALLENGER' if mg(fc)>0 else 'REJECT',f'folds={fc.to_dict("records")}; mean_gain={mg(fc):.4f}','Run family ablation before any challenger tournament')
 # 9 raw L5 margin
 l5=fold(q,['d_l5_margin']); A(9,'GREEN' if mg(l5)>0 else 'REJECT','PASS — RAW L5 ADDS' if mg(l5)>0 else 'REJECT — RAW L5 DOES NOT ADD',f'{l5.to_dict("records")}; mean_gain={mg(l5):.4f}','Compare with opponent-adjusted and recursive versions')
 # 10 raw L7
 l7=fold(q,['d_l7_margin']); A(10,'GREEN' if mg(l7)>0 else 'REJECT','PASS — RAW L7 ADDS' if mg(l7)>0 else 'REJECT — RAW L7 DOES NOT ADD',f'{l7.to_dict("records")}; mean_gain={mg(l7):.4f}','Compare with adjusted forms')
 # 11 opponent-adjusted L5
 a5=fold(q,['d_adj_l5_margin']); A(11,'GREEN' if mg(a5)>max(0,mg(l5)) else 'HOLD','PASS — OPPONENT-ADJUSTED L5 BEATS RAW' if mg(a5)>mg(l5) else 'HOLD — ADJUSTED L5 DOES NOT BEAT RAW',f'raw={mg(l5):.4f}; adjusted={mg(a5):.4f}; folds={a5.to_dict("records")}','Retain only if adjustment adds chronologically')
 # 12 adjusted L7
 a7=fold(q,['d_adj_l7_margin']); A(12,'GREEN' if mg(a7)>max(0,mg(l7)) else 'HOLD','PASS — OPPONENT-ADJUSTED L7 BEATS RAW' if mg(a7)>mg(l7) else 'HOLD — ADJUSTED L7 DOES NOT BEAT RAW',f'raw={mg(l7):.4f}; adjusted={mg(a7):.4f}; folds={a7.to_dict("records")}','Retain only if adjustment adds chronologically')
 # 13 recursive opponent form
 r5=fold(q,['d_rec_l5_margin']); r7=fold(q,['d_rec_l7_margin']); best=max(mg(r5),mg(r7)); rawbest=max(mg(l5),mg(l7))
 A(13,'GREEN' if best>rawbest and best>0 else 'HOLD','PASS — RECURSIVE OPPONENT FORM ADDS' if best>rawbest and best>0 else 'HOLD — RECURSIVE FORM DOES NOT BEAT RAW RECENCY',f'l5={r5.to_dict("records")}; l7={r7.to_dict("records")}; gains={{"rec5":{mg(r5):.4f},"rec7":{mg(r7):.4f},"raw_best":{rawbest:.4f}}}','Stop recursion depth here unless Level-1/2 clearly wins')
 # 14 home/away score decomposition: residual means on known non-neutral home games via exact frozen score means
 z=q.copy(); z['eh']=z.actual_home-z.base_home; z['ea']=z.actual_away-z.base_away
 rows14=[]
 for sy in [2023,2024,2025]:
  x=z[z.season.eq(sy)]; rows14.append({'season':sy,'n':len(x),'home_score_resid_mean':float(x.eh.mean()),'away_score_resid_mean':float(x.ea.mean()),'margin_resid_mean':float((x.eh-x.ea).mean())})
 asym=np.mean([abs(r['home_score_resid_mean']-r['away_score_resid_mean']) for r in rows14])
 A(14,'GREEN','PASS — HOME/AWAY SCORE RESIDUALS DECOMPOSED',f'{rows14}; mean_abs_side_asymmetry={asym:.4f}','Test venue mechanisms (shooting, turnovers, FTR, rebound) against score residuals next')
 # 15 matchup interactions residual screen
 # offense-vs-opponent-style alignments using home/away pre states: shooting rate x opposing turnover/rebound proxies, etc.
 q['mx_three']=q.h_pre_three_rate*q.a_pre_efg - q.a_pre_three_rate*q.h_pre_efg
 q['mx_reb']=q.h_pre_orbr*(1-q.a_pre_drbr)-q.a_pre_orbr*(1-q.h_pre_drbr)
 q['mx_tov']=q.h_pre_tovr*q.a_pre_tovr*(-1) # symmetric stress proxy
 q['mx_ftr']=q.h_pre_ftr*q.a_pre_ftr*(-1)
 inter=['mx_three','mx_reb','mx_tov','mx_ftr']; single={}
 for c in inter: single[c]=mg(fold(q,[c]))
 allx=fold(q,inter); besti=max(single.values()) if single else np.nan
 A(15,'GREEN' if mg(allx)>0 or besti>0 else 'REJECT','PASS — INITIAL MATCHUP RESIDUAL SCREEN FINDS SIGNAL' if mg(allx)>0 or besti>0 else 'REJECT — INITIAL INTERACTIONS ADD NO RESIDUAL SIGNAL',f'single_mean_gains={single}; combined={allx.to_dict("records")}; combined_gain={mg(allx):.4f}','Preregister only positive interactions for deeper mechanistic/opponent-adjusted testing; discard the rest')
 pd.DataFrame(G).to_csv(OUT/'t34_alt_factors_15_gates.csv',index=False); cdf.to_csv(OUT/'t34_factor_coefficients.csv',index=False)
 with open(OUT/'summary.json','w') as f: json.dump({'gates':G,'rows':len(q),'green':sum(x['status']=='GREEN' for x in G),'hold':sum(x['status']=='HOLD' for x in G),'reject':sum(x['status']=='REJECT' for x in G),'production_reopen':False,'outcomes_2026_used':False},f,indent=2)
 print(pd.DataFrame(G).to_string(index=False))
if __name__=='__main__': main()
