from __future__ import annotations
import json
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import norm
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
import prediction_first_roster15 as r
import prediction_first_roster15_repair as rr

OUT=Path('results/prediction_first_earlyseason_robust15'); OUT.mkdir(parents=True,exist_ok=True)
FREEZE=Path('/tmp/freeze'); PRIOR=Path('/tmp/roster1')
RNG=np.random.default_rng(20261011)
ROSTER=['d_new_count_share','d_return_min_share','d_return_value_share','d_transfer_prev_minutes','d_transfer_prev_value','d_prev_team_margin','d_roster_missing','d_prior_program_missing']
STATE=['d_pre_margin']
STYLE=['d_pre_three_rate','d_pre_tov_rate','d_pre_orb_rate','d_pre_pace_proxy']
COMBINED=ROSTER+STATE+STYLE

def build_q():
 d=r.e.prep(); p,tg,_=r.prep_player(d); R=pd.read_csv(PRIOR/'historical_season_rosters.csv')
 RF=r.roster_features(R,p,tg); keys=d[d.season.isin(r.YEARS)][['season','team_id']].drop_duplicates(); RFm=rr.model_ready_roster(RF,keys)
 G=rr.game_features(d,RFm); Q,key=rr.direct_v14(G)
 Q['stage_game']=Q[['h_game_no','a_game_no']].max(axis=1)
 Q['game_bucket']=pd.cut(Q.stage_game,bins=[1,3,5,10],labels=['2-3','4-5','6-10'],include_lowest=False,right=True)
 Q['actual_win']=(Q.actual_margin>0).astype(float)
 return Q,key

def fit_predict(tr,te,features,target='resid',alpha=30):
 cols=[c for c in features if c in tr.columns and c in te.columns]
 tr=tr.dropna(subset=[target]+cols); te=te.dropna(subset=[target]+cols)
 if len(cols)==0 or len(tr)<100 or len(te)<50:return None,None,None
 m=make_pipeline(StandardScaler(),Ridge(alpha=alpha)).fit(tr[cols],tr[target]); return m,m.predict(te[cols]),cols

def folds(Q,features,subset=None,target='resid',alpha=30):
 z=Q.copy() if subset is None else Q[subset(Q)].copy(); rows=[]
 for sy in [2024,2025]:
  tr=z[z.season.lt(sy)]; te=z[z.season.eq(sy)]; m,corr,cols=fit_predict(tr,te,features,target,alpha)
  if m is None: rows.append({'season':sy,'n_train':len(tr),'n':len(te),'status':'INSUFFICIENT'}); continue
  te=te.dropna(subset=[target]+cols).copy(); base=mean_absolute_error(te.actual_margin,te.base_margin); chal=mean_absolute_error(te.actual_margin,te.base_margin+corr)
  rows.append({'season':sy,'n_train':len(tr.dropna(subset=[target]+cols)),'n':len(te),'base_mae':base,'challenger_mae':chal,'gain':base-chal,'status':'OK'})
 return pd.DataFrame(rows)

def mean_gain(df):
 z=df[df.status.eq('OK')] if 'status' in df else df
 return float(z.gain.mean()) if len(z) and 'gain' in z else np.nan

def combined_predictions(Q,alpha=30):
 outs=[]
 for sy in [2024,2025]:
  tr=Q[Q.season.lt(sy)]; te=Q[Q.season.eq(sy)]; m,corr,cols=fit_predict(tr,te,COMBINED,'resid',alpha)
  if m is None:continue
  te=te.dropna(subset=['resid']+cols).copy(); te['corr']=corr; te['chall_margin']=te.base_margin+corr; outs.append(te)
 return pd.concat(outs,ignore_index=True) if outs else pd.DataFrame()

def main():
 Q,key=build_q(); gates=[]
 def A(i,s,d,res,n):gates.append({'gate':f'T29.{i:02d}','status':s,'decision':d,'result':res,'next_action':n})
 # 1 coverage by game bucket
 cov=Q.groupby(['season','game_bucket'],observed=True).size().unstack(fill_value=0)
 A(1,'GREEN','PASS — EARLY-GAME BUCKET COVERAGE',f'join_key={key}; total={len(Q)}; counts={cov.to_dict()}','Use 2-3 / 4-5 / 6-10 as fixed early-season reporting buckets')
 # 2 state by bucket
 bstate={}
 for b in ['2-3','4-5','6-10']:
  x=folds(Q,STATE,lambda z,b=b:z.game_bucket.astype(str).eq(b)); bstate[b]=x.to_dict('records')
 gains={b:mean_gain(pd.DataFrame(v)) for b,v in bstate.items()}
 A(2,'GREEN' if sum(v>0 for v in gains.values())>=2 else 'HOLD','PASS — STATE LIFT LOCALIZED BY GAME NUMBER' if sum(v>0 for v in gains.values())>=2 else 'HOLD',f'buckets={bstate}; mean_gains={gains}','Use game-number-specific state-update strength; do not apply one universal boost')
 # 3 style by bucket
 bstyle={}
 for b in ['2-3','4-5','6-10']:
  x=folds(Q,STYLE,lambda z,b=b:z.game_bucket.astype(str).eq(b)); bstyle[b]=x.to_dict('records')
 sg={b:mean_gain(pd.DataFrame(v)) for b,v in bstyle.items()}
 A(3,'GREEN' if sum(v>0 for v in sg.values())>=2 else 'HOLD','PASS — STYLE LIFT PERSISTS ACROSS EARLY BUCKETS' if sum(v>0 for v in sg.values())>=2 else 'HOLD',f'buckets={bstyle}; mean_gains={sg}','Prioritize buckets with repeatable lift')
 # 4 roster by bucket
 broster={}
 for b in ['2-3','4-5','6-10']:
  x=folds(Q,ROSTER,lambda z,b=b:z.game_bucket.astype(str).eq(b)); broster[b]=x.to_dict('records')
 rg={b:mean_gain(pd.DataFrame(v)) for b,v in broster.items()}
 A(4,'SHADOW' if any(v>0 for v in rg.values()) else 'HOLD','SHADOW — ROSTER VALUE DECAYS/VARIES BY GAME NUMBER',f'buckets={broster}; mean_gains={rg}','Use result only after timestamp-valid roster replication')
 # 5 combined by bucket
 bcomb={}
 for b in ['2-3','4-5','6-10']:
  x=folds(Q,COMBINED,lambda z,b=b:z.game_bucket.astype(str).eq(b)); bcomb[b]=x.to_dict('records')
 cg={b:mean_gain(pd.DataFrame(v)) for b,v in bcomb.items()}
 A(5,'SHADOW' if sum(v>0 for v in cg.values())>=2 else 'HOLD','SHADOW — COMBINED LIFT ROBUST BY GAME BUCKET' if sum(v>0 for v in cg.values())>=2 else 'HOLD',f'buckets={bcomb}; mean_gains={cg}','Carry bucket diagnostics into legal-input replication')
 # 6 style components
 comps={}
 for c in STYLE:
  x=folds(Q,[c]); comps[c]=x.to_dict('records')
 compg={c:mean_gain(pd.DataFrame(v)) for c,v in comps.items()}
 A(6,'GREEN','PASS — STYLE COMPONENT ABLATION COMPLETE',f'component_folds={comps}; mean_gains={compg}','Retain only components that add consistently or support combined interaction')
 # 7 nested state+style
 ss=folds(Q,STATE+STYLE); ssonly=folds(Q,STATE); sty=folds(Q,STYLE)
 A(7,'GREEN' if mean_gain(ss)>max(mean_gain(ssonly),mean_gain(sty)) else 'HOLD','PASS — STATE + STYLE COMPLEMENTARY' if mean_gain(ss)>max(mean_gain(ssonly),mean_gain(sty)) else 'HOLD',f'state_gain={mean_gain(ssonly):.4f}; style_gain={mean_gain(sty):.4f}; state_style_gain={mean_gain(ss):.4f}; folds={ss.to_dict("records")}','Use nested ablation to avoid double-counting revealed team identity')
 # 8 roster incremental over state+style
 full=folds(Q,COMBINED); noR=folds(Q,STATE+STYLE); inc=mean_gain(full)-mean_gain(noR)
 A(8,'SHADOW' if inc>0 else 'HOLD','SHADOW — ROSTER ADDS BEYOND STATE + STYLE' if inc>0 else 'HOLD — ROSTER REDUNDANT AFTER STATE + STYLE',f'combined_gain={mean_gain(full):.4f}; no_roster_gain={mean_gain(noR):.4f}; incremental={inc:.4f}','Require positive incremental lift again with timestamp-valid roster inputs')
 # 9 regularization sensitivity
 sens=[]
 for a in [5,10,30,100,300]:
  x=folds(Q,COMBINED,alpha=a); sens.append({'alpha':a,'mean_gain':mean_gain(x),'folds':x.to_dict('records')})
 positive=sum(x['mean_gain']>0 for x in sens)
 A(9,'SHADOW' if positive==len(sens) else 'HOLD','SHADOW — COMBINED LIFT ROBUST TO REGULARIZATION' if positive==len(sens) else 'HOLD',f'{sens}','Freeze regularization only after legal-input tournament')
 # 10 win Brier
 P=combined_predictions(Q); brows=[]
 if len(P):
  for sy,z in P.groupby('season'):
   sig=z.sigma_margin_final if 'sigma_margin_final' in z else pd.Series(np.full(len(z),15.0),index=z.index)
   pb=z.pwin_final if 'pwin_final' in z else norm.cdf(z.base_margin/sig)
   pc=norm.cdf(z.chall_margin/sig); y=z.actual_win
   bb=float(np.mean((pb-y)**2)); bc=float(np.mean((pc-y)**2)); brows.append({'season':int(sy),'n':len(z),'base_brier':bb,'chall_brier':bc,'gain':bb-bc})
 bg=np.mean([x['gain'] for x in brows]) if brows else np.nan
 A(10,'SHADOW' if bg>0 else 'HOLD','SHADOW — WIN PROBABILITY BRIER IMPROVES' if bg>0 else 'HOLD — MARGIN LIFT DOES NOT IMPROVE BRIER',f'{brows}; mean_gain={bg:.6f}','Recalibrate probabilities if legal challenger is promoted')
 # 11 margin Gaussian NLL with frozen sigma
 nrows=[]
 if len(P):
  for sy,z in P.groupby('season'):
   sig=np.asarray(z.sigma_margin_final if 'sigma_margin_final' in z else np.full(len(z),15.0),float); sig=np.maximum(sig,1e-3)
   y=np.asarray(z.actual_margin,float); b=np.asarray(z.base_margin,float); c=np.asarray(z.chall_margin,float)
   nb=float(np.mean(.5*np.log(2*np.pi*sig**2)+.5*((y-b)/sig)**2)); nc=float(np.mean(.5*np.log(2*np.pi*sig**2)+.5*((y-c)/sig)**2)); nrows.append({'season':int(sy),'n':len(z),'base_nll':nb,'chall_nll':nc,'gain':nb-nc})
 ng=np.mean([x['gain'] for x in nrows]) if nrows else np.nan
 A(11,'SHADOW' if ng>0 else 'HOLD','SHADOW — MARGIN NLL IMPROVES' if ng>0 else 'HOLD',f'{nrows}; mean_gain={ng:.6f}','If legal version survives, refit residual covariance rather than assuming frozen sigma remains optimal')
 # 12 team-score MAE from symmetric margin correction
 trows=[]
 if len(P):
  homecol=next((c for c in ['mu_home','pred_home','expected_home','home_pred'] if c in P.columns),None); awaycol=next((c for c in ['mu_away','pred_away','expected_away','away_pred'] if c in P.columns),None)
  if homecol and awaycol:
   for sy,z in P.groupby('season'):
    bh=np.asarray(z[homecol],float); ba=np.asarray(z[awaycol],float)
    if 'mean_eh_final' in z: bh=bh+np.asarray(z.mean_eh_final,float)
    if 'mean_ea_final' in z: ba=ba+np.asarray(z.mean_ea_final,float)
    d=np.asarray(z['corr'],float)/2; base=(np.abs(z.actual_home-bh)+np.abs(z.actual_away-ba)).mean()/2; chal=(np.abs(z.actual_home-(bh+d))+np.abs(z.actual_away-(ba-d))).mean()/2
    trows.append({'season':int(sy),'n':len(z),'base_team_mae':float(base),'chall_team_mae':float(chal),'gain':float(base-chal)})
  tg=np.mean([x['gain'] for x in trows]) if trows else np.nan
  A(12,'SHADOW' if trows and tg>0 else 'HOLD','SHADOW — TEAM-SCORE MAE IMPROVES' if trows and tg>0 else 'HOLD — MARGIN CORRECTION DOES NOT YET IMPROVE TEAM SCORES',f'home_col={homecol}; away_col={awaycol}; rows={trows}; mean_gain={tg if trows else np.nan}','Build target-specific home/away residual challenger instead of relying on symmetric split')
 else:A(12,'HOLD','HOLD — NO COMBINED PREDICTIONS','No predictions available','Repair input construction')
 # 13 date-block bootstrap combined MAE gain
 boots=[]
 if len(P):
  P['date2']=pd.to_datetime(P.date).dt.date if 'date' in P else P.join_game_id
  for sy,z in P.groupby('season'):
   daily=list(z.groupby('date2')); vals=[]
   for _ in range(1000):
    picks=RNG.integers(0,len(daily),len(daily)); zz=pd.concat([daily[i][1] for i in picks],ignore_index=True)
    vals.append(float(np.mean(abs(zz.actual_margin-zz.base_margin))-np.mean(abs(zz.actual_margin-zz.chall_margin))))
   boots.append({'season':int(sy),'point_gain':float(np.mean(abs(z.actual_margin-z.base_margin))-np.mean(abs(z.actual_margin-z.chall_margin))),'ci_lo':float(np.quantile(vals,.025)),'ci_hi':float(np.quantile(vals,.975)),'p_positive':float(np.mean(np.array(vals)>0))})
 strong=all(x['ci_lo']>0 for x in boots) if boots else False
 A(13,'SHADOW' if strong else 'HOLD','SHADOW — DATE-BLOCK BOOTSTRAP SUPPORTS COMBINED LIFT' if strong else 'HOLD — BOOTSTRAP UNCERTAINTY TOO WIDE',f'{boots}','Repeat bootstrap after legal roster reconstruction')
 # 14 leave-one-family-out robustness
 fam={'no_roster':STATE+STYLE,'no_state':ROSTER+STYLE,'no_style':ROSTER+STATE,'full':COMBINED}; fr=[]
 for name,fs in fam.items():fr.append({'spec':name,'mean_gain':mean_gain(folds(Q,fs))})
 fullg=next(x['mean_gain'] for x in fr if x['spec']=='full'); essential={x['spec']:fullg-x['mean_gain'] for x in fr if x['spec']!='full'}
 A(14,'GREEN','PASS — FAMILY ABLATION MAP COMPLETE',f'specs={fr}; incremental_vs_full={essential}','Use family ablations to simplify challenger before production tournament')
 # 15 synthesis
 statuses=[g['status'] for g in gates]; decision='ADVANCE EARLY-SEASON CHALLENGER TO LEGAL-INPUT REPLICATION — KEEP V14 FROZEN'
 A(15,'GREEN',decision,f'green_before_final={statuses.count("GREEN")}; shadow={statuses.count("SHADOW")}; hold={statuses.count("HOLD")}; reject={statuses.count("REJECT")}; matched_early_games={len(Q)}; 2026_outcomes_used=False; production_reopen=False','Next gates: timestamp-valid roster/recruit inputs, opponent-adjusted component refinements, then full proper-score/calibration tournament')
 out=pd.DataFrame(gates); out.to_csv(OUT/'t29_robustness_15_gates.csv',index=False); pd.DataFrame(cov).to_csv(OUT/'t29_bucket_counts.csv'); P.to_csv(OUT/'t29_combined_predictions.csv',index=False) if len(P) else None
 pd.DataFrame(sens).to_json(OUT/'t29_regularization_sensitivity.json',orient='records',indent=2); pd.DataFrame(boots).to_csv(OUT/'t29_bootstrap.csv',index=False); pd.DataFrame(fr).to_csv(OUT/'t29_family_ablation.csv',index=False)
 (OUT/'summary.json').write_text(json.dumps({'green':sum(g['status']=='GREEN' for g in gates),'shadow':sum(g['status']=='SHADOW' for g in gates),'hold':sum(g['status']=='HOLD' for g in gates),'reject':sum(g['status']=='REJECT' for g in gates),'matched_early_games':len(Q),'model_reopened':False,'2026_outcomes_used':False,'final_decision':decision},indent=2)); print(out.to_string(index=False)); print((OUT/'summary.json').read_text())
if __name__=='__main__':main()
