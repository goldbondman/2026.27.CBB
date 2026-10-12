from __future__ import annotations
import json
from pathlib import Path
import numpy as np, pandas as pd
from scipy.stats import norm
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import mean_absolute_error
import prediction_first_calibrated_recency15 as t35

OUT=Path('results/prediction_first_recency_cert15'); OUT.mkdir(parents=True,exist_ok=True)
RNG=np.random.default_rng(20261012)

def mae(a,b): return float(mean_absolute_error(a,b))
def brier(y,p): return float(np.mean((np.asarray(y)-np.asarray(p))**2))
def logloss(y,p):
 p=np.clip(np.asarray(p),1e-6,1-1e-6); y=np.asarray(y)
 return float(np.mean(-(y*np.log(p)+(1-y)*np.log(1-p))))
def ece(y,p,bins=10):
 y=np.asarray(y); p=np.asarray(p); edges=np.linspace(0,1,bins+1); out=0.0
 for i in range(bins):
  m=(p>=edges[i])&(p<(edges[i+1] if i<bins-1 else edges[i+1]+1e-12))
  if m.any(): out += m.mean()*abs(float(y[m].mean())-float(p[m].mean()))
 return float(out)

def prep():
 a,q=t35.b.build(); q=t35.add_model_adjusted_form(a,q)
 if 'actual_home' not in q and 'factor_actual_home' in q: q['actual_home']=q.factor_actual_home
 if 'actual_away' not in q and 'factor_actual_away' in q: q['actual_away']=q.factor_actual_away
 q['actual_total']=q.actual_home+q.actual_away; q['base_total']=q.base_home+q.base_away
 dates=a[['season','game_id','game_date']].drop_duplicates(['season','game_id'])
 q=q.merge(dates,on=['season','game_id'],how='left')
 return a,q

def fit_margin(q,feature,sy):
 tr=q[q.season.lt(sy)].copy(); te=q[q.season.eq(sy)].copy(); tr,te,co=t35.calibrated_baseline(tr,te)
 tr=tr.dropna(subset=['cal_resid',feature]); te=te.dropna(subset=['cal_resid',feature]).copy()
 m=make_pipeline(StandardScaler(),Ridge(alpha=30)).fit(tr[[feature]],tr.cal_resid)
 te['pred_margin']=te.cal_margin+m.predict(te[[feature]])
 te['cal_gain_abs']=abs(te.actual_margin-te.cal_margin)-abs(te.actual_margin-te.pred_margin)
 return tr,te,co,m

def affine_target(train,test,target,base):
 z=train.dropna(subset=[target,base]); X=np.c_[np.ones(len(z)),z[base].to_numpy()]; co=np.linalg.lstsq(X,z[target].to_numpy(),rcond=None)[0]
 tr=train.copy(); te=test.copy(); tr['cal_target']=co[0]+co[1]*tr[base]; te['cal_target']=co[0]+co[1]*te[base]
 return tr,te,co

def score_preds(q,sy,features,target,base):
 tr=q[q.season.lt(sy)].copy(); te=q[q.season.eq(sy)].copy(); tr,te,co=affine_target(tr,te,target,base)
 tr['resid_target']=tr[target]-tr.cal_target
 av=[c for c in features if c in q.columns]
 tr2=tr.dropna(subset=['resid_target']+av); te2=te.dropna(subset=[target,'cal_target']+av).copy()
 m=make_pipeline(StandardScaler(),Ridge(alpha=30)).fit(tr2[av],tr2.resid_target); te2['pred']=te2.cal_target+m.predict(te2[av])
 return tr2,te2,co,m

def block_boot(df,delta_col,n=1000):
 x=df.dropna(subset=['game_date',delta_col]).copy(); x['game_date']=x.game_date.astype(str)
 dates=x.game_date.unique(); by={d:x.loc[x.game_date.eq(d),delta_col].to_numpy() for d in dates}; vals=[]
 for _ in range(n):
  ds=RNG.choice(dates,size=len(dates),replace=True); arr=np.concatenate([by[d] for d in ds]); vals.append(float(arr.mean()))
 return {'point':float(x[delta_col].mean()),'lo':float(np.quantile(vals,.025)),'hi':float(np.quantile(vals,.975)),'p_pos':float(np.mean(np.array(vals)>0)),'n':len(x),'dates':len(dates)}

def main():
 a,q=prep(); G=[]
 def A(i,s,d,r,n): G.append({'gate':f'T36.{i:02d}','status':s,'decision':d,'result':r,'next_action':n})
 # 1 locked substrate
 A(1,'GREEN','PASS — T36 CERTIFICATION SUBSTRATE LOCKED',f'joined_games={len(q)}; seasons={q.groupby("season").size().to_dict()}; date_coverage={q.game_date.notna().mean():.4f}; 2026_outcomes_used=False','Use adjusted L7 as preregistered primary challenger')
 # 2 adjusted L7 date-block bootstrap
 boots=[]; preds={}
 for sy in [2024,2025]:
  _,te,_,_=fit_margin(q,'d_adj_l7_margin',sy); preds[sy]=te; boots.append({'season':sy,**block_boot(te,'cal_gain_abs',1000)})
 strong=all(r['lo']>0 for r in boots)
 A(2,'GREEN' if strong else 'HOLD','PASS — ADJUSTED L7 DATE-BLOCK ROBUST' if strong else 'HOLD — ADJUSTED L7 BOOTSTRAP CROSSES ZERO',str(boots),'Require positive CIs before any architecture tournament')
 # 3 recursive increment paired bootstrap vs adjusted L7
 recdelta=[]
 for sy in [2024,2025]:
  _,a7,_,_=fit_margin(q,'d_adj_l7_margin',sy); _,r7,_,_=fit_margin(q,'d_rec_l7_margin',sy)
  z=a7[['game_id','game_date','actual_margin','pred_margin']].rename(columns={'pred_margin':'adj'}).merge(r7[['game_id','pred_margin']],on='game_id').rename(columns={'pred_margin':'rec'})
  z['delta']=abs(z.actual_margin-z.adj)-abs(z.actual_margin-z.rec); recdelta.append({'season':sy,**block_boot(z,'delta',1000)})
 recwin=all(r['lo']>0 for r in recdelta)
 A(3,'GREEN' if recwin else 'REJECT','PASS — RECURSIVE L7 MATERIALLY BEATS ADJUSTED L7' if recwin else 'REJECT — RECURSIVE L7 INCREMENT NOT ROBUST',str(recdelta),'Prefer adjusted L7 if recursion fails paired uncertainty test')
 # 4 game-number cohort stability for adjusted L7
 cohort=[]
 for sy,te in preds.items():
  mx=te[['h_game_no','a_game_no']].max(axis=1)
  for name,mask in [('early_1_10',mx.le(10)),('middle_11_20',mx.between(11,20)),('late_21plus',mx.ge(21))]:
   x=te[mask]
   if len(x)>=100: cohort.append({'season':sy,'cohort':name,'n':len(x),'gain':float(x.cal_gain_abs.mean())})
 stable=all(r['gain']>0 for r in cohort)
 A(4,'GREEN' if stable else 'HOLD','PASS — ADJUSTED L7 POSITIVE ACROSS GAME-STAGE COHORTS' if stable else 'HOLD — ADJUSTED L7 STAGE INSTABILITY',str(cohort),'Retain only cohorts with replicated positive direction')
 # 5 calibration early vs mature contribution
 c5=[]
 for sy,te in preds.items():
  mx=te[['h_game_no','a_game_no']].max(axis=1)
  for name,mask in [('within_10',mx.le(10)),('over_10',mx.gt(10))]:
   x=te[mask]
   if len(x)>=100: c5.append({'season':sy,'cohort':name,'n':len(x),'gain':float(x.cal_gain_abs.mean())})
 A(5,'GREEN','PASS — RECENCY GAIN DECOMPOSED AROUND CALIBRATION WINDOW',str(c5),'Use this to distinguish early calibration from mature recency value')
 # 6-7 direct home/away score targets from team-side adjusted L7
 home_rows=[]; away_rows=[]; hp={}; ap={}
 score_features=['h_adj_l7_margin','a_adj_l7_margin']
 for sy in [2024,2025]:
  _,h,_,_=score_preds(q,sy,score_features,'actual_home','base_home'); _,aw,_,_=score_preds(q,sy,score_features,'actual_away','base_away')
  hp[sy]=h; ap[sy]=aw
  home_rows.append({'season':sy,'n':len(h),'base_mae':mae(h.actual_home,h.cal_target),'challenger_mae':mae(h.actual_home,h.pred),'gain':mae(h.actual_home,h.cal_target)-mae(h.actual_home,h.pred)})
  away_rows.append({'season':sy,'n':len(aw),'base_mae':mae(aw.actual_away,aw.cal_target),'challenger_mae':mae(aw.actual_away,aw.pred),'gain':mae(aw.actual_away,aw.cal_target)-mae(aw.actual_away,aw.pred)})
 A(6,'GREEN' if all(r['gain']>0 for r in home_rows) else 'HOLD','PASS — ADJUSTED L7 IMPROVES HOME SCORE' if all(r['gain']>0 for r in home_rows) else 'HOLD — HOME SCORE DOES NOT REPLICATE',str(home_rows),'Carry only if positive both folds')
 A(7,'GREEN' if all(r['gain']>0 for r in away_rows) else 'HOLD','PASS — ADJUSTED L7 IMPROVES AWAY SCORE' if all(r['gain']>0 for r in away_rows) else 'HOLD — AWAY SCORE DOES NOT REPLICATE',str(away_rows),'Carry only if positive both folds')
 # 8 coherent score derived margin/total metrics
 combo=[]; joined={}
 for sy in [2024,2025]:
  h=hp[sy][['game_id','actual_home','cal_target','pred']].rename(columns={'cal_target':'cal_h','pred':'pred_h'}); aw=ap[sy][['game_id','actual_away','cal_target','pred']].rename(columns={'cal_target':'cal_a','pred':'pred_a'})
  z=h.merge(aw,on='game_id'); z['actual_margin']=z.actual_home-z.actual_away; z['actual_total']=z.actual_home+z.actual_away; z['cal_m']=z.cal_h-z.cal_a; z['pred_m']=z.pred_h-z.pred_a; z['cal_t']=z.cal_h+z.cal_a; z['pred_t']=z.pred_h+z.pred_a; joined[sy]=z
  combo.append({'season':sy,'n':len(z),'team_gain':((abs(z.actual_home-z.cal_h)+abs(z.actual_away-z.cal_a))/2).mean()-((abs(z.actual_home-z.pred_h)+abs(z.actual_away-z.pred_a))/2).mean(),'margin_gain':mae(z.actual_margin,z.cal_m)-mae(z.actual_margin,z.pred_m),'total_gain':mae(z.actual_total,z.cal_t)-mae(z.actual_total,z.pred_t)})
 coh=all(r['team_gain']>0 and r['margin_gain']>0 and r['total_gain']>0 for r in combo)
 A(8,'GREEN' if coh else 'HOLD','PASS — DIRECT SCORE RECENCY IMPROVES TEAM/MARGIN/TOTAL' if coh else 'HOLD — DIRECT SCORE RECENCY TARGET TRADEOFF',str(combo),'Require coherent score-target improvement')
 # 9 joint score NLL with training-derived residual covariance
 nll=[]
 for sy in [2024,2025]:
  tr=q[q.season.lt(sy)].copy(); f=score_features
  trh,_,_,mh=score_preds(q,sy,f,'actual_home','base_home'); tra,_,_,ma=score_preds(q,sy,f,'actual_away','base_away')
  # Build common training residuals by refitting calibrated targets/predictions in training rows directly.
  # Simpler conservative covariance: use affine baseline training residual covariance and compare same covariance for challenger means.
  th,_,_=affine_target(tr,tr,'actual_home','base_home'); ta,_,_=affine_target(tr,tr,'actual_away','base_away'); rr=pd.DataFrame({'eh':th.actual_home-th.cal_target,'ea':ta.actual_away-ta.cal_target}).dropna(); cov=np.cov(rr[['eh','ea']].to_numpy().T); inv=np.linalg.inv(cov); ld=np.linalg.slogdet(cov)[1]
  z=joined[sy]; eb=np.c_[z.actual_home-z.cal_h,z.actual_away-z.cal_a]; ec=np.c_[z.actual_home-z.pred_h,z.actual_away-z.pred_a]
  def jn(e): return float(np.mean(np.log(2*np.pi)+.5*ld+.5*np.einsum('ni,ij,nj->n',e,inv,e)))
  bn=jn(eb); cn=jn(ec); nll.append({'season':sy,'base_nll':bn,'challenger_nll':cn,'gain':bn-cn})
 A(9,'GREEN' if all(r['gain']>0 for r in nll) else 'HOLD','PASS — DIRECT SCORE RECENCY IMPROVES JOINT NLL' if all(r['gain']>0 for r in nll) else 'HOLD — JOINT NLL DOES NOT REPLICATE',str(nll),'If positive, proceed to probability calibration')
 # 10 win-probability proper scores from direct score margin and train-derived margin sigma
 prows=[]
 for sy in [2024,2025]:
  tr=q[q.season.lt(sy)].copy(); trh,_,_=affine_target(tr,tr,'actual_home','base_home'); tra,_,_=affine_target(tr,tr,'actual_away','base_away'); sig=float(np.std((trh.actual_home-trh.cal_target)-(tra.actual_away-tra.cal_target),ddof=1))
  z=joined[sy]; y=(z.actual_margin>0).astype(int); pb=norm.cdf(z.cal_m/sig); pc=norm.cdf(z.pred_m/sig)
  prows.append({'season':sy,'n':len(z),'sigma':sig,'brier_gain':brier(y,pb)-brier(y,pc),'logloss_gain':logloss(y,pb)-logloss(y,pc),'ece_base':ece(y,pb),'ece_challenger':ece(y,pc),'ece_gain':ece(y,pb)-ece(y,pc)})
 ppass=all(r['brier_gain']>=0 and r['logloss_gain']>=0 for r in prows)
 A(10,'GREEN' if ppass else 'HOLD','PASS — DIRECT SCORE RECENCY CLEARS WIN PROPER SCORES' if ppass else 'HOLD — DIRECT SCORE RECENCY WIN PROPER SCORES MIXED',str(prows),'No betting-layer promotion unless all future folds non-negative')
 # 11 uncertainty/coverage of calibrated direct score margin
 cover=[]
 for sy in [2024,2025]:
  tr=q[q.season.lt(sy)].copy(); trh,_,_=affine_target(tr,tr,'actual_home','base_home'); tra,_,_=affine_target(tr,tr,'actual_away','base_away'); sig=float(np.std((trh.actual_home-trh.cal_target)-(tra.actual_away-tra.cal_target),ddof=1)); z=joined[sy]; err=z.actual_margin-z.pred_m
  for level,zv in [(0.5,.67448975),(0.8,1.28155157),(0.9,1.64485363)]: cover.append({'season':sy,'level':level,'coverage':float((abs(err)<=zv*sig).mean()),'error':float((abs(err)<=zv*sig).mean()-level)})
 A(11,'GREEN','PASS — DIRECT SCORE RECENCY COVERAGE MAPPED',str(cover),'Refit scale only if systematic coverage bias replicates')
 # 12 TS increment beyond adjusted L7 margin
 tsinc=[]
 for sy in [2024,2025]:
  tr=q[q.season.lt(sy)].copy(); te=q[q.season.eq(sy)].copy(); tr,te,_=t35.calibrated_baseline(tr,te)
  basef=['d_adj_l7_margin']; fullf=['d_adj_l7_margin','d_pre_ts']; ztr=tr.dropna(subset=['cal_resid']+fullf); zte=te.dropna(subset=['cal_resid']+fullf).copy(); m1=make_pipeline(StandardScaler(),Ridge(alpha=30)).fit(ztr[basef],ztr.cal_resid); m2=make_pipeline(StandardScaler(),Ridge(alpha=30)).fit(ztr[fullf],ztr.cal_resid); p1=zte.cal_margin+m1.predict(zte[basef]); p2=zte.cal_margin+m2.predict(zte[fullf]); tsinc.append({'season':sy,'n':len(zte),'adj_l7_gain':mae(zte.actual_margin,zte.cal_margin)-mae(zte.actual_margin,p1),'adj_l7_ts_gain':mae(zte.actual_margin,zte.cal_margin)-mae(zte.actual_margin,p2),'increment':mae(zte.actual_margin,p1)-mae(zte.actual_margin,p2)})
 A(12,'GREEN' if all(r['increment']>0 for r in tsinc) else 'HOLD','PASS — TS ADDS BEYOND ADJUSTED L7' if all(r['increment']>0 for r in tsinc) else 'HOLD — TS DOES NOT ADD CONSISTENTLY BEYOND ADJUSTED L7',str(tsinc),'Retain shooting state only if incremental in both folds')
 # 13 non-shooting factors beyond adjusted L7+TS
 frows=[]; full=['d_adj_l7_margin','d_pre_ts','d_pre_tovr','d_pre_orbr','d_pre_drbr','d_pre_ftr']; simple=['d_adj_l7_margin','d_pre_ts']
 for sy in [2024,2025]:
  tr=q[q.season.lt(sy)].copy(); te=q[q.season.eq(sy)].copy(); tr,te,_=t35.calibrated_baseline(tr,te); ztr=tr.dropna(subset=['cal_resid']+full); zte=te.dropna(subset=['cal_resid']+full).copy(); m1=make_pipeline(StandardScaler(),Ridge(alpha=30)).fit(ztr[simple],ztr.cal_resid); m2=make_pipeline(StandardScaler(),Ridge(alpha=30)).fit(ztr[full],ztr.cal_resid); p1=zte.cal_margin+m1.predict(zte[simple]); p2=zte.cal_margin+m2.predict(zte[full]); frows.append({'season':sy,'n':len(zte),'increment':mae(zte.actual_margin,p1)-mae(zte.actual_margin,p2)})
 A(13,'GREEN' if all(r['increment']>0 for r in frows) else 'REJECT','PASS — NON-SHOOTING FACTORS ADD BEYOND L7+TS' if all(r['increment']>0 for r in frows) else 'REJECT — NON-SHOOTING FACTORS DO NOT ADD CONSISTENTLY BEYOND L7+TS',str(frows),'Simplify factor layer if non-shooting increment fails')
 # 14 neutral/non-neutral adjusted L7 stability
 vrows=[]
 neutral_col='neutral' if 'neutral' in q.columns else None
 for sy,te in preds.items():
  if neutral_col:
   for name,mask in [('non_neutral',~te[neutral_col].fillna(False).astype(bool)),('neutral',te[neutral_col].fillna(False).astype(bool))]:
    x=te[mask]
    if len(x)>=100: vrows.append({'season':sy,'cohort':name,'n':len(x),'gain':float(x.cal_gain_abs.mean())})
 A(14,'GREEN' if vrows and all(r['gain']>0 for r in vrows) else 'HOLD','PASS — ADJUSTED L7 POSITIVE BY VENUE COHORT' if vrows and all(r['gain']>0 for r in vrows) else 'HOLD — ADJUSTED L7 VENUE INSTABILITY/UNAVAILABLE',str(vrows),'Do not specialize by venue unless cohort differences are stable/material')
 # 15 synthesis
 statuses=[g['status'] for g in G]; decision='ADVANCE ADJUSTED-L7 MEAN CHALLENGER; PROBABILITY PROMOTION ONLY IF T36 PROPER-SCORE GATES CLEAR'
 A(15,'GREEN',decision,f'green_before_final={statuses.count("GREEN")}; hold={statuses.count("HOLD")}; reject={statuses.count("REJECT")}; production_reopen=False; 2026_outcomes_used=False','Next: common-fold challenger tournament if score/proper-score gates support it; otherwise keep recency mean-only research')
 pd.DataFrame(G).to_csv(OUT/'t36_recency_cert_15_gates.csv',index=False)
 pd.DataFrame(boots).to_csv(OUT/'t36_adjusted_l7_bootstrap.csv',index=False); pd.DataFrame(recdelta).to_csv(OUT/'t36_recursive_delta_bootstrap.csv',index=False); pd.DataFrame(combo).to_csv(OUT/'t36_score_metrics.csv',index=False); pd.DataFrame(nll).to_csv(OUT/'t36_joint_nll.csv',index=False); pd.DataFrame(prows).to_csv(OUT/'t36_probability.csv',index=False); pd.DataFrame(cover).to_csv(OUT/'t36_coverage.csv',index=False)
 (OUT/'summary.json').write_text(json.dumps({'green':sum(g['status']=='GREEN' for g in G),'hold':sum(g['status']=='HOLD' for g in G),'reject':sum(g['status']=='REJECT' for g in G),'production_reopen':False,'outcomes_2026_used':False,'decision':decision},indent=2))
 print(pd.DataFrame(G).to_string(index=False))
if __name__=='__main__': main()
