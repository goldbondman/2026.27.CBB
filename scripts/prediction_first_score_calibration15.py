from __future__ import annotations
from pathlib import Path
import json, hashlib
import numpy as np, pandas as pd
from scipy.stats import norm, multivariate_normal

OUT=Path('results/prediction_first_score_calibration15'); OUT.mkdir(parents=True,exist_ok=True)
FREEZE=Path('/tmp/freeze/frozen_candidate_predictions.parquet')
RNG=np.random.default_rng(20261011)


def mae(y,p): return float(np.mean(np.abs(np.asarray(y,float)-np.asarray(p,float))))
def brier(p,y):
 p=np.asarray(p,float); y=np.asarray(y,float); return float(np.mean((p-y)**2))
def logloss(p,y):
 p=np.clip(np.asarray(p,float),1e-9,1-1e-9); y=np.asarray(y,float); return float(np.mean(-(y*np.log(p)+(1-y)*np.log(1-p))))
def ece(p,y,bins=10):
 d=pd.DataFrame({'p':np.asarray(p,float),'y':np.asarray(y,float)}).dropna(); d['b']=np.minimum((d.p*bins).astype(int),bins-1)
 return float(sum(len(z)/len(d)*abs(z.p.mean()-z.y.mean()) for _,z in d.groupby('b'))) if len(d) else np.nan

def stage_bucket(n):
 if n<=1:return 'G1'
 if n<=3:return 'G2-3'
 if n<=5:return 'G4-5'
 if n<=10:return 'G6-10'
 return 'G11+'

def fit_affine(x,y):
 X=np.c_[np.ones(len(x)),np.asarray(x,float)]; return np.linalg.lstsq(X,np.asarray(y,float),rcond=None)[0]
def apply_affine(x,b): return b[0]+b[1]*np.asarray(x,float)

def prep():
 d=pd.read_parquet(FREEZE).copy(); d['date']=pd.to_datetime(d['date']);
 # Frozen production mean scores include the stage-conditioned residual means.
 d['base_home']=d['mu_home'].astype(float)+d.get('mean_eh_final',0).astype(float)
 d['base_away']=d['mu_away'].astype(float)+d.get('mean_ea_final',0).astype(float)
 d['base_margin']=d.base_home-d.base_away; d['base_total']=d.base_home+d.base_away
 d['actual_win']=(d.actual_margin>0).astype(float)
 d['stage_game']=np.floor(d['n_avg'].astype(float)*2+1e-9).astype(int)  # n_avg is mean prior games; approximate common stage
 # also retain original stage bucket; early subset defined directly by n_avg < 10 to avoid false exactness from averaging
 d['phase']=d.n_avg.map(stage_bucket)
 return d

def score_metrics(z,h,a):
 return {'n':len(z),'team_mae':(mae(z.actual_home,h)+mae(z.actual_away,a))/2,'margin_mae':mae(z.actual_margin,h-a),'total_mae':mae(z.actual_total,h+a)}

def fit_score_affine(tr):
 return fit_affine(tr.base_home,tr.actual_home),fit_affine(tr.base_away,tr.actual_away)
def predict_score_affine(te,bh,ba): return apply_affine(te.base_home,bh),apply_affine(te.base_away,ba)

def fit_cov(tr,h,a):
 E=np.c_[np.asarray(tr.actual_home-h,float),np.asarray(tr.actual_away-a,float)]; return E.mean(0),np.cov(E.T)+np.eye(2)*1e-6

def distribution_metrics(z,h,a,mean,cov):
 h2=h+mean[0]; a2=a+mean[1]; dm=h2-a2; dt=h2+a2
 sm=float(np.sqrt(cov[0,0]+cov[1,1]-2*cov[0,1])); st=float(np.sqrt(cov[0,0]+cov[1,1]+2*cov[0,1]))
 pw=norm.cdf(dm/sm); y=z.actual_win.values
 vals=np.c_[z.actual_home.values,z.actual_away.values]
 nll=float(-np.mean(multivariate_normal.logpdf(vals,mean=np.c_[h2,a2],cov=cov))) if False else np.nan
 # scipy's multivariate_normal does not vectorize row-wise means, so compute analytically
 inv=np.linalg.inv(cov); det=np.linalg.det(cov); R=vals-np.c_[h2,a2]
 nll=float(np.mean(np.log(2*np.pi)+.5*np.log(det)+.5*np.einsum('ij,jk,ik->i',R,inv,R)))
 return {'brier':brier(pw,y),'logloss':logloss(pw,y),'ece':ece(pw,y),'joint_nll':nll,'sigma_margin':sm,'sigma_total':st,'pwin':pw,'home':h2,'away':a2}

def frozen_dist_metrics(z):
 pw=z.pwin_final.values; vals=np.c_[z.actual_home.values,z.actual_away.values]
 # reconstruct stage covariance from frozen marginal sigmas + cov_final
 vh=(z.sigma_margin_final.values**2+z.sigma_total_final.values**2+4*z.cov_final.values)/4
 va=(z.sigma_margin_final.values**2+z.sigma_total_final.values**2-4*z.cov_final.values)/4
 out=[]
 for i,r in enumerate(z.itertuples(index=False)):
  cov=np.array([[vh[i],r.cov_final],[r.cov_final,va[i]]],float); mu=np.array([r.base_home,r.base_away]); x=np.array([r.actual_home,r.actual_away]);
  out.append(-multivariate_normal.logpdf(x,mean=mu,cov=cov,allow_singular=True))
 return {'brier':brier(pw,z.actual_win),'logloss':logloss(pw,z.actual_win),'ece':ece(pw,z.actual_win),'joint_nll':float(np.mean(out))}

def walk(train,test,phase=False):
 if not phase:
  bh,ba=fit_score_affine(train); ph,pa=predict_score_affine(test,bh,ba)
  th,ta=predict_score_affine(train,bh,ba); mean,cov=fit_cov(train,th,ta)
  return ph,pa,mean,cov,{'home_coef':bh.tolist(),'away_coef':ba.tolist()}
 ph=np.empty(len(test)); pa=np.empty(len(test)); info={}
 # fit phase mappings with fallback pooled
 pbh,pba=fit_score_affine(train)
 for s in ['G1','G2-3','G4-5','G6-10']:
  tr=train[train.phase.eq(s)]; ix=np.where(test.phase.eq(s))[0]
  if len(ix)==0: continue
  bh,ba=(fit_score_affine(tr) if len(tr)>=100 else (pbh,pba)); ph[ix],pa[ix]=predict_score_affine(test.iloc[ix],bh,ba); info[s]={'n_train':len(tr),'home':bh.tolist(),'away':ba.tolist()}
 th,ta=predict_score_affine(train,pbh,pba); mean,cov=fit_cov(train,th,ta)
 return ph,pa,mean,cov,info

def cov_by_phase(train,test,h,a):
 # means are supplied calibrated predictions. estimate covariance per phase around training calibrated predictions, shrunk to global.
 pbh,pba=fit_score_affine(train); th,ta=predict_score_affine(train,pbh,pba); gm,gc=fit_cov(train,th,ta)
 nll=[]; probs=[]
 for s in ['G1','G2-3','G4-5','G6-10']:
  tr=train[train.phase.eq(s)]; teix=np.where(test.phase.eq(s))[0]
  if len(teix)==0: continue
  t_h,t_a=predict_score_affine(tr,pbh,pba); lm,lc=fit_cov(tr,t_h,t_a) if len(tr)>=50 else (gm,gc); w=len(tr)/(len(tr)+300.0); mean=w*lm+(1-w)*gm; cov=w*lc+(1-w)*gc
  zz=test.iloc[teix]; hh=h[teix]+mean[0]; aa=a[teix]+mean[1]; R=np.c_[zz.actual_home.values-hh,zz.actual_away.values-aa]; inv=np.linalg.inv(cov); det=np.linalg.det(cov)
  nll.extend((np.log(2*np.pi)+.5*np.log(det)+.5*np.einsum('ij,jk,ik->i',R,inv,R)).tolist()); sm=np.sqrt(cov[0,0]+cov[1,1]-2*cov[0,1]); probs.extend(zip(teix,norm.cdf((hh-aa)/sm)))
 p=np.full(len(test),np.nan); [p.__setitem__(i,v) for i,v in probs]
 return float(np.mean(nll)),brier(p,test.actual_win),logloss(p,test.actual_win),ece(p,test.actual_win)

def coverage(train,test,h,a,levels=(.5,.8,.9)):
 pbh,pba=fit_score_affine(train); th,ta=predict_score_affine(train,pbh,pba); mean,cov=fit_cov(train,th,ta)
 sm=np.sqrt(cov[0,0]+cov[1,1]-2*cov[0,1]); st=np.sqrt(cov[0,0]+cov[1,1]+2*cov[0,1]); dm=h+a*0-a # h-a; explicit to avoid precedence confusion
 dm=h-a+(mean[0]-mean[1]); dt=h+a+(mean[0]+mean[1]); rows=[]
 for lev in levels:
  z=norm.ppf((1+lev)/2); cm=float(np.mean(np.abs(test.actual_margin.values-dm)<=z*sm)); ct=float(np.mean(np.abs(test.actual_total.values-dt)<=z*st)); rows.append({'level':lev,'margin_cov':cm,'total_cov':ct,'margin_gap':cm-lev,'total_gap':ct-lev})
 return rows

def main():
 d=prep(); early=d[(d.season.isin([2023,2024,2025]))&(d.n_avg<10)].copy(); gates=[]
 def A(i,s,dec,res,nxt): gates.append({'gate':f'T33.{i:02d}','status':s,'decision':dec,'result':res,'next_action':nxt})
 # 1 artifact/schema integrity
 need=['season','date','mu_home','mu_away','actual_home','actual_away','actual_margin','actual_total','n_avg','sigma_margin_final','sigma_total_final','pwin_final','mean_eh_final','mean_ea_final','cov_final']; missing=[c for c in need if c not in d]
 A(1,'GREEN' if not missing else 'HOLD','PASS — IMMUTABLE SCORE TABLE CONTAINS REQUIRED FIELDS' if not missing else 'HOLD',f'rows={len(d)}; early_rows={len(early)}; by_season={early.groupby("season").size().to_dict()}; missing={missing}','Use immutable frozen table directly; no reconstruction approximation')
 # 2 API mean reconstruction consistency
 pcalc=norm.cdf(early.base_margin/early.sigma_margin_final); err=float(np.max(np.abs(pcalc-early.pwin_final)))
 A(2,'GREEN' if err<1e-10 else 'HOLD','PASS — FROZEN SCORE MEANS RECONSTRUCT API WIN PROBABILITY' if err<1e-10 else 'HOLD',f'max_abs_probability_error={err:.12g}','Freeze base_home=mu_home+mean_eh_final and base_away=mu_away+mean_ea_final as exact score means')
 # folds 2023->24 and 23-24->25
 folds=[]
 for sy in [2024,2025]:
  tr=early[early.season<sy].copy(); te=early[early.season==sy].copy(); ph,pa,mean,cov,coef=walk(tr,te,False); base=score_metrics(te,te.base_home.values,te.base_away.values); chal=score_metrics(te,ph+mean[0],pa+mean[1]); fm=frozen_dist_metrics(te); cm=distribution_metrics(te,ph,pa,mean,cov); folds.append((sy,tr,te,base,chal,fm,cm,coef,ph,pa,mean,cov))
 # 3 score MAE replication
 rows=[]
 for sy,tr,te,b,c,fm,cm,coef,*_ in folds: rows.append({'season':sy,'n':len(te),'base_team':b['team_mae'],'cal_team':c['team_mae'],'team_gain':b['team_mae']-c['team_mae'],'base_margin':b['margin_mae'],'cal_margin':c['margin_mae'],'margin_gain':b['margin_mae']-c['margin_mae'],'base_total':b['total_mae'],'cal_total':c['total_mae'],'total_gain':b['total_mae']-c['total_mae'],'coef':coef})
 A(3,'GREEN' if all(r['team_gain']>0 and r['margin_gain']>0 and r['total_gain']>0 for r in rows) else 'HOLD','PASS — SCORE CALIBRATION IMPROVES TEAM/MARGIN/TOTAL IN BOTH FUTURE FOLDS' if all(r['team_gain']>0 and r['margin_gain']>0 and r['total_gain']>0 for r in rows) else 'HOLD',json.dumps(rows),'Advance score means only if proper scores also pass')
 # 4 Brier/logloss/ECE
 prows=[]
 for sy,tr,te,b,c,fm,cm,*_ in folds: prows.append({'season':sy,'base_brier':fm['brier'],'cal_brier':cm['brier'],'brier_gain':fm['brier']-cm['brier'],'base_logloss':fm['logloss'],'cal_logloss':cm['logloss'],'logloss_gain':fm['logloss']-cm['logloss'],'base_ece':fm['ece'],'cal_ece':cm['ece'],'ece_gain':fm['ece']-cm['ece']})
 A(4,'GREEN' if all(r['brier_gain']>0 and r['logloss_gain']>0 for r in prows) else 'HOLD','PASS — WIN PROPER SCORES IMPROVE IN BOTH FUTURE FOLDS' if all(r['brier_gain']>0 and r['logloss_gain']>0 for r in prows) else 'HOLD',json.dumps(prows),'Require ECE direction as supporting, not sole, criterion')
 # 5 joint score NLL
 nrows=[]
 for sy,tr,te,b,c,fm,cm,*_ in folds:nrows.append({'season':sy,'base_nll':fm['joint_nll'],'cal_nll':cm['joint_nll'],'gain':fm['joint_nll']-cm['joint_nll']})
 A(5,'GREEN' if all(r['gain']>0 for r in nrows) else 'HOLD','PASS — JOINT SCORE NLL IMPROVES IN BOTH FUTURE FOLDS' if all(r['gain']>0 for r in nrows) else 'HOLD',json.dumps(nrows),'Use calibrated score means with prior-only residual covariance')
 # 6 coverage
 covrows=[]
 for sy,tr,te,b,c,fm,cm,coef,ph,pa,mean,cov in folds:
  for r in coverage(tr,te,ph,pa): covrows.append({'season':sy,**r})
 maxgap=max(max(abs(r['margin_gap']),abs(r['total_gap'])) for r in covrows)
 A(6,'GREEN' if maxgap<=.04 else 'HOLD','PASS — CALIBRATED COVERAGE WITHIN 4PP' if maxgap<=.04 else 'HOLD — COVERAGE NEEDS REFINEMENT',json.dumps(covrows),'Refit stage covariance only if it improves NLL/coverage')
 # 7 date-block bootstrap point metrics
 boots=[]
 for sy,tr,te,b,c,fm,cm,coef,ph,pa,mean,cov in folds:
  temp=te.copy(); temp['ch']=ph+mean[0]-(pa+mean[1]); temp['chh']=ph+mean[0]; temp['cha']=pa+mean[1]; temp['cht']=temp.chh+temp.cha; daily=list(temp.groupby(temp.date.dt.date)); vals={'margin':[],'team':[],'total':[]}
  for _ in range(1000):
   pick=RNG.integers(0,len(daily),len(daily)); z=pd.concat([daily[i][1] for i in pick],ignore_index=True); vals['margin'].append(mae(z.actual_margin,z.base_margin)-mae(z.actual_margin,z.ch)); vals['team'].append(((mae(z.actual_home,z.base_home)+mae(z.actual_away,z.base_away))-(mae(z.actual_home,z.chh)+mae(z.actual_away,z.cha)))/2); vals['total'].append(mae(z.actual_total,z.base_total)-mae(z.actual_total,z.cht))
  for k,v in vals.items():boots.append({'season':sy,'target':k,'lo':float(np.quantile(v,.025)),'hi':float(np.quantile(v,.975)),'p':float(np.mean(np.asarray(v)>0))})
 strong=all(x['lo']>0 for x in boots)
 A(7,'GREEN' if strong else 'HOLD','PASS — SCORE GAINS SURVIVE DATE-BLOCK BOOTSTRAP' if strong else 'HOLD — SOME SCORE GAINS NOT BOOTSTRAP-ROBUST',json.dumps(boots),'Retain only targets whose clustered CI excludes zero')
 # 8 stage-conditioned covariance
 sc=[]
 for sy,tr,te,b,c,fm,cm,coef,ph,pa,mean,cov in folds:
  nll,br,ll,ec=cov_by_phase(tr,te,ph,pa); sc.append({'season':sy,'global_nll':cm['joint_nll'],'stage_nll':nll,'nll_gain':cm['joint_nll']-nll,'global_brier':cm['brier'],'stage_brier':br,'brier_gain':cm['brier']-br,'stage_logloss':ll,'stage_ece':ec})
 A(8,'GREEN' if all(x['nll_gain']>=0 for x in sc) else 'HOLD','PASS — STAGE COVARIANCE ADDS NONNEGATIVE JOINT NLL' if all(x['nll_gain']>=0 for x in sc) else 'HOLD — STAGE COVARIANCE NOT STABLE',json.dumps(sc),'Use stage covariance only if proper-score gains are fold-consistent')
 # 9 phase-specific score means vs pooled
 ps=[]
 for sy in [2024,2025]:
  tr=early[early.season<sy].copy(); te=early[early.season==sy].copy(); ph,pa,mean,cov,info=walk(tr,te,True); pooled=[x for x in folds if x[0]==sy][0]; pm=pooled[4]; m=score_metrics(te,ph+mean[0],pa+mean[1]); ps.append({'season':sy,'team_inc':pm['team_mae']-m['team_mae'],'margin_inc':pm['margin_mae']-m['margin_mae'],'total_inc':pm['total_mae']-m['total_mae'],'info':info})
 A(9,'GREEN' if all(x['margin_inc']>0 for x in ps) else 'HOLD','PASS SMALL — PHASE SCORE MEANS ADD MARGIN VALUE' if all(x['margin_inc']>0 for x in ps) else 'HOLD',json.dumps(ps),'Keep phase means only with strong shrinkage toward pooled mapping')
 # 10 Game 1 specific score calibration future-fold metrics
 g1=[]
 for sy in [2024,2025]:
  tr=d[(d.season<sy)&(d.n_avg<=1)].copy(); te=d[(d.season==sy)&(d.n_avg<=1)].copy(); bh,ba=fit_score_affine(tr); ph,pa=predict_score_affine(te,bh,ba); th,ta=predict_score_affine(tr,bh,ba); mean,cov=fit_cov(tr,th,ta); b=score_metrics(te,te.base_home.values,te.base_away.values); c=score_metrics(te,ph+mean[0],pa+mean[1]); g1.append({'season':sy,'n':len(te),'team_gain':b['team_mae']-c['team_mae'],'margin_gain':b['margin_mae']-c['margin_mae'],'total_gain':b['total_mae']-c['total_mae']})
 A(10,'GREEN' if all(x['margin_gain']>0 for x in g1) else 'HOLD','PASS — OPENING-GAME SCORE CALIBRATION REPLICATES' if all(x['margin_gain']>0 for x in g1) else 'HOLD',json.dumps(g1),'Treat Game 1 as strongest-compression phase')
 # 11 neutral vs non-neutral cohorts
 venue=[]
 if 'neutral' in early:
  for sy,tr,te,b,c,fm,cm,coef,ph,pa,mean,cov in folds:
   for val in [False,True]:
    ix=np.where(te.neutral.astype(bool).values==val)[0]
    if len(ix)<50:continue
    z=te.iloc[ix]; venue.append({'season':sy,'neutral':val,'n':len(z),'margin_gain':mae(z.actual_margin,z.base_margin)-mae(z.actual_margin,(ph+mean[0]-pa-mean[1])[ix]),'total_gain':mae(z.actual_total,z.base_total)-mae(z.actual_total,(ph+mean[0]+pa+mean[1])[ix])})
 A(11,'GREEN' if venue and all(x['margin_gain']>0 for x in venue) else 'HOLD','PASS — MARGIN CALIBRATION GENERALIZES ACROSS VENUE COHORTS' if venue and all(x['margin_gain']>0 for x in venue) else 'HOLD',json.dumps(venue),'Retain neutral reliability metadata even if mean calibration generalizes')
 # 12 calibration coefficient stability
 co=[]
 for sy in [2023,2024,2025]:
  z=early[early.season==sy]; bh,ba=fit_score_affine(z); co.append({'season':sy,'home_intercept':bh[0],'home_slope':bh[1],'away_intercept':ba[0],'away_slope':ba[1]})
 A(12,'GREEN','PASS — HOME/AWAY CALIBRATION COEFFICIENTS AUDITED ACROSS THREE SEASONS',json.dumps(co),'Use expanding-history estimates; do not freeze single-season coefficients')
 # 13 rolling-last-year vs expanding history
 rr=[]
 for sy in [2024,2025]:
  te=early[early.season==sy]; tr_last=early[early.season==sy-1]; tr_exp=early[early.season<sy]
  for name,tr in [('last',tr_last),('expanding',tr_exp)]:
   bh,ba=fit_score_affine(tr); ph,pa=predict_score_affine(te,bh,ba); th,ta=predict_score_affine(tr,bh,ba); mean,cov=fit_cov(tr,th,ta); m=score_metrics(te,ph+mean[0],pa+mean[1]); rr.append({'season':sy,'spec':name,**m})
 A(13,'GREEN','PASS — TRAINING-WINDOW STABILITY MAPPED',json.dumps(rr),'Prefer expanding history unless last-year materially and consistently wins')
 # 14 artifact determinism / canonical hash
 sha=hashlib.sha256(FREEZE.read_bytes()).hexdigest(); A(14,'GREEN','PASS — CERTIFICATION USES IMMUTABLE FROZEN TABLE',f'sha256={sha}; rows={len(d)}','Record hash in tracker and require identical comparator for reopen tournament')
 # 15 synthesis
 statuses=[g['status'] for g in gates]; critical=[g for g in gates if g['gate'] in ['T33.03','T33.04','T33.05','T33.07']]
 ready=all(g['status']=='GREEN' for g in critical)
 A(15,'GREEN' if ready else 'HOLD','READY FOR FORMAL PRODUCTION-REOPEN TOURNAMENT — V14 STILL FROZEN' if ready else 'HOLD — FULL SCORE DISTRIBUTION NOT YET CERTIFIED',f'green_before_final={statuses.count("GREEN")}; hold={statuses.count("HOLD")}; critical_pass={ready}; 2026_outcomes_used=False; production_reopen=False','If ready, next gate family should be explicit incumbent-vs-calibrated production tournament with predeclared promotion rules')
 pd.DataFrame(gates).to_csv(OUT/'t33_score_calibration_15_gates.csv',index=False)
 pd.DataFrame(rows).to_csv(OUT/'t33_score_mae_folds.csv',index=False); pd.DataFrame(prows).to_csv(OUT/'t33_probability_folds.csv',index=False); pd.DataFrame(nrows).to_csv(OUT/'t33_joint_nll_folds.csv',index=False); pd.DataFrame(covrows).to_csv(OUT/'t33_coverage.csv',index=False); pd.DataFrame(boots).to_csv(OUT/'t33_bootstrap.csv',index=False); pd.DataFrame(sc).to_csv(OUT/'t33_stage_covariance.csv',index=False); pd.DataFrame(ps).to_json(OUT/'t33_phase_means.json',orient='records',indent=2); pd.DataFrame(co).to_csv(OUT/'t33_coefficients.csv',index=False); pd.DataFrame(rr).to_csv(OUT/'t33_training_window.csv',index=False)
 (OUT/'summary.json').write_text(json.dumps({'gates':gates,'freeze_sha256':sha},indent=2))
 print(pd.DataFrame(gates).to_string(index=False))

if __name__=='__main__': main()
