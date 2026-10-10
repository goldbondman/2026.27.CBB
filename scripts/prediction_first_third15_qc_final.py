from __future__ import annotations
import math, json, numpy as np, pandas as pd
from scipy.stats import norm
import prediction_first_third15_followup as f
import prediction_first_third15 as m
OUT=m.OUT

def context_params(r,tr,kind):
    E=tr[['e_home','e_away']].values; gm=E.mean(0); gc=np.cov(E.T)+np.eye(2)*1e-6
    if kind=='global': mm,cc=gm,gc
    else:
        q=tr[tr.stage==r.stage] if kind=='stage' else tr[(tr.stage==r.stage)&(tr.neutral==r.neutral)]
        w=len(q)/(len(q)+500.0)
        if len(q)>20:
            lm=q[['e_home','e_away']].mean().values; lc=np.cov(q[['e_home','e_away']].values.T)+np.eye(2)*1e-6; mm=w*lm+(1-w)*gm; cc=w*lc+(1-w)*gc
        else:mm,cc=gm,gc
    return float(mm[0]-mm[1]),float(math.sqrt(cc[0,0]+cc[1,1]-2*cc[0,1]))

def states(folds):
    rows=[]
    for sy,tr,q in folds:
        for kind in ['global','stage','stagevenue']:
            ps={'close':[],'m10':[],'m20':[]}; ys={'close':[],'m10':[],'m20':[]}
            for r in q.itertuples(index=False):
                mm,sd=context_params(r,tr,kind); mu=(r.mu_home-r.mu_away)+mm
                ps['close'].append(norm.cdf((3.5-mu)/sd)-norm.cdf((-3.5-mu)/sd)); ps['m10'].append(norm.cdf((-9.5-mu)/sd)+1-norm.cdf((9.5-mu)/sd)); ps['m20'].append(norm.cdf((-19.5-mu)/sd)+1-norm.cdf((19.5-mu)/sd)); a=r.actual_margin; ys['close'].append(abs(a)<=3); ys['m10'].append(abs(a)>=10); ys['m20'].append(abs(a)>=20)
            for st in ps:
                p=np.asarray(ps[st]); y=np.asarray(ys[st],float); rows.append({'season':sy,'variant':kind,'state':st,'n':len(y),'pred_rate':float(p.mean()),'actual_rate':float(y.mean()),'abs_rate_gap':float(abs(p.mean()-y.mean())),'brier':float(np.mean((p-y)**2))})
    R=pd.DataFrame(rows); R.to_csv(OUT/'t20_state_context_qc.csv',index=False); return R

def neutral(folds):
    rows=[]
    for sy,tr,q0 in folds:
        q=q0[q0.neutral]; nt=tr[tr.neutral]
        if len(q)<50 or len(nt)<100:continue
        gm=float(tr.mr.mean()); gs=float(tr.mr.std(ddof=1)); nm=float(nt.mr.mean()); ns=float(nt.mr.std(ddof=1))
        for var,mu,sd in [('global',gm,gs),('neutral_mean',nm,gs),('neutral_sd',gm,ns),('neutral_mean_sd',nm,ns)]:
            pred_mu=(q.mu_home-q.mu_away)+mu; p=norm.cdf(pred_mu/sd); y=(q.actual_margin>0).astype(int); nll=float(np.mean(.5*np.log(2*np.pi*sd*sd)+.5*((q.mr-mu)/sd)**2)); rows.append({'season':sy,'variant':var,'n':len(q),'train_neutral_n':len(nt),'train_neutral_mean':nm,'train_neutral_sd':ns,'nll':nll,'mae':float(np.mean(abs(q.mr-mu))),'bias':float(np.mean(q.mr-mu)),'brier':m.brier(p,y),'logloss':m.logloss(p,y),'ece':m.ece(p,y)})
    R=pd.DataFrame(rows); R.to_csv(OUT/'t21_neutral_treatment_qc.csv',index=False); return R

def main():
    gp,r2,pace,cf,venue,sched=f.load_all(); raw,pred,metrics,use=m.build_fold_baselines(gp,r2,pace,venue,sched); pm=f.postmap(); folds=f.prep(raw,use,pm); S=states(folds); N=neutral(folds)
    ss={v:{'mean_abs_rate_gap':float(q.abs_rate_gap.mean()),'mean_brier':float(q.brier.mean())} for v,q in S.groupby('variant')}
    ns={}
    for v,q in N.groupby('variant'):
        base=N[N.variant=='global'].set_index('season'); cur=q.set_index('season'); ns[v]={'mean_nll':float(q.nll.mean()),'mean_brier':float(q.brier.mean()),'mean_logloss':float(q.logloss.mean()),'mean_ece':float(q.ece.mean()),'mean_abs_bias':float(abs(q.bias).mean()),'nll_wins_vs_global':int((cur.nll<base.nll).sum()) if v!='global' else 0}
    z={'state_variants':ss,'neutral_variants':ns}; (OUT/'qc_final_summary.json').write_text(json.dumps(z,indent=2)); print(json.dumps(z,indent=2))
if __name__=='__main__':main()
