from __future__ import annotations
import numpy as np
import pandas as pd
from scipy.stats import norm, t as student_t
from pathlib import Path

IN=Path('/tmp/third15')
OUT=Path('results/prediction_first_third15_qc'); OUT.mkdir(parents=True, exist_ok=True)
p=pd.read_parquet(IN/'corrected_game_predictions.parquet')

# T21.03: position-safe structural upset buckets.
rows=[]
labels=['0-3','3-6','6-10','10-15','15+']
for sy,q0 in p.groupby('season'):
    q=q0.reset_index(drop=True)
    sm=float(q.sigma_margin.iloc[0]); mm=float(q.resmean_margin.iloc[0])
    mu=q.pred_margin.to_numpy()+mm
    ph=norm.cdf(mu/sm); gap=np.abs(mu); pu=np.minimum(ph,1-ph)
    fav_home=mu>=0
    au=np.where(fav_home,q.actual_margin.to_numpy()<0,q.actual_margin.to_numpy()>0).astype(int)
    bucket=np.asarray(pd.cut(gap,[-1,3,6,10,15,np.inf],labels=labels).astype(str))
    for b in labels:
        ix=np.flatnonzero(bucket==b)
        if len(ix)==0: continue
        pp=pu[ix]; yy=au[ix]
        rows.append({'season':int(sy),'gap':b,'n':len(ix),'pred_upset':float(pp.mean()),'actual_upset':float(yy.mean()),'calibration_gap':float(yy.mean()-pp.mean()),'brier':float(np.mean((pp-yy)**2))})
pd.DataFrame(rows).to_csv(OUT/'t21_upset_fixed.csv',index=False)

# T20.05: state probabilities under promoted Student-t df=10 marginal,
# scaled to retain the Gaussian baseline variance.
rows=[]; df=10; mult=np.sqrt((df-2)/df)
for sy,q in p.groupby('season'):
    sm=float(q.sigma_margin.iloc[0]); mm=float(q.resmean_margin.iloc[0])
    mu=q.pred_margin.to_numpy()+mm; scale=sm*mult; a=q.actual_margin.to_numpy()
    pc=student_t.cdf((3.5-mu)/scale,df)-student_t.cdf((-3.5-mu)/scale,df)
    p10=student_t.cdf((-9.5-mu)/scale,df)+(1-student_t.cdf((9.5-mu)/scale,df))
    p20=student_t.cdf((-19.5-mu)/scale,df)+(1-student_t.cdf((19.5-mu)/scale,df))
    for name,pr,y in [('one_possession',pc,(np.abs(a)<=3).astype(int)),('margin_10plus',p10,(np.abs(a)>=10).astype(int)),('margin_20plus',p20,(np.abs(a)>=20).astype(int))]:
        rows.append({'season':int(sy),'state':name,'n':len(q),'pred_rate':float(pr.mean()),'actual_rate':float(y.mean()),'rate_gap':float(pr.mean()-y.mean()),'brier':float(np.mean((pr-y)**2))})
pd.DataFrame(rows).to_csv(OUT/'t20_game_states_t10.csv',index=False)
print(pd.DataFrame(rows).to_string(index=False))
