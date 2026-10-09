from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import norm
OUT=Path('results/prediction_first_neutral_calibration'); OUT.mkdir(parents=True,exist_ok=True)
g=pd.read_parquet('/tmp/g15/game_predictions.parquet'); m=pd.read_csv('/tmp/venue/identity_match_ledger.csv',dtype={'game_id':str}); g['game_id']=g.game_id.astype(str); z=g.merge(m[['game_id','neutral']],on='game_id',how='inner'); z=z[z.neutral.notna()].copy(); z['neutral']=z.neutral.astype(bool); z['pred_diff']=z.pred0-z.pred1; z['actual_diff']=z.actual0-z.actual1; z['resid']=z.actual_diff-z.pred_diff

def ece(p,y,bins=10):
 d=pd.DataFrame({'p':p,'y':y}); d['b']=np.minimum((d.p*bins).astype(int),bins-1); return float(sum(len(q)/len(d)*abs(q.p.mean()-q.y.mean()) for _,q in d.groupby('b')))
def ll(p,y):
 p=np.clip(np.asarray(p,float),1e-9,1-1e-9); y=np.asarray(y,float); return float(np.mean(-(y*np.log(p)+(1-y)*np.log(1-p))))
rows=[]
for sy in [2023,2024,2025]:
 tr=z[(z.season>=2022)&(z.season<sy)&z.neutral].copy(); te=z[(z.season==sy)&z.neutral].copy(); mu=float(tr.resid.mean()); sd=float(tr.resid.std(ddof=1)); p=norm.cdf((te.pred_diff+mu)/sd); y=(te.actual_diff>0).astype(int); r=te.resid; rows.append({'season':sy,'train_neutral_n':len(tr),'n':len(te),'margin_mae':float(abs(r).mean()),'bias':float(r.mean()),'brier':float(np.mean((p-y)**2)),'logloss':ll(p,y),'ece':ece(p,y),'train_mu':mu,'train_sd':sd})
d=pd.DataFrame(rows); w=d.n.values; result={'weighted_margin_mae':float(np.average(d.margin_mae,weights=w)),'weighted_bias':float(np.average(d.bias,weights=w)),'weighted_brier':float(np.average(d.brier,weights=w)),'weighted_logloss':float(np.average(d.logloss,weights=w)),'weighted_ece':float(np.average(d.ece,weights=w)),'rows':rows}; d.to_csv(OUT/'neutral_calibration.csv',index=False); (OUT/'summary.json').write_text(json.dumps(result,indent=2)); print(json.dumps(result,indent=2))
