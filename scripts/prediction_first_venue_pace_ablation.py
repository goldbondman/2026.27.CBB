from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
OUT=Path('results/prediction_first_venue_pace_ablation'); OUT.mkdir(parents=True,exist_ok=True)
p=pd.read_parquet('/tmp/g10/pace_predictions.parquet'); m=pd.read_csv('/tmp/venue/identity_match_ledger.csv',dtype={'game_id':str}); p['game_id']=p.game_id.astype(str); z=p.merge(m[['game_id','neutral']],on='game_id',how='inner'); z=z[z.neutral.notna()].copy(); z['neutral']=z.neutral.astype(bool); z['resid']=z.pace_actual-z.pace_pred; rows=[]
for sy in [2023,2024,2025]:
    tr=z[(z.season>=2022)&(z.season<sy)].copy(); te=z[z.season==sy].copy(); global_corr=float(tr.resid.mean()); global_pred=te.pace_pred+global_corr; lm=LinearRegression().fit(tr[['neutral']],tr.resid); venue_pred=te.pace_pred+lm.predict(te[['neutral']]); raw=float(abs(te.pace_actual-te.pace_pred).mean()); gm=float(abs(te.pace_actual-global_pred).mean()); vm=float(abs(te.pace_actual-venue_pred).mean()); rows.append({'season':sy,'n':len(te),'raw_mae':raw,'global_intercept_mae':gm,'global_gain':raw-gm,'venue_mae':vm,'venue_incremental_gain':gm-vm,'global_corr':global_corr,'neutral_incremental_coef':float(lm.coef_[0])})
d=pd.DataFrame(rows); result={'global_mean_gain':float(d.global_gain.mean()),'global_wins':int((d.global_gain>0).sum()),'venue_incremental_mean_gain':float(d.venue_incremental_gain.mean()),'venue_incremental_wins':int((d.venue_incremental_gain>0).sum()),'venue_pace_pass':bool(d.venue_incremental_gain.mean()>=.03 and (d.venue_incremental_gain>0).sum()>=2),'rows':rows}; d.to_csv(OUT/'venue_pace_ablation.csv',index=False); (OUT/'summary.json').write_text(json.dumps(result,indent=2)); print(json.dumps(result,indent=2))
