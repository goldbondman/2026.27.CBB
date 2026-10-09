from __future__ import annotations
import json
from collections import defaultdict, deque
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression
OUT=Path('results/prediction_first_rest_pace_ablation'); OUT.mkdir(parents=True,exist_ok=True)
g=pd.read_parquet('/tmp/g15/game_predictions.parquet'); p=pd.read_parquet('/tmp/g10/pace_predictions.parquet'); r2=pd.read_parquet('/tmp/g15/r2_team_predictions.parquet')
for d in [g,p,r2]: d['game_id']=d.game_id.astype(str)
g['date']=pd.to_datetime(g.date); x=g[['game_id','season','date','team0','team1']].drop_duplicates('game_id').sort_values(['date','game_id']); hist=defaultdict(deque); rows=[]
for rr in x.itertuples(index=False):
 dt=pd.Timestamp(rr.date); rec={'game_id':rr.game_id}
 for side,t in [('a',str(rr.team0)),('b',str(rr.team1))]:
  prior=[d for d in hist[t] if d<dt]; raw=(dt-prior[-1]).days if prior else np.nan; rec[f'{side}_rest']=7.0 if pd.isna(raw) else float(np.clip(raw,0,7)); rec[f'{side}_g7']=sum(1 for d in prior if 0<(dt-d).days<=7); rec[f'{side}_g14']=sum(1 for d in prior if 0<(dt-d).days<=14)
 hist[str(rr.team0)].append(dt); hist[str(rr.team1)].append(dt); rows.append(rec)
f=pd.DataFrame(rows); f['rest_diff']=f.a_rest-f.b_rest; f['g7_diff']=f.a_g7-f.b_g7; f['g14_diff']=f.a_g14-f.b_g14; feats=['rest_diff','g7_diff','g14_diff']
pg=p[['game_id','season','pace_pred','pace_actual']].drop_duplicates('game_id').merge(f,on='game_id',how='inner'); pg['resid']=pg.pace_actual-pg.pace_pred
team=r2[['game_id','season','team','pred_ortg','pts']].copy(); out=[]
for sy in [2023,2024,2025]:
 tr=pg[(pg.season>=2022)&(pg.season<sy)].copy(); te=pg[pg.season==sy].copy(); gc=float(tr.resid.mean()); te['pace_global']=te.pace_pred+gc; lm=LinearRegression().fit(tr[feats],tr.resid); te['pace_rest']=te.pace_pred+lm.predict(te[feats]); raw=float(abs(te.pace_actual-te.pace_pred).mean()); gm=float(abs(te.pace_actual-te.pace_global).mean()); rm=float(abs(te.pace_actual-te.pace_rest).mean());
 z=team[team.season==sy].merge(te[['game_id','pace_global','pace_rest']],on='game_id',how='inner'); z['score_g']=z.pred_ortg*z.pace_global/100; z['score_r']=z.pred_ortg*z.pace_rest/100; sg=float(abs(z.pts-z.score_g).mean()); sr=float(abs(z.pts-z.score_r).mean()); q=z.groupby('game_id').agg(actual=('pts','sum'),global_total=('score_g','sum'),rest_total=('score_r','sum')).reset_index(); tg=float(abs(q.actual-q.global_total).mean()); trm=float(abs(q.actual-q.rest_total).mean()); out.append({'season':sy,'n':len(te),'raw_pace_mae':raw,'global_pace_mae':gm,'rest_pace_mae':rm,'global_gain_vs_raw':raw-gm,'rest_incremental_pace_gain':gm-rm,'global_team_score_mae':sg,'rest_team_score_mae':sr,'rest_incremental_team_score_gain':sg-sr,'global_total_mae':tg,'rest_total_mae':trm,'rest_incremental_total_gain':tg-trm,'global_corr':gc,'coef_rest':float(lm.coef_[0]),'coef_g7':float(lm.coef_[1]),'coef_g14':float(lm.coef_[2])})
d=pd.DataFrame(out); result={'global_pace_gain':float(d.global_gain_vs_raw.mean()),'rest_incremental_pace_gain':float(d.rest_incremental_pace_gain.mean()),'rest_pace_wins':int((d.rest_incremental_pace_gain>0).sum()),'rest_incremental_team_score_gain':float(d.rest_incremental_team_score_gain.mean()),'rest_team_score_wins':int((d.rest_incremental_team_score_gain>0).sum()),'rest_incremental_total_gain':float(d.rest_incremental_total_gain.mean()),'rest_total_wins':int((d.rest_incremental_total_gain>0).sum()),'rest_pass':bool(d.rest_incremental_pace_gain.mean()>=.03 and (d.rest_incremental_pace_gain>0).sum()>=2 and d.rest_incremental_total_gain.mean()>=-.02),'rows':out}; d.to_csv(OUT/'rest_pace_ablation.csv',index=False); (OUT/'summary.json').write_text(json.dumps(result,indent=2)); print(json.dumps(result,indent=2))
