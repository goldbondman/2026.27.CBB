from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

OUT=Path('results/prediction_first_venue_downstream'); OUT.mkdir(parents=True,exist_ok=True)
G15=Path('/tmp/g15'); G10=Path('/tmp/g10'); VEN=Path('/tmp/venue')
NCAA='https://raw.githubusercontent.com/sportsdataverse/ncaa-mbb-hoops-data/main/mbb/ncaa_mbb_schedule_master.parquet'
EVAL=[2023,2024,2025]

def nn(x):
    import re
    return re.sub(r'[^a-z0-9]','',str(x).lower()) if pd.notna(x) else ''

g=pd.read_parquet(G15/'game_predictions.parquet'); p=pd.read_parquet(G10/'pace_predictions.parquet'); m=pd.read_csv(VEN/'identity_match_ledger.csv',dtype={'game_id':str})
for d in [g,p]: d['game_id']=d.game_id.astype(str)
s=pd.read_parquet(NCAA); gid='game_id' if 'game_id' in s.columns else 'contest_id'; s=s.rename(columns={gid:'game_id'}); s['game_id']=s.game_id.astype(str); s['hk']=s.home.map(nn); s['ak']=s.away.map(nn)
g=g.merge(s[['game_id','hk','ak']].drop_duplicates('game_id'),on='game_id',how='left').merge(m[['game_id','neutral']],on='game_id',how='inner'); g=g[g.neutral.notna()].copy(); g['neutral']=g.neutral.astype(bool); g['t0']=g.team0.map(nn); g['t1']=g.team1.map(nn); g['h0']=g.t0.eq(g.hk); g['h1']=g.t1.eq(g.hk); g=g[g.h0^g.h1].copy();
g['ph']=np.where(g.h0,g.pred0,g.pred1); g['pa']=np.where(g.h0,g.pred1,g.pred0); g['ah']=np.where(g.h0,g.actual0,g.actual1); g['aa']=np.where(g.h0,g.actual1,g.actual0)
g['hca']=np.where(g.neutral,0.0,2.7); g['ph_hca']=g.ph+g.hca/2; g['pa_hca']=g.pa-g.hca/2
rows=[]
for sy in EVAL:
    z=g[g.season==sy]; base_margin=abs((z.ah-z.aa)-(z.ph-z.pa)).mean(); hca_margin=abs((z.ah-z.aa)-(z.ph_hca-z.pa_hca)).mean(); base_score=np.mean(np.r_[abs(z.ah-z.ph),abs(z.aa-z.pa)]); hca_score=np.mean(np.r_[abs(z.ah-z.ph_hca),abs(z.aa-z.pa_hca)]); base_total=abs((z.ah+z.aa)-(z.ph+z.pa)).mean(); hca_total=abs((z.ah+z.aa)-(z.ph_hca+z.pa_hca)).mean(); rows.append({'season':sy,'n':len(z),'margin_base':base_margin,'margin_hca':hca_margin,'margin_gain':base_margin-hca_margin,'team_score_base':base_score,'team_score_hca':hca_score,'team_score_gain':base_score-hca_score,'total_base':base_total,'total_hca':hca_total,'total_delta':hca_total-base_total})
score=pd.DataFrame(rows)
# Test whether neutral/nonneutral venue class adds incremental pace information. Learn only prior seasons.
pg=p[['game_id','season','pace_pred','pace_actual']].merge(m[['game_id','neutral']],on='game_id',how='inner'); pg=pg[pg.neutral.notna()].copy(); pg['neutral']=pg.neutral.astype(bool); pg['resid']=pg.pace_actual-pg.pace_pred; prows=[]
for sy in EVAL:
    tr=pg[(pg.season>=2022)&(pg.season<sy)].copy(); te=pg[pg.season==sy].copy(); lm=LinearRegression().fit(tr[['neutral']],tr.resid); adj=te.pace_pred+lm.predict(te[['neutral']]); b=abs(te.pace_actual-te.pace_pred).mean(); n=abs(te.pace_actual-adj).mean(); prows.append({'season':sy,'n':len(te),'base_pace_mae':b,'venue_pace_mae':n,'gain':b-n,'neutral_coef':float(lm.coef_[0]),'intercept':float(lm.intercept_)})
pace=pd.DataFrame(prows)
result={'score_mean_margin_gain':float(score.margin_gain.mean()),'score_margin_wins':int((score.margin_gain>0).sum()),'score_mean_team_score_gain':float(score.team_score_gain.mean()),'score_team_score_wins':int((score.team_score_gain>0).sum()),'score_total_max_abs_delta':float(abs(score.total_delta).max()),'pace_mean_gain':float(pace.gain.mean()),'pace_wins':int((pace.gain>0).sum()),'pace_pass':bool(pace.gain.mean()>=.03 and (pace.gain>0).sum()>=2),'score_rows':score.to_dict(orient='records'),'pace_rows':pace.to_dict(orient='records')}
score.to_csv(OUT/'venue_score_metrics.csv',index=False); pace.to_csv(OUT/'venue_pace_metrics.csv',index=False); (OUT/'summary.json').write_text(json.dumps(result,indent=2)); print(json.dumps(result,indent=2))
