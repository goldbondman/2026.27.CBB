from __future__ import annotations

import json
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

OUT=Path('results/prediction_first_pace_downstream'); OUT.mkdir(parents=True,exist_ok=True)
G15=Path('/tmp/g15'); G10=Path('/tmp/g10'); EVAL=[2023,2024,2025]; TRAIN_MIN=2022


def features(games):
    x=games[['game_id','season','date','team0','team1']].drop_duplicates('game_id').copy(); x['date']=pd.to_datetime(x.date); x=x.sort_values(['date','game_id']); hist=defaultdict(deque); rows=[]
    for r in x.itertuples(index=False):
        dt=pd.Timestamp(r.date); rec={'game_id':str(r.game_id)}
        for side,t in [('a',str(r.team0)),('b',str(r.team1))]:
            q=hist[t]; prior=[d for d in q if d<dt]; raw=(dt-prior[-1]).days if prior else np.nan; rec[f'{side}_rest']=7.0 if pd.isna(raw) else float(np.clip(raw,0,7)); rec[f'{side}_g7']=sum(1 for d in prior if 0<(dt-d).days<=7); rec[f'{side}_g14']=sum(1 for d in prior if 0<(dt-d).days<=14)
        hist[str(r.team0)].append(dt); hist[str(r.team1)].append(dt); rows.append(rec)
    f=pd.DataFrame(rows); f['rest_diff']=f.a_rest-f.b_rest; f['g7_diff']=f.a_g7-f.b_g7; f['g14_diff']=f.a_g14-f.b_g14; return f


def main():
    games=pd.read_parquet(G15/'game_predictions.parquet'); r2=pd.read_parquet(G15/'r2_team_predictions.parquet'); pace=pd.read_parquet(G10/'pace_predictions.parquet')
    for d in [games,r2,pace]: d['game_id']=d.game_id.astype(str)
    f=features(games); p=pace[['game_id','season','pace_pred','pace_actual']].drop_duplicates('game_id').merge(f,on='game_id',how='inner'); p['pace_resid']=p.pace_actual-p.pace_pred; feats=['rest_diff','g7_diff','g14_diff']; out=[]; ledgers=[]
    team=r2[['game_id','season','team','pred_ortg','pts']].copy()
    for sy in EVAL:
        tr=p[(p.season>=TRAIN_MIN)&(p.season<sy)].copy(); te=p[p.season==sy].copy(); lm=LinearRegression().fit(tr[feats],tr.pace_resid); te['pace_adj']=te.pace_pred+lm.predict(te[feats]); ledgers.append(te.assign(eval_season=sy))
        z=team[team.season==sy].merge(te[['game_id','pace_pred','pace_adj']],on='game_id',how='inner'); z['score_base']=z.pred_ortg*z.pace_pred/100; z['score_adj']=z.pred_ortg*z.pace_adj/100; z['ae_base']=abs(z.pts-z.score_base); z['ae_adj']=abs(z.pts-z.score_adj)
        gr=z.groupby('game_id').agg(actual_total=('pts','sum'),base_total=('score_base','sum'),adj_total=('score_adj','sum')).reset_index(); gr['ae_base']=abs(gr.actual_total-gr.base_total); gr['ae_adj']=abs(gr.actual_total-gr.adj_total); gr['bias_base']=gr.actual_total-gr.base_total; gr['bias_adj']=gr.actual_total-gr.adj_total
        out.append({'season':sy,'games':len(gr),'pace_mae_base':float(abs(te.pace_actual-te.pace_pred).mean()),'pace_mae_adj':float(abs(te.pace_actual-te.pace_adj).mean()),'pace_gain':float(abs(te.pace_actual-te.pace_pred).mean()-abs(te.pace_actual-te.pace_adj).mean()),'team_score_mae_base':float(z.ae_base.mean()),'team_score_mae_adj':float(z.ae_adj.mean()),'team_score_gain':float(z.ae_base.mean()-z.ae_adj.mean()),'total_mae_base':float(gr.ae_base.mean()),'total_mae_adj':float(gr.ae_adj.mean()),'total_gain':float(gr.ae_base.mean()-gr.ae_adj.mean()),'total_bias_base':float(gr.bias_base.mean()),'total_bias_adj':float(gr.bias_adj.mean())})
    d=pd.DataFrame(out); result={'mean_pace_gain':float(d.pace_gain.mean()),'pace_wins':int((d.pace_gain>0).sum()),'mean_team_score_gain':float(d.team_score_gain.mean()),'team_score_wins':int((d.team_score_gain>0).sum()),'mean_total_gain':float(d.total_gain.mean()),'total_wins':int((d.total_gain>=0).sum()),'pass_full':bool(d.pace_gain.mean()>=.03 and (d.pace_gain>0).sum()>=2 and d.total_gain.mean()>=-.02),'season_rows':out}
    d.to_csv(OUT/'pace_downstream_metrics.csv',index=False); pd.concat(ledgers,ignore_index=True).to_csv(OUT/'pace_adjusted_ledger.csv',index=False); (OUT/'summary.json').write_text(json.dumps(result,indent=2)); print(json.dumps(result,indent=2))
if __name__=='__main__': main()
