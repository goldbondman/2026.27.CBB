from __future__ import annotations
import json
from pathlib import Path
import numpy as np
import pandas as pd
import gate_batch_30 as slow

OUT=Path('results/gate_batch_30_fast'); OUT.mkdir(parents=True,exist_ok=True)

def event_consensus(raw,line_col):
    z=raw.groupby(['season','event_id','game_date','home_team_canonical','away_team_canonical','is_march_madness'],as_index=False).agg(
        line=(line_col,'median'),line_mean=(line_col,'mean'),line_sd=(line_col,'std'),books=('bookmaker','nunique'),lead_min=('lead_min','median'))
    z['line_sd']=z.line_sd.fillna(0)
    z['home_key']=z.home_team_canonical.map(slow.normname); z['away_key']=z.away_team_canonical.map(slow.normname)
    return z

def model_orientations(games):
    a=games[['season','date','game_id','team0','team1','pred_margin','actual_margin','pred_total','actual_total','n_avg']].copy()
    a['home_key']=a.team0.map(slow.normname); a['away_key']=a.team1.map(slow.normname)
    a['pred_home_margin']=a.pred_margin; a['actual_home_margin']=a.actual_margin
    b=a.copy(); b['home_key']=a.team1.map(slow.normname); b['away_key']=a.team0.map(slow.normname); b['pred_home_margin']=-a.pred_margin; b['actual_home_margin']=-a.actual_margin
    return pd.concat([a,b],ignore_index=True)[['season','date','game_id','home_key','away_key','pred_home_margin','actual_home_margin','pred_total','actual_total','n_avg']]

def join_market(events,games):
    ori=model_orientations(games)
    m=events.merge(ori,left_on=['season','game_date','home_key','away_key'],right_on=['season','date','home_key','away_key'],how='inner')
    return m.drop_duplicates('event_id')

def main():
    games=slow.load_model_games()
    sp=slow.load_market(slow.SPREAD_URL,'spread'); to=slow.load_market(slow.TOTAL_URL,'total')
    se=event_consensus(sp,'spread'); te=event_consensus(to,'total')
    sc=join_market(se,games); tc=join_market(te,games)
    sz,sev,st=slow.spread_tests(sc); tz,tev,tt=slow.total_tests(tc)
    source={
      'spread_rows':int(len(sp)),'total_rows':int(len(to)),'spread_events':int(se.event_id.nunique()),'total_events':int(te.event_id.nunique()),
      'spread_books':int(sp.bookmaker.nunique()),'total_books':int(to.bookmaker.nunique()),
      'spread_match_events':int(sc.event_id.nunique()),'total_match_events':int(tc.event_id.nunique()),
      'spread_match_rate':float(sc.event_id.nunique()/se.event_id.nunique()),'total_match_rate':float(tc.event_id.nunique()/te.event_id.nunique()),
      'spread_season_events':{str(int(k)):int(v) for k,v in sc.groupby('season').size().items()},
      'total_season_events':{str(int(k)):int(v) for k,v in tc.groupby('season').size().items()},
      'lead_min':{'spread_median':float(sp.lead_min.median()),'spread_p95':float(sp.lead_min.quantile(.95)),'total_median':float(to.lead_min.median()),'total_p95':float(to.lead_min.quantile(.95)),'spread_nonpregame':int((sp.lead_min<0).sum()),'total_nonpregame':int((to.lead_min<0).sum())},
      'books_per_event':{'spread_median':float(sc.books.median()),'total_median':float(tc.books.median())},
      'line_dispersion':{'spread_median_sd':float(sc.line_sd.median()),'total_median_sd':float(tc.line_sd.median())},
    }
    loo={'spread':{},'total':{}}
    for sy in slow.YEARS:
        loo['spread'][str(sy)]=slow.perf_binary(sz.loc[(sz.season==sy)&(sz.edge_abs>=6),'ats_margin'])
        loo['total'][str(sy)]=slow.perf_binary(tz.loc[(tz.season==sy)&(tz.edge_abs>=8),'ou_margin'])
    summary={'source':source,'spread':st,'total':tt,'leave_one_season':loo}
    sz.to_csv(OUT/'spread_games.csv',index=False); sev.to_csv(OUT/'spread_ev_proxy.csv',index=False); tz.to_csv(OUT/'total_games.csv',index=False); tev.to_csv(OUT/'total_ev_proxy.csv',index=False)
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2,default=str))
    print('GATE_BATCH_30_FAST'); print(json.dumps(summary,indent=2,default=str))
if __name__=='__main__': main()
