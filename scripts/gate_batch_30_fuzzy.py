from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
import gate_batch_30 as core
import gate_batch_30_fast as fast

OUT=Path('results/gate_batch_30_fuzzy'); OUT.mkdir(parents=True,exist_ok=True)

def fuzzy_join(events,games):
    rows=[]; ambiguous=0; date_offset_counts={-1:0,0:0,1:0}
    by={(int(s),d):q for (s,d),q in games.groupby(['season','date'])}
    for r in events.itertuples(index=False):
        matches=[]
        for off in [0,-1,1]:
            d=r.game_date+pd.Timedelta(days=off)
            q=by.get((int(r.season),d),None)
            if q is None: continue
            for x in q.itertuples(index=False):
                if core.name_match(r.home_team_canonical,x.team0) and core.name_match(r.away_team_canonical,x.team1): matches.append((x,True,off))
                elif core.name_match(r.home_team_canonical,x.team1) and core.name_match(r.away_team_canonical,x.team0): matches.append((x,False,off))
            if matches: break
        # unique game id only
        ids={str(x[0].game_id) for x in matches}
        if len(ids)!=1:
            if len(ids)>1: ambiguous+=1
            continue
        x,home0,off=matches[0]; date_offset_counts[off]+=1
        d=r._asdict(); d.update({'game_id':x.game_id,'pred_home_margin':float(x.pred_margin if home0 else -x.pred_margin),'actual_home_margin':float(x.actual_margin if home0 else -x.actual_margin),'pred_total':float(x.pred_total),'actual_total':float(x.actual_total),'n_avg':float(x.n_avg),'date_offset':off})
        rows.append(d)
    return pd.DataFrame(rows),ambiguous,date_offset_counts

def main():
    games=core.load_model_games(); sp=core.load_market(core.SPREAD_URL,'spread'); to=core.load_market(core.TOTAL_URL,'total')
    se=fast.event_consensus(sp,'spread'); te=fast.event_consensus(to,'total')
    sc,sa,so=fuzzy_join(se,games); tc,ta,toff=fuzzy_join(te,games)
    sz,sev,st=core.spread_tests(sc); tz,tev,tt=core.total_tests(tc)
    source={'spread_rows':len(sp),'total_rows':len(to),'spread_events':len(se),'total_events':len(te),'spread_match_events':len(sc),'total_match_events':len(tc),'spread_match_rate':len(sc)/len(se),'total_match_rate':len(tc)/len(te),'spread_ambiguous':sa,'total_ambiguous':ta,'spread_date_offsets':so,'total_date_offsets':toff,'spread_season_events':{str(int(k)):int(v) for k,v in sc.groupby('season').size().items()},'total_season_events':{str(int(k)):int(v) for k,v in tc.groupby('season').size().items()},'lead_min':{'spread_median':float(sp.lead_min.median()),'spread_p95':float(sp.lead_min.quantile(.95)),'total_median':float(to.lead_min.median()),'total_p95':float(to.lead_min.quantile(.95)),'spread_nonpregame':int((sp.lead_min<0).sum()),'total_nonpregame':int((to.lead_min<0).sum())},'books_per_event':{'spread_median':float(sc.books.median()),'total_median':float(tc.books.median())},'line_dispersion':{'spread_median_sd':float(sc.line_sd.median()),'total_median_sd':float(tc.line_sd.median())}}
    summary={'source':source,'spread':st,'total':tt}
    sz.to_csv(OUT/'spread_games.csv',index=False); sev.to_csv(OUT/'spread_ev_proxy.csv',index=False); tz.to_csv(OUT/'total_games.csv',index=False); tev.to_csv(OUT/'total_ev_proxy.csv',index=False)
    (OUT/'summary.json').write_text(json.dumps(summary,indent=2,default=str))
    print('GATE_BATCH_30_FUZZY'); print(json.dumps(summary,indent=2,default=str))
if __name__=='__main__': main()
