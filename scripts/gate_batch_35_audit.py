from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
URL='https://raw.githubusercontent.com/JDSB123/ncaam-historical-data/main/odds/normalized/odds_consolidated_canonical.csv'
OUT=Path('results/gate_batch_35_audit'); OUT.mkdir(parents=True,exist_ok=True)

def main():
    use=['event_id','commence_time','game_date','season','bookmaker','timestamp','home_team_canonical','away_team_canonical','spread','spread_home_price','spread_away_price','moneyline_home_price','moneyline_away_price']
    d=pd.read_csv(URL,usecols=lambda c:c in use,low_memory=False)
    d['game_date']=pd.to_datetime(d.game_date,errors='coerce')
    d['season_raw']=pd.to_numeric(d.season,errors='coerce')
    d['derived_end_season']=d.game_date.dt.year + (d.game_date.dt.month>=7).astype('Int64')
    d['timestamp']=pd.to_datetime(d.timestamp,utc=True,errors='coerce')
    d['commence_time']=pd.to_datetime(d.commence_time,utc=True,errors='coerce')
    d['lead_min']=(d.commence_time-d.timestamp).dt.total_seconds()/60
    pre=d[d.lead_min>=0].copy()
    season_cross=pd.crosstab(pre.season_raw,pre.derived_end_season,dropna=False)
    season_cross.to_csv(OUT/'season_crosswalk.csv')
    bydate=pre.groupby([pre.game_date.dt.year.rename('calendar_year'),pre.game_date.dt.month.rename('month'), 'season_raw','derived_end_season']).agg(rows=('event_id','size'),events=('event_id','nunique')).reset_index()
    bydate.to_csv(OUT/'season_months.csv',index=False)
    snaps=pre.groupby(['event_id','bookmaker']).agg(n=('timestamp','size'),min_ts=('timestamp','min'),max_ts=('timestamp','max'),commence=('commence_time','first')).reset_index()
    multi=float((snaps.n>1).mean()) if len(snaps) else 0.0
    span=(snaps.max_ts-snaps.min_ts).dt.total_seconds()/60
    audit={
      'rows':int(len(pre)), 'events':int(pre.event_id.nunique()),
      'raw_seasons':{str(k):int(v) for k,v in pre.groupby('season_raw').size().items()},
      'derived_seasons':{str(k):int(v) for k,v in pre.groupby('derived_end_season').size().items()},
      'season_equal_share':float((pre.season_raw==pre.derived_end_season).mean()),
      'multi_snapshot_share_event_book':multi,
      'max_snapshots_event_book':int(snaps.n.max()) if len(snaps) else 0,
      'median_snapshot_span_min_multi':float(span[snaps.n>1].median()) if (snaps.n>1).any() else None,
      'median_lead_min_all':float(pre.lead_min.median()),
      'p90_lead_min_all':float(pre.lead_min.quantile(.9)),
    }
    (OUT/'summary.json').write_text(json.dumps(audit,indent=2,default=str))
    print('G35_AUDIT'); print(json.dumps(audit,indent=2,default=str))
if __name__=='__main__': main()
