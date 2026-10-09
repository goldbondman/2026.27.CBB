from __future__ import annotations
import json
from pathlib import Path
import pandas as pd
URL='https://raw.githubusercontent.com/JDSB123/ncaam-historical-data/main/odds/normalized/odds_consolidated_canonical.csv'
OUT=Path('results/gate_batch_35_price_coverage'); OUT.mkdir(parents=True,exist_ok=True)
def main():
 use=['event_id','season','game_date','spread','spread_home_price','spread_away_price','moneyline_home_price','moneyline_away_price','total','total_over_price','total_under_price','bookmaker','timestamp','commence_time']
 d=pd.read_csv(URL,usecols=lambda c:c in use,low_memory=False)
 d['season']=pd.to_numeric(d.season,errors='coerce')
 d=d[d.season.between(2021,2025)].copy()
 rows=[]
 for sy,g in d.groupby('season'):
  def cnt(cols):
   q=g.dropna(subset=cols)
   return {'rows':int(len(q)),'events':int(q.event_id.nunique()),'books':int(q.bookmaker.nunique())}
  rows.append({'season':int(sy),'all_rows':int(len(g)),'events':int(g.event_id.nunique()),'spread':cnt(['spread','spread_home_price','spread_away_price']),'ml':cnt(['moneyline_home_price','moneyline_away_price']),'total':cnt(['total','total_over_price','total_under_price'])})
 out={'by_season':rows}
 (OUT/'summary.json').write_text(json.dumps(out,indent=2)); print('PRICE_COVERAGE'); print(json.dumps(out,indent=2))
if __name__=='__main__':main()
