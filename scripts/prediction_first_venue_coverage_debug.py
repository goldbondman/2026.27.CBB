from __future__ import annotations
import re
from collections import defaultdict
from difflib import SequenceMatcher
import pandas as pd

G15='/tmp/g15/game_predictions.parquet'
NCAA='https://raw.githubusercontent.com/sportsdataverse/ncaa-mbb-hoops-data/main/mbb/ncaa_mbb_schedule_master.parquet'
ESPN='https://github.com/sportsdataverse/sportsdataverse-data/releases/download/espn_mens_college_basketball_schedules/mbb_schedule_{}.parquet'

def nn(x): return re.sub(r'[^a-z0-9]','',str(x).lower()) if pd.notna(x) else ''

g=pd.read_parquet(G15); g['game_id']=g.game_id.astype(str); g['date']=pd.to_datetime(g.date).dt.normalize()
s=pd.read_parquet(NCAA); gid='game_id' if 'game_id' in s.columns else 'contest_id'; s=s.rename(columns={gid:'game_id'}); s['game_id']=s.game_id.astype(str); s['hk']=s.home.map(nn); s['ak']=s.away.map(nn)
b=g.merge(s[['game_id','home','away','hk','ak']].drop_duplicates('game_id'),on='game_id',how='left'); b['t0']=b.team0.map(nn); b['t1']=b.team1.map(nn); b=b[(b.t0==b.hk)|(b.t1==b.hk)].copy()

xs=[]
for sy in range(2021,2027):
    try:
        d=pd.read_parquet(ESPN.format(sy)); d['file_sy']=sy; xs.append(d)
    except Exception as e: print('LOADFAIL',sy,str(e)[:120])
e=pd.concat(xs,ignore_index=True); e['edate']=pd.to_datetime(e.game_date).dt.normalize(); hc='home_location'; ac='away_location'; e['hk']=e[hc].map(nn); e['ak']=e[ac].map(nn)
print('ESPN FILES')
for sy,z in e.groupby('file_sy'):
    print(sy,len(z),z.edate.min(),z.edate.max(),sorted(z.season.dropna().unique())[:10])

keys=set(zip(e.edate,e.hk,e.ak))
rows=[]
for sy,z in b.groupby('season'):
    exact=sum((r.date,r.hk,r.ak) in keys for r in z.itertuples(index=False)); pm1=sum(((r.date-pd.Timedelta(days=1),r.hk,r.ak) in keys or (r.date+pd.Timedelta(days=1),r.hk,r.ak) in keys) for r in z.itertuples(index=False) if (r.date,r.hk,r.ak) not in keys)
    rows.append((sy,len(z),exact,pm1,z.date.min(),z.date.max()))
print('BASE COVERAGE EXACT')
for r in rows: print(r)

bydate=defaultdict(list)
for r in e.itertuples(index=False): bydate[r.edate].append((r.hk,r.ak,r.file_sy))
for sy,z in b.groupby('season'):
    un=[r for r in z.itertuples(index=False) if (r.date,r.hk,r.ak) not in keys]
    print('UNMATCHED SAMPLE',sy)
    for r in un[:8]:
        cand=bydate.get(r.date,[]); scored=[]
        for h,a,f in cand:
            sc=(SequenceMatcher(None,r.hk,h).ratio()+SequenceMatcher(None,r.ak,a).ratio())/2; scored.append((sc,h,a,f))
        scored.sort(reverse=True)
        print(r.date,r.home,r.away,'BEST',scored[:2])
