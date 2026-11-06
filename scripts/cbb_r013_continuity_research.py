import pyarrow.parquet as pq,pandas as pd,numpy as np,json,math,hashlib
from pathlib import Path
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from scipy.stats import bootstrap
p=Path('/tmp/cbb')
def read(n,cols): return pq.read_table(p/(n+'.parquet'),columns=cols).to_pandas()
playercols=['athlete_id','team_id','game_id','minutes','points','field_goals_attempted','assists','rebounds']
boxcols=['game_id','game_date','team_id','opponent_team_id','team_home_away','team_score','opponent_team_score','field_goals_attempted','offensive_rebounds','total_turnovers','free_throws_attempted']
raw={}
for y in [2024,2025,2026]:
 raw['p'+str(y)]=read('ep'+str(y),playercols)
 raw['b'+str(y)]=read('eb'+str(y),boxcols)
 if y>=2025:raw['r'+str(y)]=read('er'+str(y),['team_id','athlete_id'])
for y in [2024,2025,2026]:
 d=raw['p'+str(y)]
 d['athlete_id']=pd.to_numeric(d['athlete_id'],errors='coerce').astype('Int64').astype(str)
 d['team_id']=d['team_id'].astype(str)
 for c in ['minutes','points','field_goals_attempted','assists','rebounds']: d[c]=pd.to_numeric(d[c],errors='coerce').fillna(0)
 b=raw['b'+str(y)]
 for c in ['team_id','opponent_team_id']:b[c]=b[c].astype(str)
 b['game_date']=pd.to_datetime(b['game_date'],errors='coerce')
 b['poss']=b.field_goals_attempted-b.offensive_rebounds+b.total_turnovers+.475*b.free_throws_attempted
for y in [2025,2026]:
 d=raw['r'+str(y)];d['athlete_id']=d.athlete_id.astype(str);d['team_id']=d.team_id.astype(str)
summary={}
frames={}
for y in [2025,2026]:
 prev=raw['p'+str(y-1)]
 prior=raw['b'+str(y-1)]
 upcoming=raw['b'+str(y)]
 roster=raw['r'+str(y)]
 prev_pg=prev.groupby(['team_id','athlete_id'],as_index=False)[['minutes','points','field_goals_attempted','assists','rebounds']].sum()
 membership=roster[['team_id','athlete_id']].drop_duplicates()
 ret=prev_pg.merge(membership.assign(returning=1),on=['team_id','athlete_id'],how='left')
 ret['returning']=ret.returning....[truncated]