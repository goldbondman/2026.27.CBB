import urllib.request,pathlib,pyarrow.parquet as pq,numpy as np,pandas as pd,json,hashlib
from pathlib import Path
p=Path('/tmp/cbb')
sources={'ncaa_poss27':'https://raw.githubusercontent.com/sportsdataverse/ncaa-mbb-hoops-data/main/mbb/possessions/parquet/ncaa_mbb_possessions_2027.parquet','ncaa_team27':'https://raw.githubusercontent.com/sportsdataverse/ncaa-mbb-hoops-data/main/mbb/team_box/parquet/ncaa_mbb_team_box_2027.parquet','ncaa_player27':'https://raw.githubusercontent.com/sportsdataverse/ncaa-mbb-hoops-data/main/mbb/player_box/parquet/ncaa_mbb_player_box_2027.parquet','espn_team27':'https://raw.githubusercontent.com/sportsdataverse/hoopR-mbb-data/main/mbb/team_box/parquet/team_box_2027.parquet'}
for k,u in sources.items():
 file=p/(k+'.parquet')
 if not file.exists():file.write_bytes(urllib.request.urlopen(u,timeout=30).read())
pos=pq.read_table(p/'ncaa_poss27.parquet').to_pandas()
team=pq.read_table(p/'ncaa_team27.parquet').to_pandas()
player=pq.read_table(p/'ncaa_player27.parquet').to_pandas()
espn=pq.read_table(p/'espn_team27.parquet').to_pandas()
pg=pos.groupby(['contest_id','poss_team'],dropna=False).agg(events=('poss_num','size'),eventpts=('pts','sum')).reset_index().rename(columns={'poss_team':'team'})
m=team.merge(pg,on=['contest_id','team'],validate='one_to_one')
pt=player.groupby(['contest_id','team'],dropna=False).pts.sum().reset_index().rename(columns={'pts':'playerpts'})
m=m.merge(pt,on=['contest_id','team'],validate='one_to_one')
for k,c in [('p44',.44),('p475',.475)]:m[k]=m.fga-m.orb+m['to']+c*m.fta
m['error_44']=(m.p44-m.events).abs()
m['error_475']=(m.p475-m.events).abs()
m['paired_gain']=m.error_44-m.error_475
bygame=m.groupby('contest_id').paired_gain.mean()
rng=np.random.default_rng(4112026)
boot=np.array([rng.choice(bygame.values,size=len(bygame),replace=True).mean() for _ in range(4000)])
cs={
"n_games":int(m.contest_id.nunique()),"n_team_rows":len(m),"n_possession_rows":len(pos),"n_player_rows":len(player),
"event_key_duplicates":int(...[truncated]