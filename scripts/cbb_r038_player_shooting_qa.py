import pandas as pd, numpy as np, json, hashlib
from pathlib import Path
p=Path('/tmp/cbb');out=p/'r036';out.mkdir(exist_ok=True)
f=['athlete_id','team_id','minutes','three_point_field_goals_made','three_point_field_goals_attempted','free_throws_made','free_throws_attempted','field_goals_attempted','points']
data={}
for y in [2024,2025,2026]:
 d=pd.read_parquet(p/f'ep{y}.parquet',columns=f)
 d=d[d.athlete_id.notna()].copy()
 d['athlete_id']=pd.to_numeric(d.athlete_id,errors='coerce').astype('Int64').astype(str)
 d['team_id']=pd.to_numeric(d.team_id,errors='coerce').astype('Int64').astype(str)
 for c in f[2:]:d[c]=pd.to_numeric(d[c],errors='coerce').fillna(0)
 a=d.groupby('athlete_id')[f[2:]].sum()
 # first team by largest minutes; transfer classification is retrospective and not pregame certified
 t=d.groupby(['athlete_id','team_id']).minutes.sum().reset_index().sort_values(['athlete_id','minutes'],ascending=[True,False]).drop_duplicates('athlete_id')
 a=a.merge(t[['athlete_id','team_id']],on='athlete_id',how='left')
 data[y]=a
res={}
for y in [2025,2026]:
 A=data[y-1];B=data[y]
 m=A.merge(B,on='athlete_id',suffixes=('_prev','_next'),validate='one_to_one')
 m=m[(m.three_point_field_goals_attempted_prev>=30)&(m.three_point_field_goals_attempted_next>=30)]
 pprev=m.three_point_field_goals_made_prev/m.three_point_field_goals_attempted_prev
 pnext=m.three_point_field_goals_made_next/m.three_point_field_goals_attempted_next
 lg=float(A.three_point_field_goals_made.sum()/max(1,A.three_point_field_goals_attempted.sum()))
 c={"players":len(m),"transfers":int((m.team_id_prev!=m.team_id_next).sum()),"same_team":int((m.team_id_prev==m.team_id_next).sum()),"year":f'{y-1}-{y}',"league_shooting_prior":lg,"correlation":float(pprev.corr(pnext))}
 for k in [0,25,50,100,200]:
  projected=(m.three_point_field_goals_made_prev+k*lg)/(m.three_point_field_goals_attempted_prev+k)
  c[f'mae_shrink_{k}']=float((projected-pnext).abs().mean())
  c[f'weighted_MAE_shrink_{k}']=float(np.average(np.abs(projected-pnext),weights=m.three_point_field_goals_attempted_next))
 res[str(y)]=c
print(json.dumps(res,indent=2))
(out/'player_shooting_reliability.json').write_text(json.dumps({"diagnostic":"Retrospective surviving NCAA player population, not all historical players; must not be used for pregame roster availability","data_source":"ESPN sportsdataverse ep Parquet","source_hash":{str(y):hashlib.sha256((p/f'ep{y}.parquet').read_bytes()).hexdigest() for y in [2024,2025,2026]},"results":res},indent=2))

