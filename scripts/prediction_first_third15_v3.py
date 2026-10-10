from __future__ import annotations

import numpy as np
import pandas as pd
import prediction_first_third15 as m


def clean_id(x):
    if pd.isna(x):
        return np.nan
    s=str(x).strip()
    if s.endswith('.0'):
        s=s[:-2]
    return s


def load_all_repaired():
    gp,r2,pace,cf,venue,sched=m.load_all.__wrapped__() if hasattr(m.load_all,'__wrapped__') else (None,None,None,None,None,None)

# Keep the repair self-contained rather than depending on wrapper import side effects.
def load_all_v3():
    gp=pd.read_parquet(m.G15/'game_predictions.parquet'); gp['game_id']=gp.game_id.astype(str); gp['date']=pd.to_datetime(gp.date)
    r2=pd.read_parquet(m.G15/'r2_team_predictions.parquet'); r2['game_id']=r2.game_id.astype(str); r2['date']=pd.to_datetime(r2.date)
    pace=pd.read_parquet(m.G10/'pace_predictions.parquet'); pace['game_id']=pace.game_id.astype(str)
    if 'pred_pace' not in pace.columns:
        p=pace[['game_id','pace_pred']].drop_duplicates('game_id').rename(columns={'pace_pred':'pred_pace'})
        pace=r2[['game_id','team']].merge(p,on='game_id',how='left',validate='many_to_one')
    cf=pd.read_parquet(m.PF/'component_features.parquet'); cf['game_id']=cf.game_id.astype(str); cf['date']=pd.to_datetime(cf.date)
    venue=pd.read_csv(m.VEN/'identity_match_ledger.csv',dtype={'game_id':str})
    venue['neutral_known']=venue.neutral.notna()
    venue['neutral']=venue.neutral.map(lambda x: str(x).lower()=='true' if pd.notna(x) else np.nan)
    venue['espn_game_id']=venue['espn_game_id'].map(clean_id)
    sched=pd.read_parquet(m.RAW+'mbb/ncaa_mbb_schedule_master.parquet'); gid='game_id' if 'game_id' in sched.columns else 'contest_id'; sched=sched.rename(columns={gid:'game_id'}); sched['game_id']=sched.game_id.astype(str)
    sched['hk']=sched.home.map(m.nn); sched['ak']=sched.away.map(m.nn)
    return gp,r2,pace,cf,venue,sched

_orig_post=m.load_postseason_map
def post_v3(venue):
    z=_orig_post(venue)
    z['espn_game_id']=z['espn_game_id'].map(clean_id)
    return z

m.load_all=load_all_v3
m.load_postseason_map=post_v3
m.main()
