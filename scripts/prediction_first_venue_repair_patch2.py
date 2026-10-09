from __future__ import annotations

import numpy as np
import pandas as pd

import prediction_first_venue_repair as m


def load_base():
    games = pd.read_parquet(m.G15 / 'game_predictions.parquet')
    pace = pd.read_parquet(m.G10 / 'pace_predictions.parquet')
    games['game_id'] = games.game_id.astype(str)
    pace['game_id'] = pace.game_id.astype(str)
    games['date'] = pd.to_datetime(games['date'], errors='coerce').dt.normalize()
    s = pd.read_parquet(m.NCAA_RAW + 'mbb/ncaa_mbb_schedule_master.parquet')
    gid = m.pick(s, ['game_id', 'contest_id'])
    s = s.rename(columns={gid: 'game_id'})
    s['game_id'] = s.game_id.astype(str)
    s['home_key'] = s.home.map(m.norm_name)
    s['away_key'] = s.away.map(m.norm_name)
    b = games.merge(
        s[['game_id','home','away','home_key','away_key']].drop_duplicates('game_id'),
        on='game_id', how='left', validate='one_to_one'
    )
    b['team0_key'] = b.team0.map(m.norm_name); b['team1_key'] = b.team1.map(m.norm_name)
    b['home_is_0'] = b.team0_key.eq(b.home_key); b['home_is_1'] = b.team1_key.eq(b.home_key)
    b['orientation_ok'] = b.home_is_0 ^ b.home_is_1
    b = b[b.orientation_ok].copy()
    b['pred_home'] = np.where(b.home_is_0,b.pred0,b.pred1); b['pred_away'] = np.where(b.home_is_0,b.pred1,b.pred0)
    b['actual_home'] = np.where(b.home_is_0,b.actual0,b.actual1); b['actual_away'] = np.where(b.home_is_0,b.actual1,b.actual0)
    b['pred_home_margin'] = b.pred_home-b.pred_away; b['actual_home_margin'] = b.actual_home-b.actual_away
    b['home_margin_resid'] = b.actual_home_margin-b.pred_home_margin
    b['pred_total_ha'] = b.pred_home+b.pred_away; b['actual_total_ha'] = b.actual_home+b.actual_away
    return b, pace


def load_espn():
    xs=[]
    for sy in range(2022,2026):
        d=pd.read_parquet(m.ESPN_REL+f'mbb_schedule_{sy}.parquet')
        d['season_key']=sy; xs.append(d)
    e=pd.concat(xs,ignore_index=True)
    e['espn_date']=pd.to_datetime(e['game_date'],errors='coerce').dt.normalize()
    # ESPN location is the school/location label (e.g., Duke) rather than the
    # mascot-bearing display_name (e.g., Duke Blue Devils), making this the
    # correct outcome-blind identity key for the NCAA schedule join.
    hcol='home_location' if 'home_location' in e.columns else ('home_name' if 'home_name' in e.columns else 'home_display_name')
    acol='away_location' if 'away_location' in e.columns else ('away_name' if 'away_name' in e.columns else 'away_display_name')
    e['ehome']=e[hcol].map(m.norm_name); e['eaway']=e[acol].map(m.norm_name)
    keep=['game_id','season_key','espn_date','ehome','eaway','neutral_site','venue_id','venue_full_name','venue_address_city','venue_address_state','home_venue_id']
    for c in keep:
        if c not in e: e[c]=np.nan
    return e[keep].copy()

m.load_base=load_base
m.load_espn=load_espn
m.main()
