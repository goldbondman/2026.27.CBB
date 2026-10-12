from __future__ import annotations
import pandas as pd
import prediction_first_alt_factors15 as o
import prediction_first_earlyseason15 as e


def build():
    d=e.prep()
    a=d.groupby(['season','game_id','game_date','team_id','opponent_team_id','home_away'],as_index=False).agg(
      fgm=('field_goals_made','sum'),fga=('field_goals_attempted','sum'),fg3m=('three_point_field_goals_made','sum'),fg3a=('three_point_field_goals_attempted','sum'),
      ftm=('free_throws_made','sum'),fta=('free_throws_attempted','sum'),orb=('offensive_rebounds','sum'),dreb=('defensive_rebounds','sum'),tov=('turnovers','sum'),pts=('points','sum'))
    opp=a[['season','game_id','team_id','orb','dreb','tov','fga','fta','pts']].rename(columns={'team_id':'opponent_team_id','orb':'opp_orb','dreb':'opp_dreb','tov':'opp_tov','fga':'opp_fga','fta':'opp_fta','pts':'opp_pts'})
    a=a.merge(opp,on=['season','game_id','opponent_team_id'],how='left')
    a['poss']=a.fga-a.orb+a.tov+.44*a.fta
    a['efg']=(a.fgm+.5*a.fg3m)/a.fga.replace(0,pd.NA)
    a['ts']=a.pts/(2*(a.fga+.44*a.fta)).replace(0,pd.NA)
    a['tovr']=a.tov/a.poss.replace(0,pd.NA)
    a['orbr']=a.orb/(a.orb+a.opp_dreb).replace(0,pd.NA)
    a['drbr']=a.dreb/(a.dreb+a.opp_orb).replace(0,pd.NA)
    a['ftr']=a.fta/a.fga.replace(0,pd.NA)
    a['three_rate']=a.fg3a/a.fga.replace(0,pd.NA)
    a['margin']=a.pts-a.opp_pts
    a=a.sort_values(['season','team_id','game_date','game_id']).copy(); a['game_no']=a.groupby(['season','team_id']).cumcount()+1
    mets=['efg','ts','tovr','orbr','drbr','ftr','three_rate','poss','margin']
    for c in mets:
      a['pre_'+c]=a.groupby(['season','team_id'])[c].transform(lambda s:s.expanding().mean().shift())
      for w in [5,7]: a[f'l{w}_{c}']=a.groupby(['season','team_id'])[c].transform(lambda s,w=w:s.rolling(w,min_periods=max(3,w-2)).mean().shift())
    pre=a[['season','game_id','team_id']+['pre_'+c for c in mets]+[f'l{w}_{c}' for w in [5,7] for c in ['margin']]].copy()
    pre=pre.rename(columns={'team_id':'opponent_team_id',**{c:'opp_'+c for c in pre.columns if c not in ['season','game_id','team_id']}})
    a=a.merge(pre,on=['season','game_id','opponent_team_id'],how='left')
    a['adj_game_margin']=a['margin']+a['opp_pre_margin']
    for w in [5,7]:
      a[f'adj_l{w}_margin']=a.groupby(['season','team_id'])['adj_game_margin'].transform(lambda s,w=w:s.rolling(w,min_periods=max(3,w-2)).mean().shift())
      a[f'rec_game_{w}']=a['margin']+a[f'opp_l{w}_margin'].fillna(a['opp_pre_margin'])
      a[f'rec_l{w}_margin']=a.groupby(['season','team_id'])[f'rec_game_{w}'].transform(lambda s,w=w:s.rolling(w,min_periods=max(3,w-2)).mean().shift())
    num=[c for c in a.columns if c.startswith('pre_') or c.startswith('l5_') or c.startswith('l7_') or c.startswith('adj_l') or c.startswith('rec_l')]+['game_no']
    h=a[a.home_away.eq('home')][['season','game_id','team_id','pts','margin']+num].rename(columns={'team_id':'home_team_id','pts':'factor_actual_home','margin':'factor_actual_margin',**{c:'h_'+c for c in num}})
    aw=a[a.home_away.eq('away')][['season','game_id','team_id','pts']+num].rename(columns={'team_id':'away_team_id','pts':'factor_actual_away',**{c:'a_'+c for c in num}})
    g=h.merge(aw,on=['season','game_id'])
    for c in num:
      if c!='game_no': g['d_'+c]=g['h_'+c]-g['a_'+c]
    p=pd.read_parquet(o.FREEZE/'frozen_candidate_predictions.parquet').copy(); p['game_id']=pd.to_numeric(p.game_id,errors='coerce').astype('Int64'); g['game_id']=pd.to_numeric(g.game_id,errors='coerce').astype('Int64')
    p['base_home']=p.mu_home+p.mean_eh_final; p['base_away']=p.mu_away+p.mean_ea_final; p['base_margin']=p.base_home-p.base_away
    if 'actual_margin' not in p: p['actual_margin']=p.actual_home-p.actual_away
    q=p.merge(g,on=['season','game_id'],how='inner'); q=q[q.season.isin([2023,2024,2025])].copy(); q['resid']=q.actual_margin-q.base_margin
    return a,q

if __name__=='__main__':
    o.build=build
    o.main()
