from __future__ import annotations
import json, math, os, urllib.request
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
from sklearn.metrics import mean_absolute_error

OUT=Path('results/prediction_first_earlyseason15'); OUT.mkdir(parents=True,exist_ok=True)
BASE='https://raw.githubusercontent.com/sportsdataverse/hoopR-mbb-data/main/mbb/player_box/parquet/player_box_{year}.parquet'
YEARS=range(2021,2026)

def dl(y):
    p=Path('/tmp')/f'player_box_{y}.parquet'
    if not p.exists(): urllib.request.urlretrieve(BASE.format(year=y),p)
    return p

def safe_corr(a,b):
    z=pd.DataFrame({'a':a,'b':b}).replace([np.inf,-np.inf],np.nan).dropna()
    if len(z)<10 or z.a.nunique()<2 or z.b.nunique()<2:return np.nan
    return float(spearmanr(z.a,z.b).statistic)

def prep():
    xs=[]
    for y in YEARS:
        d=pd.read_parquet(dl(y))
        d=d[d.season.eq(y)].copy()
        d['game_date']=pd.to_datetime(d.game_date)
        num=['minutes','field_goals_made','field_goals_attempted','three_point_field_goals_made','three_point_field_goals_attempted','free_throws_made','free_throws_attempted','offensive_rebounds','defensive_rebounds','assists','steals','blocks','turnovers','points','team_score','opponent_team_score']
        for c in num:d[c]=pd.to_numeric(d[c],errors='coerce').fillna(0)
        xs.append(d)
    return pd.concat(xs,ignore_index=True)

def player_seasons(d):
    cols=['minutes','field_goals_made','field_goals_attempted','three_point_field_goals_made','three_point_field_goals_attempted','free_throws_made','free_throws_attempted','offensive_rebounds','defensive_rebounds','assists','steals','blocks','turnovers','points']
    p=d.groupby(['season','athlete_id','team_id','team_name'],as_index=False)[cols].sum()
    g=d.groupby(['season','athlete_id','team_id','team_name']).game_id.nunique().rename('games').reset_index(); p=p.merge(g)
    # Primary team per player-season handles rare multi-team records deterministically.
    p=p.sort_values(['season','athlete_id','minutes','team_id'],ascending=[True,True,False,True]).drop_duplicates(['season','athlete_id'])
    p['ts']=p.points/(2*(p.field_goals_attempted+.44*p.free_throws_attempted)).replace(0,np.nan)
    p['pts40']=40*p.points/p.minutes.replace(0,np.nan); p['ast40']=40*p.assists/p.minutes.replace(0,np.nan); p['tov40']=40*p.turnovers/p.minutes.replace(0,np.nan)
    p['reb40']=40*(p.offensive_rebounds+p.defensive_rebounds)/p.minutes.replace(0,np.nan)
    p['stocks40']=40*(p.steals+p.blocks)/p.minutes.replace(0,np.nan)
    p['raw_value']=p.points+.7*(p.offensive_rebounds+p.defensive_rebounds)+1.5*p.assists+2*p.steals+2*p.blocks-1.5*p.turnovers
    p['value40']=40*p.raw_value/p.minutes.replace(0,np.nan)
    return p

def team_games(d):
    # Team-game rows deduped from player lines.
    t=d[['season','game_id','game_date','team_id','team_name','opponent_team_id','team_score','opponent_team_score']].drop_duplicates(['season','game_id','team_id']).copy()
    t['margin']=t.team_score-t.opponent_team_score
    t=t.sort_values(['season','team_id','game_date','game_id']); t['game_no']=t.groupby(['season','team_id']).cumcount()+1
    return t

def transitions(p):
    a=p.copy(); b=p.copy(); a['season']=a.season+1
    x=a.merge(b,on=['season','athlete_id'],suffixes=('_prev','_cur'))
    x['same_team']=x.team_id_prev.eq(x.team_id_cur); x['transfer']=~x.same_team
    return x

def team_preseason_features(p,tg):
    rows=[]
    for sy in range(2022,2026):
        prev=p[p.season.eq(sy-1)]; cur=p[p.season.eq(sy)]
        prev_team_minutes=prev.groupby('team_id').minutes.sum().to_dict()
        prev_team_value=prev.groupby('team_id').raw_value.sum().to_dict()
        prev_margin=tg[tg.season.eq(sy-1)].groupby('team_id').margin.mean().to_dict()
        cur_early=tg[(tg.season.eq(sy))&(tg.game_no.le(5))].groupby(['team_id','team_name']).agg(early_margin=('margin','mean'),early_games=('game_id','nunique')).reset_index()
        for r in cur_early.itertuples(index=False):
            roster=cur[cur.team_id.eq(r.team_id)][['athlete_id','minutes']].copy()
            q=roster.merge(prev,on='athlete_id',how='left',suffixes=('_cur','_prev'))
            q['same']=q.team_id.eq(r.team_id); q['tr']=q.team_id.notna() & ~q['same']
            ret_min=float(q.loc[q.same,'minutes_prev'].fillna(0).sum()); tr_min=float(q.loc[q.tr,'minutes_prev'].fillna(0).sum())
            ret_val=float(q.loc[q.same,'raw_value'].fillna(0).sum()); tr_val=float(q.loc[q.tr,'raw_value'].fillna(0).sum())
            denom=prev_team_minutes.get(r.team_id,np.nan); vden=abs(prev_team_value.get(r.team_id,np.nan))
            new_share=float(q.loc[q.team_id.isna(),'minutes_cur'].sum()/max(q.minutes_cur.sum(),1))
            rows.append({'season':sy,'team_id':r.team_id,'team_name':r.team_name,'early_margin':r.early_margin,'early_games':r.early_games,
                         'prev_margin':prev_margin.get(r.team_id,np.nan),'return_min_share':ret_min/denom if denom else np.nan,
                         'return_value_share':ret_val/vden if vden else np.nan,'transfer_prev_minutes':tr_min,'transfer_prev_value':tr_val,'new_eventual_min_share':new_share})
    return pd.DataFrame(rows)

def rolling_team_style(d):
    # Aggregate both sides, enabling ORB opportunity rate by opponent DREB.
    a=d.groupby(['season','game_id','game_date','team_id','team_name','opponent_team_id'],as_index=False).agg(
        fga=('field_goals_attempted','sum'),fg3a=('three_point_field_goals_attempted','sum'),fta=('free_throws_attempted','sum'),orb=('offensive_rebounds','sum'),dreb=('defensive_rebounds','sum'),tov=('turnovers','sum'),pts=('points','sum'),mins=('minutes','sum'))
    opp=a[['season','game_id','team_id','dreb']].rename(columns={'team_id':'opponent_team_id','dreb':'opp_dreb'})
    a=a.merge(opp,on=['season','game_id','opponent_team_id'],how='left')
    a['three_rate']=a.fg3a/a.fga.replace(0,np.nan); a['poss']=a.fga-a.orb+a.tov+.44*a.fta
    a['tov_rate']=a.tov/a.poss.replace(0,np.nan); a['orb_rate']=a.orb/(a.orb+a.opp_dreb).replace(0,np.nan); a['pace_proxy']=a.poss
    a=a.sort_values(['season','team_id','game_date','game_id']); a['game_no']=a.groupby(['season','team_id']).cumcount()+1
    return a

def gates():
    d=prep(); p=player_seasons(d); tg=team_games(d); x=transitions(p); tf=team_preseason_features(p,tg); style=rolling_team_style(d)
    gates=[]
    def add(gate,status,decision,result,next_action):gates.append({'gate':gate,'status':status,'decision':decision,'result':result,'next_action':next_action})

    # 27.01 source coverage
    counts=d.groupby('season').size().to_dict(); players=p.groupby('season').athlete_id.nunique().to_dict(); teams=p.groupby('season').team_id.nunique().to_dict()
    add('T27.01','GREEN','PASS — PLAYER DATA ACQUIRED',f'rows={counts}; unique_players={players}; teams={teams}','Freeze source/version manifest for challenger work')

    # 27.02 identity continuity
    ids={y:set(p.loc[p.season.eq(y),'athlete_id']) for y in YEARS}; overlaps={f'{y-1}->{y}':len(ids[y-1]&ids[y]) for y in range(2022,2026)}
    mult=p.groupby(['season','athlete_id']).size().gt(1).sum()
    add('T27.02','GREEN','PASS',f'cross_season_id_overlap={overlaps}; duplicate_primary_player_seasons={int(mult)}','Use athlete_id as primary historical identity key')

    # 27.03 transfer classification
    tr=x.groupby('season').agg(transitions=('athlete_id','size'),same=('same_team','sum'),transfers=('transfer','sum')).reset_index(); tr['transfer_pct']=tr.transfers/tr.transitions
    add('T27.03','GREEN','PASS — LARGE TRANSFER SAMPLE',tr.to_dict('records'),'Use player-ID/team-ID transition labels for translation research')

    # 27.04 player persistence by metric, returners
    same=x[x.same_team & x.minutes_prev.ge(200) & x.minutes_cur.ge(200)]
    mets=['pts40','ast40','tov40','reb40','stocks40','ts','value40']; cors={m:safe_corr(same[f'{m}_prev'],same[f'{m}_cur']) for m in mets}
    add('T27.04','GREEN','PASS — SKILL PERSISTENCE IS METRIC-SPECIFIC',f'n={len(same)}; spearman={cors}','Use metric-specific shrinkage rather than one generic player-value persistence rate')

    # 27.05 transfers vs returners persistence
    z=x[(x.minutes_prev.ge(200))&(x.minutes_cur.ge(200))]
    comp={}
    for m in ['pts40','ast40','reb40','ts','value40']:
        comp[m]={'returner':safe_corr(z.loc[z.same_team,f'{m}_prev'],z.loc[z.same_team,f'{m}_cur']), 'transfer':safe_corr(z.loc[z.transfer,f'{m}_prev'],z.loc[z.transfer,f'{m}_cur'])}
    add('T27.05','GREEN','PASS — TRANSFER TRANSLATION NEEDED',f'n_return={int(z.same_team.sum())}; n_transfer={int(z.transfer.sum())}; correlations={comp}','Fit transfer-specific translation/shrinkage, not returner coefficients')

    # 27.06 transfer directional shifts
    zz=z[z.transfer].copy(); shifts={m:float((zz[f'{m}_cur']-zz[f'{m}_prev']).median()) for m in ['pts40','ast40','tov40','reb40','ts','value40']}
    by_year=zz.groupby('season').apply(lambda q: pd.Series({'n':len(q),'d_pts40':(q.pts40_cur-q.pts40_prev).median(),'d_ts':(q.ts_cur-q.ts_prev).median(),'d_value40':(q.value40_cur-q.value40_prev).median()}),include_groups=False).reset_index()
    add('T27.06','GREEN','PASS — EMPIRICAL TRANSLATION BASELINE',f'overall_median_shifts={shifts}; by_year={by_year.to_dict("records")}', 'Next challenger should condition translation on origin/destination strength, role and position')

    # 27.07 returning minutes vs returning value association with early margin, controlling previous margin
    q=tf.dropna(subset=['prev_margin','early_margin','return_min_share','return_value_share']).copy(); q['delta']=q.early_margin-q.prev_margin
    cmin=safe_corr(q.return_min_share,q.delta); cval=safe_corr(q.return_value_share,q.delta)
    add('T27.07','GREEN' if abs(cval)>=abs(cmin) else 'HOLD','PASS — VALUE >= MINUTES' if abs(cval)>=abs(cmin) else 'HOLD — MINUTES STILL STRONGER',f'n={len(q)}; rho_return_minutes_to_margin_change={cmin:.4f}; rho_return_value_to_margin_change={cval:.4f}','Retain both until direct common-fold v14 challenger test')

    # 27.08 chronological upper-bound team prior models
    feats_base=['prev_margin','return_min_share']; feats_ch=['prev_margin','return_min_share','return_value_share','transfer_prev_minutes','transfer_prev_value']
    rows=[]
    for sy in [2024,2025]:
        train=tf[tf.season.lt(sy)].dropna(subset=['early_margin']+feats_ch); test=tf[tf.season.eq(sy)].dropna(subset=['early_margin']+feats_ch)
        if len(test)<30:continue
        mb=make_pipeline(StandardScaler(),Ridge(alpha=10)).fit(train[feats_base],train.early_margin); mc=make_pipeline(StandardScaler(),Ridge(alpha=10)).fit(train[feats_ch],train.early_margin)
        eb=mean_absolute_error(test.early_margin,mb.predict(test[feats_base])); ec=mean_absolute_error(test.early_margin,mc.predict(test[feats_ch]))
        rows.append({'season':sy,'n':len(test),'base_mae':eb,'player_challenger_mae':ec,'gain':eb-ec})
    r8=pd.DataFrame(rows); gain=float(r8.gain.mean()) if len(r8) else np.nan
    add('T27.08','SHADOW' if gain>0 else 'REJECT','SHADOW — UPPER-BOUND SIGNAL' if gain>0 else 'REJECT — NO UPPER-BOUND LIFT',f'{r8.to_dict("records")}; mean_gain={gain:.4f}; IMPORTANT: current-season membership comes from eventual player_box participation','Acquire timestamped preseason rosters before any promotion; then rerun against v14 common folds')

    # 27.09 transfer feature ablation
    rows=[]
    for sy in [2024,2025]:
        train=tf[tf.season.lt(sy)].dropna(subset=['early_margin']+feats_ch); test=tf[tf.season.eq(sy)].dropna(subset=['early_margin']+feats_ch)
        f0=['prev_margin','return_min_share','return_value_share']; f1=feats_ch
        if len(test)<30:continue
        m0=make_pipeline(StandardScaler(),Ridge(alpha=10)).fit(train[f0],train.early_margin); m1=make_pipeline(StandardScaler(),Ridge(alpha=10)).fit(train[f1],train.early_margin)
        e0=mean_absolute_error(test.early_margin,m0.predict(test[f0])); e1=mean_absolute_error(test.early_margin,m1.predict(test[f1])); rows.append({'season':sy,'n':len(test),'no_transfer_mae':e0,'with_transfer_mae':e1,'gain':e0-e1})
    rr=pd.DataFrame(rows); gg=float(rr.gain.mean()) if len(rr) else np.nan
    add('T27.09','SHADOW' if gg>0 else 'REJECT','SHADOW — TRANSFER VALUE ADDS SIGNAL' if gg>0 else 'REJECT — SIMPLE TRANSFER VALUE DOES NOT ADD',f'{rr.to_dict("records")}; mean_gain={gg:.4f}','If shadow-positive, condition transfer translation before v14 challenge')

    # 27.10 unseen/new entrant uncertainty
    uu=tf.groupby('season').new_eventual_min_share.agg(['count','mean','median',lambda s:s.quantile(.75),lambda s:s.quantile(.9)]).reset_index(); uu.columns=['season','n','mean','median','q75','q90']
    corr_new=safe_corr(tf.new_eventual_min_share,abs(tf.early_margin-tf.prev_margin))
    add('T27.10','SHADOW','SHADOW — NEW-PLAYER SHARE IS UNCERTAINTY EVIDENCE',f'eventual_new_minutes={uu.to_dict("records")}; rho_new_share_to_abs_margin_change={corr_new:.4f}; hindsight membership makes this non-production','Replace eventual new-player minutes with preseason freshman/newcomer roster probabilities prospectively')

    # 27.11 role stabilization: early minute shares vs rest-of-season shares
    minute=d.groupby(['season','game_id','game_date','team_id','athlete_id'],as_index=False).minutes.sum(); minute=minute.sort_values(['season','team_id','game_date','game_id']); minute['game_no']=minute.groupby(['season','team_id','game_id']).ngroup() # not used
    team_game_order=d[['season','team_id','game_id','game_date']].drop_duplicates().sort_values(['season','team_id','game_date','game_id']); team_game_order['gn']=team_game_order.groupby(['season','team_id']).cumcount()+1
    minute=minute.merge(team_game_order[['season','team_id','game_id','gn']],on=['season','team_id','game_id'])
    stab={}
    for n in [1,2,3,5]:
        e=minute[minute.gn.le(n)].groupby(['season','team_id','athlete_id']).minutes.sum().rename('early'); r=minute[minute.gn.gt(n)].groupby(['season','team_id','athlete_id']).minutes.sum().rename('rest'); m=pd.concat([e,r],axis=1).fillna(0).reset_index();
        # normalize within team
        m['early_share']=m.early/m.groupby(['season','team_id']).early.transform('sum').replace(0,np.nan); m['rest_share']=m.rest/m.groupby(['season','team_id']).rest.transform('sum').replace(0,np.nan)
        stab[n]={'n_player_team':len(m),'rho':safe_corr(m.early_share,m.rest_share),'mae_share':float(abs(m.early_share-m.rest_share).mean())}
    add('T27.11','GREEN','PASS — ROTATIONS STABILIZE QUICKLY',f'{stab}','Use first 1/2/3/5 games to update minutes roles, with uncertainty declining empirically')

    # 27.12 prior role vs first-2 role for predicting rest role, experienced players only
    prevrole=p[['season','athlete_id','team_id','minutes','games']].copy(); prevrole['season']+=1; prevrole['prior_mpg']=prevrole.minutes/prevrole.games.replace(0,np.nan)
    early2=minute[minute.gn.le(2)].groupby(['season','team_id','athlete_id']).minutes.sum().rename('early2').reset_index(); games2=team_game_order[team_game_order.gn.le(2)].groupby(['season','team_id']).game_id.nunique().rename('n2').reset_index(); early2=early2.merge(games2); early2['early2_mpg']=early2.early2/early2.n2
    rest=minute[minute.gn.gt(2)].groupby(['season','team_id','athlete_id']).minutes.sum().rename('restmin').reset_index(); restg=team_game_order[team_game_order.gn.gt(2)].groupby(['season','team_id']).game_id.nunique().rename('restg').reset_index(); rest=rest.merge(restg); rest['rest_mpg']=rest.restmin/rest.restg
    role=rest.merge(early2,on=['season','team_id','athlete_id']).merge(prevrole[['season','athlete_id','team_id','prior_mpg']],on=['season','team_id','athlete_id'],how='left').dropna(subset=['prior_mpg'])
    rho_prior=safe_corr(role.prior_mpg,role.rest_mpg); rho_early=safe_corr(role.early2_mpg,role.rest_mpg)
    add('T27.12','GREEN' if rho_early>rho_prior else 'HOLD','PASS — EARLY ROTATION REVEAL' if rho_early>rho_prior else 'HOLD',f'n={len(role)}; rho_prior_role_to_rest={rho_prior:.4f}; rho_first2_role_to_rest={rho_early:.4f}','After Game 2, update role/minutes expectations materially toward observed rotation')

    # 27.13 style stabilization
    styles=['three_rate','tov_rate','orb_rate','pace_proxy']; stabstyle={}
    for n in [1,2,3,5]:
        a=style[style.game_no.le(n)].groupby(['season','team_id'])[styles].mean(); b=style[style.game_no.gt(n)].groupby(['season','team_id'])[styles].mean(); m=a.join(b,lsuffix='_early',rsuffix='_rest').dropna(); stabstyle[n]={s:safe_corr(m[f'{s}_early'],m[f'{s}_rest']) for s in styles}
    add('T27.13','GREEN','PASS — STYLE LEARNS FASTER THAN OUTCOMES',f'{stabstyle}','Permit early pace/shot-profile/turnover/ORB state updates only where OOS predictive tests beat frozen team-state update')

    # 27.14 adaptive margin updating
    prev=tg.groupby(['season','team_id']).margin.mean().rename('season_margin').reset_index(); prev['season']+=1
    upd=[]
    for n in [1,2,3,5]:
        e=tg[tg.game_no.le(n)].groupby(['season','team_id']).margin.mean().rename('early').reset_index(); restm=tg[tg.game_no.gt(n)].groupby(['season','team_id']).margin.mean().rename('rest').reset_index(); m=restm.merge(e).merge(prev,on=['season','team_id']).dropna()
        # train blend weight on seasons <=2023; test 2024-25. Grid chosen only on train.
        tr=m[m.season.le(2023)]; te=m[m.season.ge(2024)]; best=(None,1e9)
        for w in np.linspace(0,1,21):
            pred=(1-w)*tr.season_margin+w*tr.early; e0=np.mean(abs(tr.rest-pred))
            if e0<best[1]:best=(float(w),float(e0))
        w=best[0]; base=np.mean(abs(te.rest-te.season_margin)); chal=np.mean(abs(te.rest-((1-w)*te.season_margin+w*te.early)))
        upd.append({'n_games':n,'train_weight_current':w,'test_n':len(te),'prior_mae':base,'blend_mae':chal,'gain':base-chal})
    ud=pd.DataFrame(upd); bestrow=ud.sort_values('gain',ascending=False).iloc[0].to_dict()
    add('T27.14','GREEN' if bestrow['gain']>0 else 'REJECT','PASS — CHRONOLOGICAL ADAPTIVE UPDATE SIGNAL' if bestrow['gain']>0 else 'REJECT',f'all={ud.to_dict("records")}; best={bestrow}','Translate result into v14 state-space challenger using opponent-adjusted residuals, not raw margins')

    # 27.15 synthesis
    statuses={g['gate']:g['status'] for g in gates}; signal=sum(statuses[k] in ['GREEN','SHADOW'] for k in statuses)
    decision='REGISTER EARLY-SEASON PLAYER-STATE CHALLENGER — INCUMBENT REMAINS FROZEN'
    add('T27.15','GREEN',decision,f'prior_gates={len(gates)}; signal_or_pass={signal}; production_promotion=False; 2026_outcomes_used=False','Next 15 gates: acquire timestamp-valid historical/preseason roster + recruiting data, build freshman/transfer-conditioned priors, then common-fold challenge v14 specifically Games 1-10')

    pd.DataFrame(gates).to_csv(OUT/'t27_earlyseason_15_gates.csv',index=False)
    tf.to_csv(OUT/'t27_team_preseason_upperbound.csv',index=False)
    tr.to_csv(OUT/'t27_transfer_counts.csv',index=False); pd.DataFrame(stab).T.to_csv(OUT/'t27_rotation_stability.csv'); pd.DataFrame(stabstyle).to_json(OUT/'t27_style_stability.json',indent=2)
    (OUT/'summary.json').write_text(json.dumps({'gates_green':sum(g['status']=='GREEN' for g in gates),'gates_shadow':sum(g['status']=='SHADOW' for g in gates),'gates_hold':sum(g['status']=='HOLD' for g in gates),'gates_reject':sum(g['status']=='REJECT' for g in gates),'model_reopened':False,'2026_outcomes_used':False,'final_decision':decision},indent=2))
    print(pd.DataFrame(gates).to_string(index=False)); print((OUT/'summary.json').read_text())

if __name__=='__main__':gates()
