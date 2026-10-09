from __future__ import annotations

import json
import math
import re
from collections import defaultdict, deque
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import LinearRegression

OUT = Path('results/prediction_first_15_gates')
OUT.mkdir(parents=True, exist_ok=True)
G15 = Path('/tmp/g15')
G10 = Path('/tmp/g10')
RAW = 'https://raw.githubusercontent.com/sportsdataverse/ncaa-mbb-hoops-data/main/'
EVAL_SEASONS = [2023, 2024, 2025]
TRAIN_MIN_SEASON = 2022


def pick(df: pd.DataFrame, options: list[str], required: bool = True):
    for c in options:
        if c in df.columns:
            return c
    if required:
        raise KeyError(f'missing one of {options}; have {list(df.columns)}')
    return None


def norm_name(x):
    if pd.isna(x):
        return ''
    return re.sub(r'[^a-z0-9]', '', str(x).lower())


def ece(p, y, bins=10):
    d = pd.DataFrame({'p': p, 'y': y}).dropna()
    if d.empty:
        return np.nan
    d['b'] = np.minimum((d['p'] * bins).astype(int), bins - 1)
    return float(sum(len(z) / len(d) * abs(z.p.mean() - z.y.mean()) for _, z in d.groupby('b')))


def logloss(p, y):
    p = np.clip(np.asarray(p, float), 1e-9, 1 - 1e-9)
    y = np.asarray(y, float)
    return float(np.mean(-(y * np.log(p) + (1 - y) * np.log(1 - p))))


def gaussian_nll(resid, sigma, mean=0.0):
    r = np.asarray(resid, float)
    s = np.clip(np.asarray(sigma, float), 1e-6, None)
    return float(np.mean(0.5 * np.log(2 * np.pi * s * s) + 0.5 * ((r - mean) / s) ** 2))


def stage_bucket(v):
    x = float(v) if pd.notna(v) else 0.0
    if x <= 1:
        return '0-1'
    if x <= 4:
        return '2-4'
    if x <= 9:
        return '5-9'
    return '10+'


def load_inputs():
    games = pd.read_parquet(G15 / 'game_predictions.parquet')
    r2 = pd.read_parquet(G15 / 'r2_team_predictions.parquet')
    pace = pd.read_parquet(G10 / 'pace_predictions.parquet')
    schedule = pd.read_parquet(RAW + 'mbb/ncaa_mbb_schedule_master.parquet')
    for d in [games, r2, pace]:
        d['game_id'] = d['game_id'].astype(str)
    sgid = pick(schedule, ['contest_id', 'game_id', 'id'])
    schedule = schedule.rename(columns={sgid: 'game_id'}).copy()
    schedule['game_id'] = schedule['game_id'].astype(str)
    return games, r2, pace, schedule


def audit_schedule_schema(schedule):
    columns = list(schedule.columns)
    loc_tokens = ['home', 'away', 'neutral', 'venue', 'location', 'city', 'state', 'latitude', 'longitude', 'lat', 'lon', 'timezone', 'altitude', 'elevation']
    location_cols = [c for c in columns if any(t in c.lower() for t in loc_tokens)]
    schema = {'columns': columns, 'location_related_columns': location_cols, 'row_count': int(len(schedule))}
    (OUT / 'schedule_schema.json').write_text(json.dumps(schema, indent=2, default=str))
    return schema


def infer_schedule_columns(schedule):
    home = pick(schedule, ['home_team', 'home_team_name', 'home', 'home_name', 'home_team_display_name', 'home_team_short_display_name', 'home_team_location'], required=False)
    away = pick(schedule, ['away_team', 'away_team_name', 'away', 'away_name', 'away_team_display_name', 'away_team_short_display_name', 'away_team_location'], required=False)
    neutral = pick(schedule, ['neutral_site', 'neutral', 'is_neutral', 'neutralSite'], required=False)
    date = pick(schedule, ['date', 'game_date', 'start_date', 'game_datetime', 'start_time'], required=False)
    venue = pick(schedule, ['venue', 'venue_name', 'venue_full_name', 'arena', 'location'], required=False)
    return {'home': home, 'away': away, 'neutral': neutral, 'date': date, 'venue': venue}


def load_schedule_fallback(seasons=range(2021, 2026)):
    frames = []
    for sy in seasons:
        url = RAW + f'mbb/schedule/parquet/ncaa_mbb_schedule_{sy}.parquet'
        try:
            d = pd.read_parquet(url)
            d['season'] = sy
            frames.append(d)
        except Exception:
            continue
    if not frames:
        return pd.DataFrame()
    z = pd.concat(frames, ignore_index=True)
    gid = pick(z, ['contest_id', 'game_id', 'id'])
    z = z.rename(columns={gid: 'game_id'})
    z['game_id'] = z['game_id'].astype(str)
    return z


def build_game_context(games, schedule):
    cols = infer_schedule_columns(schedule)
    source = 'schedule_master'
    if cols['home'] is None or cols['away'] is None:
        fb = load_schedule_fallback()
        if not fb.empty:
            schedule = fb
            cols = infer_schedule_columns(schedule)
            source = 'season_schedule'
    if cols['home'] is None or cols['away'] is None:
        return None, {'source': source, 'error': 'No home/away team-name fields found', 'columns': list(schedule.columns)}
    keep = ['game_id', cols['home'], cols['away']]
    for c in [cols['neutral'], cols['date'], cols['venue']]:
        if c and c not in keep:
            keep.append(c)
    s = schedule[keep].drop_duplicates('game_id').copy()
    rename = {cols['home']: 'home_team', cols['away']: 'away_team'}
    if cols['neutral']:
        rename[cols['neutral']] = 'neutral_raw'
    if cols['date']:
        rename[cols['date']] = 'schedule_date'
    if cols['venue']:
        rename[cols['venue']] = 'venue'
    s = s.rename(columns=rename)
    if 'neutral_raw' not in s:
        s['neutral_raw'] = False
    if 'schedule_date' not in s:
        s['schedule_date'] = pd.NaT
    if 'venue' not in s:
        s['venue'] = np.nan
    def neutral_bool(v):
        if pd.isna(v):
            return False
        if isinstance(v, (bool, np.bool_)):
            return bool(v)
        if isinstance(v, (int, float, np.integer, np.floating)):
            return float(v) != 0
        return str(v).strip().lower() in {'true', 't', 'yes', 'y', '1', 'neutral', 'n'}
    s['neutral'] = s['neutral_raw'].map(neutral_bool)
    s['home_key'] = s['home_team'].map(norm_name)
    s['away_key'] = s['away_team'].map(norm_name)
    g = games.merge(s[['game_id','home_team','away_team','home_key','away_key','neutral','venue','schedule_date']], on='game_id', how='left')
    g['team0_key'] = g['team0'].map(norm_name)
    g['team1_key'] = g['team1'].map(norm_name)
    g['home_is_0'] = g['team0_key'].eq(g['home_key'])
    g['home_is_1'] = g['team1_key'].eq(g['home_key'])
    g['orientation_ok'] = g['home_is_0'] ^ g['home_is_1']
    ok = g[g.orientation_ok].copy()
    ok['pred_home'] = np.where(ok.home_is_0, ok.pred0, ok.pred1)
    ok['pred_away'] = np.where(ok.home_is_0, ok.pred1, ok.pred0)
    ok['actual_home'] = np.where(ok.home_is_0, ok.actual0, ok.actual1)
    ok['actual_away'] = np.where(ok.home_is_0, ok.actual1, ok.actual0)
    ok['pred_home_margin'] = ok.pred_home - ok.pred_away
    ok['actual_home_margin'] = ok.actual_home - ok.actual_away
    ok['pred_total_ha'] = ok.pred_home + ok.pred_away
    ok['actual_total_ha'] = ok.actual_home + ok.actual_away
    ok['home_margin_resid'] = ok.actual_home_margin - ok.pred_home_margin
    ok['home_key2'] = ok.home_team.map(norm_name)
    ok['away_key2'] = ok.away_team.map(norm_name)
    ok['date'] = pd.to_datetime(ok['date'], errors='coerce')
    diag = {'source': source, 'schedule_rows': int(len(schedule)), 'prediction_games': int(len(games)), 'joined_homeaway_rows': int(g.home_team.notna().sum()), 'orientation_ok_rows': int(g.orientation_ok.sum()), 'orientation_coverage': float(g.orientation_ok.mean()), 'neutral_known_rows': int(ok.neutral.notna().sum()), 'neutral_games': int(ok.neutral.sum()), 'non_neutral_games': int((~ok.neutral).sum()), 'venue_nonnull': int(ok.venue.notna().sum()), 'selected_columns': cols}
    return ok, diag


def score_metrics(df, pred_margin_col, hca_col=None):
    z = df.copy(); pm = z[pred_margin_col].astype(float); am = z.actual_home_margin.astype(float); resid = am - pm
    out = {'n': int(len(z)), 'margin_mae': float(np.mean(np.abs(resid))), 'margin_rmse': float(np.sqrt(np.mean(resid**2))), 'margin_bias_actual_minus_pred': float(np.mean(resid))}
    if hca_col is not None:
        h = z[hca_col].astype(float); ph = z.pred_home + h / 2.0; pa = z.pred_away - h / 2.0
        out['team_score_mae'] = float(np.mean(np.r_[np.abs(z.actual_home - ph), np.abs(z.actual_away - pa)])); out['total_mae'] = float(np.mean(np.abs(z.actual_total_ha - (ph + pa))))
    else:
        out['team_score_mae'] = float(np.mean(np.r_[np.abs(z.actual_home - z.pred_home), np.abs(z.actual_away - z.pred_away)])); out['total_mae'] = float(np.mean(np.abs(z.actual_total_ha - z.pred_total_ha)))
    return out


def fit_prob_metrics(train, test, pred_col):
    tr_res = train.actual_home_margin - train[pred_col]; mu = float(tr_res.mean()); sd = float(tr_res.std(ddof=1)); p = norm.cdf((test[pred_col] + mu) / max(sd, 1e-6)); y = (test.actual_home_margin > 0).astype(int)
    return {'brier': float(np.mean((p - y)**2)), 'logloss': logloss(p, y), 'ece': ece(p, y), 'train_resid_mean': mu, 'train_resid_sd': sd, 'pwin_mean': float(np.mean(p)), 'actual_win_rate': float(np.mean(y))}


def location_fold_predictions(ctx):
    fold_rows = []; season_summaries = []
    for sy in EVAL_SEASONS:
        tr = ctx[(ctx.season >= TRAIN_MIN_SEASON) & (ctx.season < sy) & (~ctx.neutral)].copy(); te = ctx[ctx.season == sy].copy()
        if tr.empty or te.empty: continue
        pooled = float(tr.home_margin_resid.mean()); te['hca_none'] = 0.0; te['hca_static27'] = np.where(te.neutral, 0.0, 2.7); te['hca_pooled'] = np.where(te.neutral, 0.0, pooled)
        hs = tr.groupby('home_key2').home_margin_resid.agg(['mean','count'])
        for k in [5.0, 10.0, 20.0]:
            vals=[]
            for r in te.itertuples(index=False):
                if r.neutral: vals.append(0.0); continue
                if r.home_key2 in hs.index:
                    m=float(hs.loc[r.home_key2,'mean']); n=float(hs.loc[r.home_key2,'count']); w=n/(n+k); vals.append(pooled+w*(m-pooled))
                else: vals.append(pooled)
            te[f'hca_team_k{int(k)}']=vals
        variants=['none','static27','pooled','team_k5','team_k10','team_k20']
        for v in variants:
            hc=f'hca_{v}'; pc=f'pred_margin_{v}'; te[pc]=te.pred_home_margin+te[hc]; tr2=ctx[(ctx.season>=TRAIN_MIN_SEASON)&(ctx.season<sy)].copy()
            if v=='none': tr2[pc]=tr2.pred_home_margin
            elif v=='static27': tr2[pc]=tr2.pred_home_margin+np.where(tr2.neutral,0.0,2.7)
            elif v=='pooled': tr2[pc]=tr2.pred_home_margin+np.where(tr2.neutral,0.0,pooled)
            else:
                k=float(v.split('k')[1]); vv=[]
                for r in tr2.itertuples(index=False):
                    if r.neutral: vv.append(0.0)
                    elif r.home_key2 in hs.index:
                        m=float(hs.loc[r.home_key2,'mean']); n=float(hs.loc[r.home_key2,'count']); w=n/(n+k); vv.append(pooled+w*(m-pooled))
                    else: vv.append(pooled)
                tr2[pc]=tr2.pred_home_margin+np.asarray(vv)
            sm=score_metrics(te,pc,hc); sm.update(fit_prob_metrics(tr2,te,pc)); sm.update({'season':sy,'variant':v,'pooled_hca_train':pooled}); season_summaries.append(sm)
        fold_rows.append(te)
    return pd.concat(fold_rows,ignore_index=True),pd.DataFrame(season_summaries)


def choose_location_model(summary):
    piv=summary.groupby('variant').agg(margin_mae=('margin_mae','mean'),team_score_mae=('team_score_mae','mean'),brier=('brier','mean'),logloss=('logloss','mean'),ece=('ece','mean')).sort_values(['margin_mae','brier']); base=piv.loc['none']
    pooled_seasons=summary[summary.variant=='pooled'].set_index('season').margin_mae; none_seasons=summary[summary.variant=='none'].set_index('season').margin_mae; pooled_wins=int((pooled_seasons<none_seasons).sum()); pooled_gain=float(base.margin_mae-piv.loc['pooled','margin_mae']); pooled_pass=pooled_gain>=0.05 and pooled_wins>=2
    best_team=min(['team_k5','team_k10','team_k20'],key=lambda v:piv.loc[v,'margin_mae']); team_s=summary[summary.variant==best_team].set_index('season').margin_mae; team_wins=int((team_s<pooled_seasons).sum()); team_gain=float(piv.loc['pooled','margin_mae']-piv.loc[best_team,'margin_mae']); team_pass=pooled_pass and team_gain>=0.03 and team_wins>=2
    if team_pass: selected=best_team
    elif pooled_pass: selected='pooled'
    else:
        static_s=summary[summary.variant=='static27'].set_index('season').margin_mae; static_gain=float(base.margin_mae-piv.loc['static27','margin_mae']); static_wins=int((static_s<none_seasons).sum()); selected='static27' if (static_gain>=.05 and static_wins>=2) else 'none'
    return piv,{'pooled_gain_margin_mae':pooled_gain,'pooled_season_wins':pooled_wins,'pooled_pass':pooled_pass,'best_team_variant':best_team,'team_gain_vs_pooled':team_gain,'team_season_wins_vs_pooled':team_wins,'team_pass':team_pass,'selected_variant':selected}


def build_team_schedule_features(ctx):
    x=ctx[['game_id','season','date','home_team','away_team','home_key2','away_key2','neutral']].drop_duplicates('game_id').copy().sort_values(['date','game_id']).reset_index(drop=True); hist=defaultdict(deque); rows=[]
    for r in x.itertuples(index=False):
        dt=pd.Timestamp(r.date).normalize(); feat={'game_id':r.game_id}
        for side,key in [('home',r.home_key2),('away',r.away_key2)]:
            q=hist[key]; prior=[d for d in q if d<dt]; rest=int((dt-prior[-1]).days) if prior else np.nan; g7=sum(1 for d in prior if 0<(dt-d).days<=7); g14=sum(1 for d in prior if 0<(dt-d).days<=14)
            feat[f'{side}_rest_raw']=rest; feat[f'{side}_rest']=7.0 if pd.isna(rest) else float(min(max(rest,0),7)); feat[f'{side}_games7']=int(g7); feat[f'{side}_games14']=int(g14)
        rows.append(feat); hist[r.home_key2].append(dt); hist[r.away_key2].append(dt)
    f=pd.DataFrame(rows); f['rest_diff']=f.home_rest-f.away_rest; f['games7_diff']=f.home_games7-f.away_games7; f['games14_diff']=f.home_games14-f.away_games14; f['min_rest']=f[['home_rest','away_rest']].min(axis=1); f['max_games7']=f[['home_games7','away_games7']].max(axis=1); f['schedule_stress']=((f.min_rest<=1)|(f.max_games7>=3)).astype(int); return f


def apply_location_variant_to_fold(ctx,sy,variant):
    tr_non=ctx[(ctx.season>=TRAIN_MIN_SEASON)&(ctx.season<sy)&(~ctx.neutral)].copy(); tr_all=ctx[(ctx.season>=TRAIN_MIN_SEASON)&(ctx.season<sy)].copy(); te=ctx[ctx.season==sy].copy(); pooled=float(tr_non.home_margin_resid.mean()) if len(tr_non) else 0.0; hs=tr_non.groupby('home_key2').home_margin_resid.agg(['mean','count']) if len(tr_non) else pd.DataFrame()
    def hca_for(d):
        if variant=='none': return np.zeros(len(d))
        if variant=='static27': return np.where(d.neutral,0.0,2.7)
        if variant=='pooled': return np.where(d.neutral,0.0,pooled)
        k=float(variant.split('k')[1]); out=[]
        for r in d.itertuples(index=False):
            if r.neutral: out.append(0.0)
            elif r.home_key2 in hs.index:
                m=float(hs.loc[r.home_key2,'mean']); n=float(hs.loc[r.home_key2,'count']); w=n/(n+k); out.append(pooled+w*(m-pooled))
            else: out.append(pooled)
        return np.asarray(out)
    for d in [tr_all,te]:
        d['hca_selected']=hca_for(d); d['pred_margin_loc']=d.pred_home_margin+d.hca_selected; d['margin_resid_loc']=d.actual_home_margin-d.pred_margin_loc
    return tr_all,te


def neutral_audit(ctx, selected_variant):
    rows=[]
    for sy in EVAL_SEASONS:
        tr,te=apply_location_variant_to_fold(ctx,sy,selected_variant); te=te[te.neutral].copy()
        if te.empty: continue
        trn=tr[tr.neutral].copy(); ref=trn if len(trn)>=100 else tr; mu=float(ref.margin_resid_loc.mean()); sd=float(ref.margin_resid_loc.std(ddof=1)); p=norm.cdf((te.pred_margin_loc+mu)/max(sd,1e-6)); y=(te.actual_home_margin>0).astype(int); resid=te.margin_resid_loc
        rows.append({'season':sy,'n':len(te),'margin_mae':float(abs(resid).mean()),'margin_bias':float(resid.mean()),'brier':float(np.mean((p-y)**2)),'logloss':logloss(p,y),'ece':ece(p,y),'train_neutral_n':int(len(trn)),'train_ref_n':int(len(ref))})
    return pd.DataFrame(rows)


def rest_mean_test(ctx,schedf,selected_variant):
    x=ctx.merge(schedf,on='game_id',how='left',validate='one_to_one'); rows=[]; features=['rest_diff','games7_diff','games14_diff']
    for sy in EVAL_SEASONS:
        tr,te=apply_location_variant_to_fold(x,sy,selected_variant); tr=tr.dropna(subset=features+['margin_resid_loc']); te=te.dropna(subset=features+['margin_resid_loc']); model=LinearRegression().fit(tr[features],tr.margin_resid_loc); corr=model.predict(te[features]); te['pred_margin_rest']=te.pred_margin_loc+corr; te['margin_resid_rest']=te.actual_home_margin-te.pred_margin_rest
        base=float(np.mean(np.abs(te.margin_resid_loc))); new=float(np.mean(np.abs(te.margin_resid_rest))); tr_corr=model.predict(tr[features]); tr_res2=tr.margin_resid_loc-tr_corr; mu=float(np.mean(tr_res2)); sd=float(np.std(tr_res2,ddof=1)); p0=norm.cdf((te.pred_margin_loc+float(tr.margin_resid_loc.mean()))/float(tr.margin_resid_loc.std(ddof=1))); p1=norm.cdf((te.pred_margin_rest+mu)/sd); y=(te.actual_home_margin>0).astype(int)
        rows.append({'season':sy,'n':len(te),'base_margin_mae':base,'rest_margin_mae':new,'mae_gain':base-new,'base_brier':float(np.mean((p0-y)**2)),'rest_brier':float(np.mean((p1-y)**2)),'base_logloss':logloss(p0,y),'rest_logloss':logloss(p1,y),'coef_rest_diff':float(model.coef_[0]),'coef_games7_diff':float(model.coef_[1]),'coef_games14_diff':float(model.coef_[2]),'intercept':float(model.intercept_)})
    res=pd.DataFrame(rows); gain=float(res.mae_gain.mean()); wins=int((res.mae_gain>0).sum()); return res,{'mean_mae_gain':gain,'season_wins':wins,'pass':bool(gain>=0.05 and wins>=2)}


def pace_test(ctx,schedf,r2,pace):
    rp=r2[['game_id','team','gposs']].copy(); pp=pace[['game_id','team','pred_pace']].copy(); rp['game_id']=rp.game_id.astype(str); pp['game_id']=pp.game_id.astype(str); t=rp.merge(pp,on=['game_id','team'],how='inner'); pg=t.groupby('game_id').agg(actual_pace=('gposs','mean'),pred_pace=('pred_pace','mean')).reset_index(); x=ctx[['game_id','season']].drop_duplicates().merge(pg,on='game_id',how='inner').merge(schedf,on='game_id',how='inner'); x['pace_resid']=x.actual_pace-x.pred_pace; features=['rest_diff','games7_diff','games14_diff']; rows=[]
    for sy in EVAL_SEASONS:
        tr=x[(x.season>=TRAIN_MIN_SEASON)&(x.season<sy)].dropna(subset=features+['pace_resid']); te=x[x.season==sy].dropna(subset=features+['pace_resid']).copy(); m=LinearRegression().fit(tr[features],tr.pace_resid); te['pred_pace_rest']=te.pred_pace+m.predict(te[features]); b=float(np.mean(abs(te.actual_pace-te.pred_pace))); n=float(np.mean(abs(te.actual_pace-te.pred_pace_rest))); rows.append({'season':sy,'n':len(te),'base_pace_mae':b,'rest_pace_mae':n,'mae_gain':b-n,'coef_rest_diff':float(m.coef_[0]),'coef_games7_diff':float(m.coef_[1]),'coef_games14_diff':float(m.coef_[2])})
    d=pd.DataFrame(rows); gain=float(d.mae_gain.mean()); wins=int((d.mae_gain>0).sum()); return d,{'mean_mae_gain':gain,'season_wins':wins,'pass':bool(gain>=0.03 and wins>=2),'joined_games':int(len(x))}


def travel_source_audit(schedule_schema,context_diag):
    cols=schedule_schema['columns']; coord=[c for c in cols if any(t in c.lower() for t in ['latitude','longitude','lat','lon','elevation','altitude','timezone'])]; city=[c for c in cols if any(t in c.lower() for t in ['city','state','venue'])]; has_lat=any('lat' in c.lower() for c in coord); has_lon=any(('lon' in c.lower() or 'long' in c.lower()) for c in coord); ready=bool(has_lat and has_lon and context_diag.get('venue_nonnull',0)>0)
    return {'coordinate_columns':coord,'venue_location_columns':city,'schedule_has_latlon':ready,'decision':'PASS-DATA' if ready else 'HOLD','reason':'Schedule provides usable coordinate fields' if ready else 'Current canonical schedule lacks a complete venue/team coordinate contract; external geocoding would be required.'}


def decide_gate_rows(diag,loc_pivot,loc_dec,neutral,schedf,rest_dec,pace_dec,travel,unc_dec):
    gates=[]
    def add(gid,status,decision,strength,result,next_action): gates.append({'gate_id':gid,'status':status,'decision':decision,'evidence_strength':strength,'result':result,'next_action':next_action})
    add('T16.01','GREEN','PASS','STRONG','Certified R1/R2 artifact generation uses expected ORtg = 100 + O_A - D_B and symmetric pace state with no home/away/neutral input; current frozen control therefore has zero explicit venue adjustment.','Use zero-venue control in T16 tournament; do not port old HCA code unchanged.')
    cov=diag.get('orientation_coverage',0); add('T16.02','GREEN' if cov>=0.95 else 'HOLD','PASS' if cov>=0.95 else 'HOLD','STRONG' if cov>=0.95 else 'MODERATE',f"Canonical schedule home/away orientation joined {diag.get('orientation_ok_rows',0):,}/{diag.get('prediction_games',0):,} prediction games ({cov:.2%}); neutral games={diag.get('neutral_games',0):,}; venue non-null={diag.get('venue_nonnull',0):,}.",'Preserve unique game-id orientation and neutral assertions.' if cov>=0.95 else 'Repair remaining site/orientation coverage before production venue use.')
    add('T16.03','GREEN' if loc_dec['pooled_pass'] else 'REJECTED','PASS' if loc_dec['pooled_pass'] else 'REJECTED','STRONG',f"Chronology-safe pooled HCA mean margin-MAE gain vs no-HCA={loc_dec['pooled_gain_margin_mae']:.4f} points with {loc_dec['pooled_season_wins']}/3 OOS season wins.",'Carry pooled learned HCA into model tournament.' if loc_dec['pooled_pass'] else 'Do not add pooled HCA to frozen mean model.')
    selected=loc_dec['selected_variant']; row=loc_pivot.loc[selected]; add('T16.04','GREEN','PASS','STRONG',f"Predeclared no-HCA/static-2.7/learned-pooled/team-shrunk tournament selected {selected}; mean margin MAE={row.margin_mae:.4f}, Brier={row.brier:.5f}, log loss={row.logloss:.5f}.",f'Use {selected} as location control for downstream T17/T20 unless T16.06 earns extra complexity.')
    if neutral.empty: add('T16.05','HOLD','HOLD','MODERATE','No neutral-site games could be identified from canonical schedule context.','Acquire/repair neutral-site flag before tournament certification.')
    else: add('T16.05','GREEN','PASS','STRONG',f"Neutral-site OOS audit covers {int(neutral.n.sum()):,} games across {len(neutral)} seasons; weighted margin bias={np.average(neutral.margin_bias,weights=neutral.n):.3f}, weighted Brier={np.average(neutral.brier,weights=neutral.n):.5f}.",'Use zero additive HCA on neutral games; full postseason transfer remains T21.')
    add('T16.06','GREEN' if loc_dec['team_pass'] else 'REJECTED','PASS' if loc_dec['team_pass'] else 'REJECTED','STRONG',f"Best fixed partial-pooling challenger={loc_dec['best_team_variant']}; incremental margin-MAE gain vs pooled={loc_dec['team_gain_vs_pooled']:.4f} with {loc_dec['team_season_wins_vs_pooled']}/3 season wins.",'Promote selected partial pooling.' if loc_dec['team_pass'] else 'Reject team-specific venue complexity; retain simpler location winner.')
    add('T16.07','FROZEN','FROZEN','STRONG',f"Location architecture frozen for this research stage as {selected}; venue is applied exactly once to margin/team scores and zeroed on neutral sites.",'Use this location specification as control for T17 and later distribution refresh.')
    covr=float((schedf.home_rest_raw.notna()&schedf.away_rest_raw.notna()).mean()); add('T17.01','GREEN','PASS','STRONG',f"Rest/compression features derived deterministically from prior schedule only: paired prior-rest coverage={covr:.2%}; all games have capped rest plus prior games-in-7/14 counts; current game is appended only after features are computed.",'Use rest_diff, games7_diff, games14_diff as the registered simple context family.')
    add('T17.02','GREEN' if rest_dec['pass'] else 'REJECTED','PASS' if rest_dec['pass'] else 'REJECTED','STRONG',f"Rest/compression margin correction mean OOS MAE gain={rest_dec['mean_mae_gain']:.4f} with {rest_dec['season_wins']}/3 season wins.",'Promote schedule-load mean correction.' if rest_dec['pass'] else 'Reject schedule-load mean correction; preserve context only for diagnostics/variance tests.')
    add('T17.03','GREEN' if pace_dec['pass'] else 'REJECTED','PASS' if pace_dec['pass'] else 'REJECTED','STRONG',f"Rest/compression pace challenger joined {pace_dec['joined_games']:,} games; mean possession-MAE gain={pace_dec['mean_mae_gain']:.4f} with {pace_dec['season_wins']}/3 season wins.",'Promote pace context.' if pace_dec['pass'] else 'Retain simpler pace state; no schedule-load pace adjustment.')
    if travel['decision']=='PASS-DATA':
        add('T17.04','GREEN','PASS','MODERATE/STRONG',f"Canonical schedule exposes coordinate fields sufficient to build venue travel features: {travel['coordinate_columns']}.",'Build chronology-safe team-base/venue distance features next.'); add('T17.05','HOLD','HOLD','MODERATE','Travel-effect empirical test not executed in this 15-gate run because T17.04 only just established the coordinate contract.','Execute distance/time-zone/altitude challenger in next run.')
    else:
        add('T17.04','HOLD','HOLD','STRONG',travel['reason']+f" Location-like columns found: {travel['venue_location_columns'][:12]}.",'Do not manufacture geocodes; use an audited external team/venue coordinate source before travel modeling.'); add('T17.05','HOLD','HOLD','STRONG','Travel-effect gate cannot execute without a production-grade coordinate contract; rest/compression tests continue independently.','Resume only after T17.04 source HOLD is solved.')
    add('T17.06','GREEN' if unc_dec['pass'] else 'REJECTED','PASS' if unc_dec['pass'] else 'REJECTED','STRONG',f"Stage+schedule-stress uncertainty challenger mean NLL gain={unc_dec['mean_nll_gain']:.5f} with {unc_dec['season_wins']}/3 season wins.",'Promote stress multiplier in sigma model.' if unc_dec['pass'] else 'Retain stage/maturity-only uncertainty width.')
    add('T17.07','GREEN','PASS','STRONG',f"Schedule-context decision complete for executable branches: mean={'PROMOTE' if rest_dec['pass'] else 'REJECT'}, pace={'PROMOTE' if pace_dec['pass'] else 'REJECT'}, uncertainty={'PROMOTE' if unc_dec['pass'] else 'REJECT'}, travel=HOLD pending source.",'Carry only empirically promoted schedule effects; keep travel as isolated optional HOLD.')
    return gates


def main():
    games,r2,pace,schedule=load_inputs(); schema=audit_schedule_schema(schedule); ctx,diag=build_game_context(games,schedule)
    if ctx is None or len(ctx)<1000: raise RuntimeError(f'Could not build sufficient home/away context: {diag}')
    (OUT/'context_diagnostics.json').write_text(json.dumps(diag,indent=2,default=str))
    folds,loc_summary=location_fold_predictions(ctx); loc_pivot,loc_dec=choose_location_model(loc_summary); loc_summary.to_csv(OUT/'location_season_metrics.csv',index=False); loc_pivot.reset_index().to_csv(OUT/'location_model_summary.csv',index=False); (OUT/'location_decision.json').write_text(json.dumps(loc_dec,indent=2,default=str)); neutral=neutral_audit(ctx,loc_dec['selected_variant']); neutral.to_csv(OUT/'neutral_site_audit.csv',index=False)
    schedf=build_team_schedule_features(ctx); schedf.to_csv(OUT/'schedule_context_features.csv',index=False); rest_res,rest_dec=rest_mean_test(ctx,schedf,loc_dec['selected_variant']); rest_res.to_csv(OUT/'rest_margin_metrics.csv',index=False); pace_res,pace_dec=pace_test(ctx,schedf,r2,pace); pace_res.to_csv(OUT/'rest_pace_metrics.csv',index=False)
    xctx=ctx.merge(schedf,on='game_id',how='left',validate='one_to_one'); urows=[]
    for sy in EVAL_SEASONS:
        tr,te=apply_location_variant_to_fold(xctx,sy,loc_dec['selected_variant']); tr['stage']=tr.n_avg.map(stage_bucket); te['stage']=te.n_avg.map(stage_bucket); stage_sd=tr.groupby('stage').margin_resid_loc.std(ddof=1).to_dict(); global_sd=float(tr.margin_resid_loc.std(ddof=1)); base_sigma=te.stage.map(lambda s:stage_sd.get(s,global_sd)).astype(float).values; tr_sigma=tr.stage.map(lambda s:stage_sd.get(s,global_sd)).astype(float).values; tr_std=tr.margin_resid_loc.values/np.clip(tr_sigma,1e-6,None); stress=tr.schedule_stress.fillna(0).values==1; non=~stress
        if stress.sum()>=20 and non.sum()>=50:
            sd_s=float(np.std(tr_std[stress],ddof=1)); sd_n=float(np.std(tr_std[non],ddof=1)); raw=sd_s/max(sd_n,1e-6); w=stress.sum()/(stress.sum()+50.0); mult=1+w*(raw-1)
        else: mult=1.0
        chall=base_sigma*np.where(te.schedule_stress.fillna(0).values==1,mult,1.0); b=gaussian_nll(te.margin_resid_loc.values,base_sigma); c=gaussian_nll(te.margin_resid_loc.values,chall); urows.append({'season':sy,'n':len(te),'base_nll':b,'stress_nll':c,'nll_gain':b-c,'stress_multiplier':mult,'stress_n':int((te.schedule_stress.fillna(0)==1).sum()),'stress_mae':float(te.loc[te.schedule_stress==1,'margin_resid_loc'].abs().mean()) if (te.schedule_stress==1).any() else np.nan,'nonstress_mae':float(te.loc[te.schedule_stress==0,'margin_resid_loc'].abs().mean()) if (te.schedule_stress==0).any() else np.nan})
    unc_res=pd.DataFrame(urows); unc_res.to_csv(OUT/'schedule_uncertainty_metrics.csv',index=False); ugain=float(unc_res.nll_gain.mean()); uwins=int((unc_res.nll_gain>0).sum()); unc_dec={'mean_nll_gain':ugain,'season_wins':uwins,'pass':bool(ugain>0 and uwins>=2)}; travel=travel_source_audit(schema,diag); (OUT/'travel_source_audit.json').write_text(json.dumps(travel,indent=2,default=str))
    gates=decide_gate_rows(diag,loc_pivot,loc_dec,neutral,schedf,rest_dec,pace_dec,travel,unc_dec); pd.DataFrame(gates).to_csv(OUT/'gate_decisions_T16_T17.csv',index=False); summary={'context':diag,'location_decision':loc_dec,'neutral':neutral.to_dict(orient='records'),'rest_decision':rest_dec,'pace_decision':pace_dec,'uncertainty_decision':unc_dec,'travel_source':travel,'gates':gates}; (OUT/'summary.json').write_text(json.dumps(summary,indent=2,default=str)); print(json.dumps(summary,indent=2,default=str))


if __name__=='__main__': main()
