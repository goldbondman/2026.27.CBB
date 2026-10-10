from __future__ import annotations

import numpy as np
import pandas as pd

import prediction_first_third15 as m

_orig_load=m.load_all


def _canon_espn_id(x):
    if pd.isna(x):
        return np.nan
    try:
        return str(int(float(x)))
    except Exception:
        s=str(x).strip()
        return s if s else np.nan


def load_all_repaired():
    gp,r2,pace,cf,venue,sched=_orig_load()
    # G10 stores one pace row per game as pace_pred/pace_actual. Expand it to the
    # two frozen R2 team rows so the downstream score reconstruction remains
    # exactly tied to the certified production pace artifact.
    if 'pred_pace' not in pace.columns:
        p=pace[['game_id','pace_pred']].drop_duplicates('game_id').rename(columns={'pace_pred':'pred_pace'})
        pace=r2[['game_id','team']].merge(p,on='game_id',how='left',validate='many_to_one')
    # Venue identity ledger has missing values, which causes pandas to infer the
    # ESPN game ID as float64. Canonicalize to integer-like strings so it joins
    # exactly to the postseason schedule map without changing eligibility/data.
    if 'espn_game_id' in venue.columns:
        venue['espn_game_id']=venue['espn_game_id'].map(_canon_espn_id)
    return gp,r2,pace,cf,venue,sched


def reliability_models_repaired(pred):
    """Chronological T22.01 evaluation only when a genuine prior OOS history exists.

    The corrected prediction table contains 2023-25 OOS games. Therefore 2023
    cannot be a reliability test fold without inventing a training sample.
    Evaluate 2024 from 2023 history and 2025 from 2023-24 history instead.
    """
    rec=[]; dec=[]
    pred=pred.copy()
    pred['abs_margin']=abs(pred.margin_resid)
    pred['abs_total']=abs(pred.total_resid)
    pred['team_err']=(abs(pred.actual_home-pred.mu_home)+abs(pred.actual_away-pred.mu_away))/2
    pred['win_err']=(pred.pwin_base-(pred.actual_margin>0).astype(int))**2
    pred['gap_abs']=abs(pred.pred_margin)
    pred['win_uncert']=pred.pwin_base*(1-pred.pwin_base)
    pred['neutral_i']=pred.neutral.astype(int)
    pred['post_i']=pred.postseason.astype(int)
    features=['n_avg','gap_abs','pred_total','pred_pace','win_uncert','neutral_i','post_i','sigma_margin','sigma_total']
    targets={'team_score':'team_err','margin':'abs_margin','total':'abs_total','win_brier':'win_err'}
    for sy in m.EVAL:
        tr=pred[pred.season<sy].copy()
        te=pred[pred.season==sy].copy()
        if tr.empty or te.empty:
            continue
        for name,targ in targets.items():
            model=m.make_pipeline(m.StandardScaler(),m.Ridge(alpha=20.0)).fit(
                tr[features].fillna(0), np.log1p(tr[targ])
            )
            risk=model.predict(te[features].fillna(0))
            rho=float(m.spearmanr(risk,te[targ]).statistic)
            te2=pd.DataFrame({'risk':risk,'err':te[targ].values})
            te2['decile']=pd.qcut(te2.risk,10,labels=False,duplicates='drop')
            d=te2.groupby('decile').err.mean()
            ratio=float(d.iloc[-1]/max(d.iloc[0],1e-9))
            mono=float(m.spearmanr(d.index,d.values).statistic) if len(d)>=3 else np.nan
            rec.append({'season':sy,'target':name,'n':len(te),'spearman':rho,
                        'top_bottom_ratio':ratio,'decile_monotonicity':mono,
                        'low_decile_error':float(d.iloc[0]),'high_decile_error':float(d.iloc[-1])})
            for ix,val in d.items():
                dec.append({'season':sy,'target':name,'decile':int(ix),
                            'mean_error':float(val),'n':int((te2.decile==ix).sum())})
    R=pd.DataFrame(rec)
    R.to_csv(m.OUT/'t22_target_reliability.csv',index=False)
    pd.DataFrame(dec).to_csv(m.OUT/'t22_reliability_deciles.csv',index=False)
    return R


m.load_all=load_all_repaired
m.reliability_models=reliability_models_repaired
m.main()
