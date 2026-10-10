from __future__ import annotations

import numpy as np
import pandas as pd
from scipy.stats import spearmanr
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import make_pipeline
import prediction_first_third15 as m


def clean_id(x):
    if pd.isna(x): return np.nan
    s=str(x).strip()
    return s[:-2] if s.endswith('.0') else s


def load_all_v4():
    gp=pd.read_parquet(m.G15/'game_predictions.parquet'); gp['game_id']=gp.game_id.astype(str); gp['date']=pd.to_datetime(gp.date)
    r2=pd.read_parquet(m.G15/'r2_team_predictions.parquet'); r2['game_id']=r2.game_id.astype(str); r2['date']=pd.to_datetime(r2.date)
    pace=pd.read_parquet(m.G10/'pace_predictions.parquet'); pace['game_id']=pace.game_id.astype(str)
    if 'pred_pace' not in pace.columns:
        p=pace[['game_id','pace_pred']].drop_duplicates('game_id').rename(columns={'pace_pred':'pred_pace'})
        pace=r2[['game_id','team']].merge(p,on='game_id',how='left',validate='many_to_one')
    cf=pd.read_parquet(m.PF/'component_features.parquet'); cf['game_id']=cf.game_id.astype(str); cf['date']=pd.to_datetime(cf.date)
    venue=pd.read_csv(m.VEN/'identity_match_ledger.csv',dtype={'game_id':str})
    venue['neutral_known']=venue.neutral.notna(); venue['neutral']=venue.neutral.map(lambda x: str(x).lower()=='true' if pd.notna(x) else np.nan); venue['espn_game_id']=venue.espn_game_id.map(clean_id)
    sched=pd.read_parquet(m.RAW+'mbb/ncaa_mbb_schedule_master.parquet'); gid='game_id' if 'game_id' in sched.columns else 'contest_id'; sched=sched.rename(columns={gid:'game_id'}); sched['game_id']=sched.game_id.astype(str); sched['hk']=sched.home.map(m.nn); sched['ak']=sched.away.map(m.nn)
    return gp,r2,pace,cf,venue,sched

_orig_post=m.load_postseason_map
def post_v4(venue):
    z=_orig_post(venue); z['espn_game_id']=z.espn_game_id.map(clean_id); return z


def reliability_models_v4(pred):
    rec=[]; dec=[]; pred=pred.copy()
    pred['abs_margin']=abs(pred.margin_resid); pred['abs_total']=abs(pred.total_resid); pred['team_err']=(abs(pred.actual_home-pred.mu_home)+abs(pred.actual_away-pred.mu_away))/2; pred['win_err']=(pred.pwin_base-(pred.actual_margin>0).astype(int))**2; pred['gap_abs']=abs(pred.pred_margin); pred['win_uncert']=pred.pwin_base*(1-pred.pwin_base); pred['neutral_i']=pred.neutral.astype(int); pred['post_i']=pred.postseason.astype(int)
    features=['n_avg','gap_abs','pred_total','pred_pace','win_uncert','neutral_i','post_i','sigma_margin','sigma_total']; targets={'team_score':'team_err','margin':'abs_margin','total':'abs_total','win_brier':'win_err'}
    # 2023 is the first reliability-labelled OOS cohort and therefore training-only.
    for sy in [2024,2025]:
        tr=pred[(pred.season>=2023)&(pred.season<sy)].copy(); te=pred[pred.season==sy].copy()
        if len(tr)<100 or len(te)<100: continue
        for name,targ in targets.items():
            model=make_pipeline(StandardScaler(),Ridge(alpha=20.0)).fit(tr[features].fillna(0),np.log1p(tr[targ])); risk=model.predict(te[features].fillna(0)); rho=float(spearmanr(risk,te[targ]).statistic)
            te2=pd.DataFrame({'risk':risk,'err':te[targ].values}); te2['decile']=pd.qcut(te2.risk,10,labels=False,duplicates='drop'); d=te2.groupby('decile').err.mean(); ratio=float(d.iloc[-1]/max(d.iloc[0],1e-9)); mono=float(spearmanr(d.index,d.values).statistic) if len(d)>=3 else np.nan
            rec.append({'season':sy,'target':name,'n':len(te),'spearman':rho,'top_bottom_ratio':ratio,'decile_monotonicity':mono,'low_decile_error':float(d.iloc[0]),'high_decile_error':float(d.iloc[-1])})
            for ix,val in d.items(): dec.append({'season':sy,'target':name,'decile':int(ix),'mean_error':float(val),'n':int((te2.decile==ix).sum())})
    R=pd.DataFrame(rec); R.to_csv(m.OUT/'t22_target_reliability.csv',index=False); pd.DataFrame(dec).to_csv(m.OUT/'t22_reliability_deciles.csv',index=False); return R

m.load_all=load_all_v4; m.load_postseason_map=post_v4; m.reliability_models=reliability_models_v4
m.main()
