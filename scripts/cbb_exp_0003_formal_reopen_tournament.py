from __future__ import annotations

import json, math
from pathlib import Path
import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import prediction_first_recency_activation15_repair as repair

# CBB-EXP-0003
# Objective CBB-OBJ-006 / Program CBB-PROG-016 / Question CBB-Q-037 / Ticket CBB-RT-003
# FORMAL IMMUTABLE-v14 REOPEN TOURNAMENT
#
# PREDECLARED RULES
# 1) Frozen v14 rows/means/native pwin are immutable benchmark outputs from artifact 11709535789.
# 2) Historical test folds remain 2024 and 2025. Each uses only prior seasons for calibration/model fitting.
# 3) Historical activation thresholds remain their leakage-safe prior-trained values: 12 for 2024, 15 for 2025.
#    The separately frozen 2026 deployment threshold=7 from CBB-EXP-0001 is NOT retroactively imposed.
# 4) Final challenger factor spec is CBB-EXP-0002 winner: adjusted-L7 + ORB + DRB.
# 5) Challenger overlay:
#    a) pooled affine early margin calibration only when both teams <= Game 10;
#    b) mature non-neutral active rows get Ridge residual correction from L7+ORB+DRB;
#    c) all other rows remain frozen v14;
#    d) any missing active features fail safe to frozen v14.
# 6) To preserve coherent scores, any margin delta is split +delta/2 home, -delta/2 away. Total mean is unchanged.
# 7) Native v14 pwin_final is the exact probability benchmark. Challenger uses frozen stage sigma_margin_final times
#    a multiplier selected on training data only by Brier minimization. No market data are used.
# 8) Joint score NLL is an independent common-fold audit: for each model, fit stage-conditioned bivariate Gaussian
#    score residual means/covariances on training rows only (20% shrinkage to global covariance), then score test rows.
# 9) Promotion/reopen PASS requires BOTH future folds:
#    - margin MAE gain > 0 and 95% date-block bootstrap lower bound > 0;
#    - team-score MAE gain >= 0; total MAE gain >= -1e-10;
#    - Brier gain >= 0 and log-loss gain >= 0 vs exact native v14 pwin_final;
#    - common-fold joint score NLL gain >= 0;
#    - challenger 50/80/90 margin coverage max absolute error <= 0.03;
#    - no audited cohort with n>=250 has margin-MAE regression worse than 0.10 points.
# 10) PASS authorizes a production REOPEN, not immediate deployment. v14 remains frozen until native architecture,
#     snapshot/API/ops recertification and new version/hash are completed.

FREEZE=Path('/tmp/freeze')
OUT=Path('results/cbb_exp_0003_reopen_tournament'); OUT.mkdir(parents=True,exist_ok=True)
FOLD_THRESH={2024:12,2025:15}
FEATURES=['d_adj_l7_margin','d_pre_orbr','d_pre_drbr']
SEED=37003; N_BOOT=1000
r=repair.r


def mae(y,p): return float(np.mean(np.abs(np.asarray(y,float)-np.asarray(p,float))))
def brier(y,p): return float(np.mean((np.asarray(y,float)-np.asarray(p,float))**2))
def logloss(y,p):
    p=np.clip(np.asarray(p,float),1e-8,1-1e-8); y=np.asarray(y,float)
    return float(np.mean(-(y*np.log(p)+(1-y)*np.log(1-p))))
def ece(y,p,bins=10):
    y=np.asarray(y,float); p=np.asarray(p,float); edges=np.linspace(0,1,bins+1); out=0.0
    for i in range(bins):
        m=(p>=edges[i])&(p<(edges[i+1] if i<bins-1 else edges[i+1]+1e-12))
        if m.any(): out += m.mean()*abs(p[m].mean()-y[m].mean())
    return float(out)

def fast_date_boot(df, n=N_BOOT, seed=SEED):
    x=df.dropna(subset=['game_date','actual_margin','v14_margin','ch_margin']).copy()
    x['gain']=np.abs(x.actual_margin-x.v14_margin)-np.abs(x.actual_margin-x.ch_margin)
    g=x.groupby('game_date')['gain'].agg(['sum','count']).reset_index(drop=True)
    sums=g['sum'].to_numpy(float); counts=g['count'].to_numpy(float); k=len(g); rng=np.random.default_rng(seed); vals=[]
    for _ in range(n):
        idx=rng.integers(0,k,size=k); vals.append(float(sums[idx].sum()/counts[idx].sum()))
    a=np.asarray(vals)
    return float(a.mean()),float(np.quantile(a,.025)),float(np.quantile(a,.975)),float(np.mean(a>0))

def early_coef(train):
    m=train.h_game_no.le(10)&train.a_game_no.le(10)
    z=train[m].dropna(subset=['actual_margin','v14_margin'])
    if len(z)<300: return np.array([0.,1.])
    return np.linalg.lstsq(np.c_[np.ones(len(z)),z.v14_margin],z.actual_margin,rcond=None)[0]

def add_early(df,coef):
    out=df.copy(); out['cal_margin']=out.v14_margin
    m=out.h_game_no.le(10)&out.a_game_no.le(10)
    out.loc[m,'cal_margin']=coef[0]+coef[1]*out.loc[m,'v14_margin']
    out['cal_resid']=out.actual_margin-out.cal_margin
    return out

def fit_overlay(train,test,th):
    tr=add_early(train,early_coef(train)); te=add_early(test,early_coef(train))
    active_tr=tr.is_non_neutral & tr.min_game_no.ge(th)
    active_te=te.is_non_neutral & te.min_game_no.ge(th)
    trfit=tr[active_tr].dropna(subset=['cal_resid']+FEATURES)
    tefit=te[active_te].dropna(subset=FEATURES)
    model=make_pipeline(StandardScaler(),Ridge(alpha=30)).fit(trfit[FEATURES],trfit.cal_resid)
    # full policy begins with early-calibrated baseline; missing/neutral/inactive stays there/frozen as defined.
    tr['ch_margin']=tr.cal_margin; te['ch_margin']=te.cal_margin
    # in-sample training predictions used only to select sigma scale, never to score a future fold.
    trix=trfit.index; teix=tefit.index
    tr.loc[trix,'ch_margin']=tr.loc[trix,'cal_margin']+model.predict(trfit[FEATURES])
    te.loc[teix,'ch_margin']=te.loc[teix,'cal_margin']+model.predict(tefit[FEATURES])
    tr['active_feature_overlay']=False; te['active_feature_overlay']=False
    tr.loc[trix,'active_feature_overlay']=True; te.loc[teix,'active_feature_overlay']=True
    return tr,te,model

def select_sigma_scale(train):
    y=(train.actual_margin>0).astype(int).to_numpy(); scales=np.linspace(.70,1.20,51); best=None
    for sc in scales:
        p=norm.cdf(train.ch_margin/(train.sigma_margin_final*sc)); val=brier(y,p)
        if best is None or val<best[0]: best=(val,float(sc))
    return best[1]

def fit_stage_dist(train, home_col, away_col):
    z=train.dropna(subset=['actual_home','actual_away',home_col,away_col,'stage']).copy()
    z['rh']=z.actual_home-z[home_col]; z['ra']=z.actual_away-z[away_col]
    global_mean=np.array([z.rh.mean(),z.ra.mean()],float); global_cov=np.cov(z[['rh','ra']].to_numpy().T)+np.eye(2)*1e-6
    params={}
    for st,g in z.groupby('stage'):
        if len(g)<100:
            params[str(st)]=(global_mean,global_cov); continue
        mu=np.array([g.rh.mean(),g.ra.mean()],float); cov=np.cov(g[['rh','ra']].to_numpy().T)
        cov=.8*cov+.2*global_cov+np.eye(2)*1e-6
        params[str(st)]=(mu,cov)
    return params,(global_mean,global_cov)
def joint_nll(test, home_col, away_col, params, glob):
    vals=[]
    for x in test.dropna(subset=['actual_home','actual_away',home_col,away_col,'stage']).itertuples():
        mu,cov=params.get(str(x.stage),glob); resid=np.array([x.actual_home-getattr(x,home_col),x.actual_away-getattr(x,away_col)])-mu
        sign,ld=np.linalg.slogdet(cov)
        if sign<=0: continue
        vals.append(math.log(2*math.pi)+.5*ld+.5*float(resid@np.linalg.solve(cov,resid)))
    return float(np.mean(vals))
def coverage(test,scale):
    out=[]
    for level,zv in [(0.5,norm.ppf(.75)),(0.8,norm.ppf(.9)),(0.9,norm.ppf(.95))]:
        hw=zv*test.sigma_margin_final*scale
        emp=float((np.abs(test.actual_margin-test.ch_margin)<=hw).mean())
        out.append({'level':level,'empirical':emp,'error':emp-level})
    return out

def cohort_audit(test):
    q4=float(test.pred_pace.quantile(.75))
    cohorts={
        'early_0_1':test.min_game_no.le(1),
        'mature_10plus':test.min_game_no.ge(10),
        'neutral':~test.is_non_neutral,
        'postseason':test.postseason.astype(bool),
        'pace_q4':test.pred_pace.ge(q4),
        'gap_0_3':test.gap_abs.le(3),
        'gap_15plus':test.gap_abs.ge(15),
    }
    rows=[]
    for name,m in cohorts.items():
        z=test[m]
        if len(z)==0: continue
        bg=mae(z.actual_margin,z.v14_margin); cg=mae(z.actual_margin,z.ch_margin)
        rows.append({'cohort':name,'n':len(z),'v14_margin_mae':bg,'challenger_margin_mae':cg,'gain':bg-cg})
    return rows

def main():
    frozen=pd.read_parquet(FREEZE/'frozen_candidate_predictions.parquet').copy(); frozen['game_id']=frozen.game_id.astype(str)
    assert frozen.game_id.duplicated().sum()==0 and frozen.season.max()<=2025
    q=repair.repaired_prep().copy(); q['game_id']=q.game_id.astype(str)
    keep=['season','game_id','h_game_no','a_game_no','min_game_no','is_non_neutral','game_date']+FEATURES
    feat=q[keep].drop_duplicates(['season','game_id'])
    d=frozen.merge(feat,on=['season','game_id'],how='left',validate='one_to_one')
    d['v14_margin']=d.pred_margin; d['v14_total']=d.pred_total; d['v14_home']=d.mu_home; d['v14_away']=d.mu_away
    d['min_game_no']=pd.to_numeric(d.min_game_no,errors='coerce')
    d['h_game_no']=pd.to_numeric(d.h_game_no,errors='coerce'); d['a_game_no']=pd.to_numeric(d.a_game_no,errors='coerce')
    d['is_non_neutral']=d.is_non_neutral.fillna(~d.neutral.astype(bool)).astype(bool)
    d['game_date']=pd.to_datetime(d.game_date.fillna(d.date),errors='coerce')

    folds=[]; boots=[]; probs=[]; covrows=[]; cohorts=[]; prediction_rows=[]
    for sy in [2024,2025]:
        train=d[d.season.lt(sy)].copy(); test=d[d.season.eq(sy)].copy(); th=FOLD_THRESH[sy]
        # Rows missing game-number metadata cannot be calibrated/activated: fill impossible large values only after preserving a fail-safe mask.
        meta_train=train.h_game_no.notna()&train.a_game_no.notna()&train.min_game_no.notna(); meta_test=test.h_game_no.notna()&test.a_game_no.notna()&test.min_game_no.notna()
        tr0=train[meta_train].copy(); te0=test[meta_test].copy()
        tr,te,_=fit_overlay(tr0,te0,th)
        # Merge policy outputs back; unmatched metadata rows remain frozen v14.
        for base in (train,test): base['ch_margin']=base.v14_margin; base['active_feature_overlay']=False
        train.loc[tr.index,'ch_margin']=tr.ch_margin; train.loc[tr.index,'active_feature_overlay']=tr.active_feature_overlay
        test.loc[te.index,'ch_margin']=te.ch_margin; test.loc[te.index,'active_feature_overlay']=te.active_feature_overlay
        # coherent score split, unchanged total
        train['delta']=train.ch_margin-train.v14_margin; test['delta']=test.ch_margin-test.v14_margin
        for z in (train,test):
            z['ch_home']=z.v14_home+z.delta/2; z['ch_away']=z.v14_away-z.delta/2; z['ch_total']=z.ch_home+z.ch_away
        scale=select_sigma_scale(train.dropna(subset=['sigma_margin_final','ch_margin','actual_margin']))
        y=(test.actual_margin>0).astype(int).to_numpy(); pch=norm.cdf(test.ch_margin/(test.sigma_margin_final*scale)); pb=test.pwin_final.to_numpy(float)
        # Common-fold stage Gaussian audit.
        bp,bg=fit_stage_dist(train,'v14_home','v14_away'); cp,cg=fit_stage_dist(train,'ch_home','ch_away')
        bnll=joint_nll(test,'v14_home','v14_away',bp,bg); cnll=joint_nll(test,'ch_home','ch_away',cp,cg)
        cov=coverage(test,scale); maxcov=max(abs(x['error']) for x in cov)
        bm,lo,hi,pp=fast_date_boot(test,seed=SEED+sy)
        team_base=float((np.abs(test.actual_home-test.v14_home).mean()+np.abs(test.actual_away-test.v14_away).mean())/2)
        team_ch=float((np.abs(test.actual_home-test.ch_home).mean()+np.abs(test.actual_away-test.ch_away).mean())/2)
        metrics={
            'season':sy,'n':len(test),'matched_metadata_n':int(meta_test.sum()),'active_overlay_n':int(test.active_feature_overlay.sum()),'historical_activation_threshold':th,'sigma_scale':scale,
            'v14_margin_mae':mae(test.actual_margin,test.v14_margin),'challenger_margin_mae':mae(test.actual_margin,test.ch_margin),
            'margin_mae_gain':mae(test.actual_margin,test.v14_margin)-mae(test.actual_margin,test.ch_margin),
            'v14_team_mae':team_base,'challenger_team_mae':team_ch,'team_mae_gain':team_base-team_ch,
            'v14_total_mae':mae(test.actual_total,test.v14_total),'challenger_total_mae':mae(test.actual_total,test.ch_total),'total_mae_gain':mae(test.actual_total,test.v14_total)-mae(test.actual_total,test.ch_total),
            'v14_brier':brier(y,pb),'challenger_brier':brier(y,pch),'brier_gain':brier(y,pb)-brier(y,pch),
            'v14_logloss':logloss(y,pb),'challenger_logloss':logloss(y,pch),'logloss_gain':logloss(y,pb)-logloss(y,pch),
            'v14_ece':ece(y,pb),'challenger_ece':ece(y,pch),'ece_gain':ece(y,pb)-ece(y,pch),
            'v14_joint_nll_commonfold':bnll,'challenger_joint_nll_commonfold':cnll,'joint_nll_gain':bnll-cnll,'max_abs_coverage_error':maxcov,
        }
        folds.append(metrics); boots.append({'season':sy,'mean_gain':bm,'lo95':lo,'hi95':hi,'p_positive':pp})
        for x in cov: covrows.append({'season':sy,'scale':scale,**x})
        for x in cohort_audit(test): cohorts.append({'season':sy,**x})
        prediction_rows.append(test[['season','game_id','date','actual_home','actual_away','actual_margin','actual_total','v14_home','v14_away','v14_margin','v14_total','pwin_final','ch_home','ch_away','ch_margin','ch_total','active_feature_overlay','stage','neutral','postseason','pred_pace','gap_abs']].copy())

    F=pd.DataFrame(folds); B=pd.DataFrame(boots); C=pd.DataFrame(covrows); K=pd.DataFrame(cohorts); P=pd.concat(prediction_rows,ignore_index=True)
    # Promotion gates.
    core=(F.margin_mae_gain>0).all() and (B.lo95>0).all() and (F.team_mae_gain>=-1e-10).all() and (F.total_mae_gain>=-1e-10).all()
    proper=(F.brier_gain>=0).all() and (F.logloss_gain>=0).all() and (F.joint_nll_gain>=0).all()
    cal=(F.max_abs_coverage_error<=.03).all()
    cohort_bad=K[(K.n>=250)&(K.gain<-.10)]
    cohort_ok=len(cohort_bad)==0
    passed=bool(core and proper and cal and cohort_ok)
    decision='REOPEN_APPROVED_PENDING_NATIVE_ARCHITECTURE_AND_OPERATIONAL_RECERTIFICATION' if passed else 'NO_REOPEN_KEEP_V14_FROZEN'

    F.to_csv(OUT/'fold_tournament.csv',index=False); B.to_csv(OUT/'date_block_bootstrap.csv',index=False); C.to_csv(OUT/'coverage.csv',index=False); K.to_csv(OUT/'competence_slices.csv',index=False); P.to_parquet(OUT/'common_fold_predictions.parquet',index=False)
    report={'experiment_id':'CBB-EXP-0003','ticket_id':'CBB-RT-003','objective_id':'CBB-OBJ-006','program_id':'CBB-PROG-016','question_id':'CBB-Q-037',
            'decision':decision,'promotion_gates':{'core_mean':bool(core),'proper_scores':bool(proper),'coverage':bool(cal),'competence_slices':bool(cohort_ok)},
            'cohort_regressions_gt_0_10':cohort_bad.to_dict('records'),'historical_fold_thresholds':FOLD_THRESH,'2026_deployment_threshold_from_exp0001':7,
            'factor_spec_from_exp0002':'L7_ORB_DRB','2026_outcomes_used':False,'production_model_changed':False,
            'fold_metrics':F.to_dict('records'),'bootstrap':B.to_dict('records'),'coverage':C.to_dict('records'),'competence_slices':K.to_dict('records')}
    (OUT/'report.json').write_text(json.dumps(report,indent=2)); print(json.dumps(report,indent=2))

if __name__=='__main__': main()
