from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.stats import norm
from sklearn.linear_model import Ridge
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

import prediction_first_recency_activation15_repair as repair

# CBB-EXP-0002
# Objective: CBB-OBJ-006
# Program: CBB-PROG-013
# Question: CBB-Q-029
# Ticket: CBB-RT-002
# Work type: ABLATION / ROBUSTNESS
#
# PREDECLARED DESIGN:
# - Use the original leakage-safe T37 fold thresholds (2024=12, 2025=15), each
#   selected from prior data only. Do NOT retroactively impose the 2026 threshold=7
#   on historical selection folds.
# - Same eligible cohort for every spec: non-neutral, min game number >= fold threshold,
#   complete d_adj_l7_margin / ORB / DRB values.
# - Compare L7, L7+ORB, L7+DRB, L7+ORB+DRB with identical Ridge alpha=30.
# - An add-on may enter the final challenger only if vs L7 it improves margin MAE in
#   both future folds AND the 95% date-block bootstrap lower bound is >0 in both folds.
# - Proper-score check is secondary: training-only sigma scale is selected separately
#   for each spec; Brier and logloss may not be worse in either future fold.
# - If no add-on clears all rules, the final factor spec is L7_ONLY.

OUT=Path('results/cbb_exp_0002_clean_rebound')
OUT.mkdir(parents=True,exist_ok=True)
THRESHOLDS={2024:12,2025:15}
SPECS={
    'L7':['d_adj_l7_margin'],
    'L7_ORB':['d_adj_l7_margin','d_pre_orbr'],
    'L7_DRB':['d_adj_l7_margin','d_pre_drbr'],
    'L7_ORB_DRB':['d_adj_l7_margin','d_pre_orbr','d_pre_drbr'],
}
SEED=29002
N_BOOT=1000
r=repair.r


def mae(y,p): return float(np.mean(np.abs(np.asarray(y)-np.asarray(p))))
def brier(y,p): return float(np.mean((np.asarray(y)-np.asarray(p))**2))
def logloss(y,p):
    p=np.clip(np.asarray(p),1e-6,1-1e-6); y=np.asarray(y)
    return float(np.mean(-(y*np.log(p)+(1-y)*np.log(1-p))))


def fast_pair_boot(df, base_col, cand_col, n=N_BOOT, seed=SEED):
    x=df.dropna(subset=['game_date','actual_margin',base_col,cand_col]).copy()
    x['gain']=np.abs(x.actual_margin-x[base_col])-np.abs(x.actual_margin-x[cand_col])
    g=x.groupby('game_date')['gain'].agg(['sum','count']).reset_index(drop=True)
    sums=g['sum'].to_numpy(float); counts=g['count'].to_numpy(float); k=len(g)
    rng=np.random.default_rng(seed); vals=[]
    for _ in range(n):
        idx=rng.integers(0,k,size=k); vals.append(float(sums[idx].sum()/counts[idx].sum()))
    a=np.asarray(vals)
    return float(a.mean()),float(np.quantile(a,.025)),float(np.quantile(a,.975)),float(np.mean(a>0))


def fit_specs(q,sy):
    tr,te,_=r.calibrated(q,sy); th=THRESHOLDS[sy]
    all_feats=['d_adj_l7_margin','d_pre_orbr','d_pre_drbr']
    trc=tr[(tr.is_non_neutral)&tr.min_game_no.ge(th)].dropna(subset=['cal_resid']+all_feats).copy()
    tec=te[(te.is_non_neutral)&te.min_game_no.ge(th)].dropna(subset=['cal_resid','game_date']+all_feats).copy()
    out={'train':trc,'test':tec,'models':{},'train_pred':{},'test_pred':{},'sigmas':{},'scales':{}}
    scales=np.linspace(.65,1.20,56)
    for name,feats in SPECS.items():
        m=make_pipeline(StandardScaler(),Ridge(alpha=30)).fit(trc[feats],trc.cal_resid)
        trpred=trc.cal_margin+m.predict(trc[feats]); tepred=tec.cal_margin+m.predict(tec[feats])
        resid=trc.actual_margin-trpred; sig=float(np.std(resid,ddof=1)); ytr=(trc.actual_margin>0).astype(int).to_numpy()
        best=None
        for sc in scales:
            br=brier(ytr,norm.cdf(trpred/(sig*sc)))
            if best is None or br<best[0]: best=(br,float(sc))
        out['models'][name]=m; out['train_pred'][name]=trpred; out['test_pred'][name]=tepred; out['sigmas'][name]=sig; out['scales'][name]=best[1]
    return out


def main():
    q=repair.repaired_prep(); seasons=sorted(int(x) for x in pd.Series(q.season).dropna().unique()); assert max(seasons)<=2025,seasons
    fold_rows=[]; boot_rows=[]; prob_rows=[]
    for sy in [2024,2025]:
        z=fit_specs(q,sy); te=z['test'].copy(); y=(te.actual_margin>0).astype(int).to_numpy()
        for name in SPECS:
            te[f'pred_{name}']=z['test_pred'][name]
            pred=z['test_pred'][name]; sig=z['sigmas'][name]; sc=z['scales'][name]
            prob=norm.cdf(pred/(sig*sc))
            fold_rows.append({'season':sy,'spec':name,'n':len(te),'mae':mae(te.actual_margin,pred),'sigma':sig,'sigma_scale':sc})
            prob_rows.append({'season':sy,'spec':name,'brier':brier(y,prob),'logloss':logloss(y,prob)})
        base=te['pred_L7']
        for name in ['L7_ORB','L7_DRB','L7_ORB_DRB']:
            gain=mae(te.actual_margin,base)-mae(te.actual_margin,te[f'pred_{name}'])
            bm,lo,hi,pp=fast_pair_boot(te,'pred_L7',f'pred_{name}',seed=SEED+sy+len(name))
            boot_rows.append({'season':sy,'spec':name,'mae_gain_vs_L7':gain,'bootstrap_mean_gain':bm,'lo95':lo,'hi95':hi,'p_positive':pp})

    folds=pd.DataFrame(fold_rows); boots=pd.DataFrame(boot_rows); probs=pd.DataFrame(prob_rows)
    folds.to_csv(OUT/'fold_metrics.csv',index=False); boots.to_csv(OUT/'bootstrap_vs_l7.csv',index=False); probs.to_csv(OUT/'proper_scores.csv',index=False)

    summary=[]
    for name in ['L7_ORB','L7_DRB','L7_ORB_DRB']:
        b=boots[boots.spec.eq(name)].sort_values('season')
        p=probs[probs.spec.eq(name)].merge(probs[probs.spec.eq('L7')][['season','brier','logloss']],on='season',suffixes=('_cand','_base'))
        p['brier_gain']=p.brier_base-p.brier_cand; p['logloss_gain']=p.logloss_base-p.logloss_cand
        robust=bool(len(b)==2 and (b.mae_gain_vs_L7>0).all() and (b.lo95>0).all())
        proper=bool(len(p)==2 and (p.brier_gain>=0).all() and (p.logloss_gain>=0).all())
        summary.append({'spec':name,'robust_margin_pass':robust,'proper_score_pass':proper,'eligible':robust and proper,
                        'mean_mae_gain':float(b.mae_gain_vs_L7.mean()),'worst_mae_gain':float(b.mae_gain_vs_L7.min()),
                        'min_lo95':float(b.lo95.min()),'min_brier_gain':float(p.brier_gain.min()),'min_logloss_gain':float(p.logloss_gain.min())})
    s=pd.DataFrame(summary); s.to_csv(OUT/'selection_summary.csv',index=False)
    elig=s[s.eligible].copy()
    if elig.empty:
        selected='L7_ONLY'
    else:
        # strongest worst-season gain; ties within .002 prefer fewer added factors
        best=float(elig.worst_mae_gain.max()); near=elig[elig.worst_mae_gain.ge(best-.002)].copy()
        complexity={'L7_ORB':1,'L7_DRB':1,'L7_ORB_DRB':2}; near['complexity']=near.spec.map(complexity); selected=str(near.sort_values(['complexity','spec']).iloc[0].spec)

    report={'experiment_id':'CBB-EXP-0002','ticket_id':'CBB-RT-002','objective_id':'CBB-OBJ-006','program_id':'CBB-PROG-013','question_id':'CBB-Q-029',
            'historical_fold_thresholds':THRESHOLDS,'selected_factor_spec':selected,'status':'PASS_COMPLETE','2026_outcomes_used':False,'production_model_changed':False,
            'selection_summary':s.to_dict('records'),'fold_metrics':folds.to_dict('records'),'bootstrap':boots.to_dict('records'),'proper_scores':probs.to_dict('records')}
    (OUT/'report.json').write_text(json.dumps(report,indent=2)); print(json.dumps(report,indent=2))

if __name__=='__main__': main()
