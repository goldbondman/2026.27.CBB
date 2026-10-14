from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pandas as pd

import prediction_first_recency_activation15_repair as repair

# CBB-EXP-0001
# Objective: CBB-OBJ-006
# Program: CBB-PROG-014
# Question: CBB-Q-032
# Ticket: CBB-RT-001
# Work type: GOVERNANCE / ROBUSTNESS
#
# PREDECLARED SELECTION RULE (written before this experiment is executed):
# 1) Candidate minimum-game thresholds are fixed at [7, 8, 10, 12, 15].
# 2) Evaluate each threshold as a *policy* on chronological OOS 2024 and 2025:
#    active non-neutral games receive adjusted-L7; all other non-neutral games retain
#    the calibrated baseline. This avoids selecting on active-only effect size.
# 3) A candidate is ELIGIBLE only if its policy gain is >0 in both seasons and the
#    95% date-block bootstrap lower bound is >0 in both seasons.
# 4) Among eligible candidates choose the threshold with the largest WORST-SEASON
#    policy gain (maximin robustness). If candidates are within 0.005 MAE on the
#    maximin score, choose the EARLIER/smaller threshold.
# 5) The selected threshold is the frozen 2026 activation threshold. It is not
#    changed from 2026 outcomes. Reopening requires the ordinary model-reopen rules.

OUT = Path('results/cbb_exp_0001_activation_policy')
OUT.mkdir(parents=True, exist_ok=True)
CANDIDATES = [7, 8, 10, 12, 15]
TIE_TOL = 0.005
SEASONS = [2024, 2025]
SEED = 37001
N_BOOT = 1000

r = repair.r


def policy_for_threshold(q: pd.DataFrame, sy: int, th: int) -> pd.DataFrame:
    tr, te, _ = r.calibrated(q, sy)
    et = tr.is_non_neutral & tr.min_game_no.ge(th)
    ee = te.is_non_neutral & te.min_game_no.ge(th)
    _, _, active = r.fit_predict(tr, te, ['d_adj_l7_margin'], et, ee)

    z = te[te.is_non_neutral].copy()
    z['pred'] = z.cal_margin
    z['active'] = False
    if active is not None:
        ix = z.index.intersection(active.index)
        z.loc[ix, 'pred'] = active.loc[ix, 'pred']
        z.loc[ix, 'active'] = True
    return z


def evaluate(q: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for th in CANDIDATES:
        for sy in SEASONS:
            z = policy_for_threshold(q, sy, th)
            gain = r.mae(z.actual_margin, z.cal_margin) - r.mae(z.actual_margin, z.pred)
            mean_boot, lo, hi, ppos = repair.fast_date_boot(
                z, base='cal_margin', pred='pred', n=N_BOOT, seed=SEED + sy + th
            )
            rows.append({
                'threshold': th,
                'season': sy,
                'n': len(z),
                'active_n': int(z.active.sum()),
                'active_fraction': float(z.active.mean()),
                'policy_gain': float(gain),
                'bootstrap_mean_gain': float(mean_boot),
                'lo95': float(lo),
                'hi95': float(hi),
                'p_positive': float(ppos),
            })
    return pd.DataFrame(rows)


def select_policy(df: pd.DataFrame):
    summaries = []
    for th in CANDIDATES:
        z = df[df.threshold.eq(th)].sort_values('season')
        eligible = bool(len(z) == len(SEASONS) and (z.policy_gain > 0).all() and (z.lo95 > 0).all())
        summaries.append({
            'threshold': th,
            'eligible': eligible,
            'mean_policy_gain': float(z.policy_gain.mean()),
            'worst_season_gain': float(z.policy_gain.min()),
            'min_lo95': float(z.lo95.min()),
            'mean_active_fraction': float(z.active_fraction.mean()),
        })
    s = pd.DataFrame(summaries)
    elig = s[s.eligible].copy()
    if elig.empty:
        return None, s
    best_worst = float(elig.worst_season_gain.max())
    near = elig[elig.worst_season_gain.ge(best_worst - TIE_TOL)].sort_values('threshold')
    selected = int(near.iloc[0].threshold)
    return selected, s


def main():
    q = repair.repaired_prep()
    # Integrity assertion: development substrate contains no 2026 outcomes.
    seasons = sorted(int(x) for x in pd.Series(q.season).dropna().unique())
    assert max(seasons) <= 2025, seasons

    results = evaluate(q)
    selected, summary = select_policy(results)
    results.to_csv(OUT / 'candidate_fold_results.csv', index=False)
    summary.to_csv(OUT / 'candidate_summary.csv', index=False)

    rule = {
        'experiment_id': 'CBB-EXP-0001',
        'ticket_id': 'CBB-RT-001',
        'objective_id': 'CBB-OBJ-006',
        'program_id': 'CBB-PROG-014',
        'question_id': 'CBB-Q-032',
        'as_of': '2026-10-14',
        'candidate_thresholds': CANDIDATES,
        'selection_rule': 'eligible iff policy_gain>0 and date-block lo95>0 in 2024 and 2025; choose maximin worst-season policy gain; if within 0.005 choose smaller threshold',
        'selected_threshold': selected,
        'status': 'PASS_FROZEN' if selected is not None else 'HOLD_NO_ELIGIBLE_THRESHOLD',
        '2026_outcomes_used': False,
        'production_model_changed': False,
        'reopen_condition': 'ordinary frozen-v14 reopen governance only',
    }
    canonical = json.dumps(rule, sort_keys=True, separators=(',', ':')).encode()
    rule['config_sha256'] = hashlib.sha256(canonical).hexdigest()
    (OUT / 'frozen_activation_policy.json').write_text(json.dumps(rule, indent=2))

    report = {
        'experiment': rule,
        'candidate_summary': summary.to_dict('records'),
        'fold_results': results.to_dict('records'),
    }
    (OUT / 'report.json').write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == '__main__':
    main()
