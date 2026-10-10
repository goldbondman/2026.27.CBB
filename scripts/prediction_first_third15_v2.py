from __future__ import annotations

import prediction_first_third15 as m

_orig_load=m.load_all

def load_all_repaired():
    gp,r2,pace,cf,venue,sched=_orig_load()
    # G10 stores one pace row per game as pace_pred/pace_actual. Expand it to the
    # two frozen R2 team rows so the downstream score reconstruction remains
    # exactly tied to the certified production pace artifact.
    if 'pred_pace' not in pace.columns:
        p=pace[['game_id','pace_pred']].drop_duplicates('game_id').rename(columns={'pace_pred':'pred_pace'})
        pace=r2[['game_id','team']].merge(p,on='game_id',how='left',validate='many_to_one')
    return gp,r2,pace,cf,venue,sched

m.load_all=load_all_repaired
m.main()
