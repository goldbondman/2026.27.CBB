from __future__ import annotations

import pandas as pd
import prediction_first_fourth15 as m

_orig = m.old_vs_champion

def old_vs_champion_fixed(gp, r2, pace, venue, sched, pm):
    # The site ledger intentionally leaves neutral/site fields null for unmatched
    # historical games. Convert those to an explicit False eligibility flag so
    # they are quarantined instead of causing a pandas boolean-mask failure.
    all_ids = pd.DataFrame({'game_id': gp['game_id'].astype(str).unique()})
    v = all_ids.merge(venue.copy(), on='game_id', how='left')
    v['neutral_known'] = v['neutral_known'].fillna(False).astype(bool)
    return _orig(gp, r2, pace, v, sched, pm)

m.old_vs_champion = old_vs_champion_fixed
m.main()
