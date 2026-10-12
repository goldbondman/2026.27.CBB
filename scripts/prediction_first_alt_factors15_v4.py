from __future__ import annotations
import pandas as pd

# Apply the already-certified T28 identity adapter to the v2 build merge without
# changing any T34 hypotheses, features, thresholds, or chronological folds.
_orig_merge = pd.DataFrame.merge

def _merge(self, right, *args, **kwargs):
    on = kwargs.get('on', None)
    if on == ['season','game_id'] and 'espn_game_id' in self.columns and 'factor_actual_home' in getattr(right,'columns',[]):
        left = self.copy()
        left['game_id'] = pd.to_numeric(left['espn_game_id'], errors='coerce').astype('Int64')
        return _orig_merge(left, right, *args, **kwargs)
    return _orig_merge(self, right, *args, **kwargs)

pd.DataFrame.merge = _merge
import prediction_first_alt_factors15_v3 as v3

if __name__=='__main__':
    v3.main()
