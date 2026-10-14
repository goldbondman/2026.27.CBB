from __future__ import annotations

import pandas as pd
import cbb_exp_0003_formal_reopen_tournament as exp

# Engineering-only identity repair for CBB-EXP-0003.
# The preregistered hypotheses, historical fold thresholds, factor specification,
# metrics, bootstrap, probability/distribution tests and promotion rules are unchanged.
#
# T35/T37's certified ESPN adapter rewrites its internal feature-table `game_id`
# to the ESPN game identity. The immutable v14 artifact retains NCAA `game_id`
# and stores the matching ESPN key separately in `espn_game_id`. The first EXP-0003
# attempt joined NCAA game_id to ESPN game_id and therefore produced zero fitted rows.
# This adapter joins frozen.espn_game_id -> feature.game_id while retaining the
# original NCAA key in `ncaa_game_id` for auditability.

_prior_merge = pd.DataFrame.merge


def _identity_repaired_merge(self, right, *args, **kwargs):
    on = kwargs.get('on', None)
    right_cols = set(getattr(right, 'columns', []))
    left_cols = set(getattr(self, 'columns', []))
    if (
        on == ['season', 'game_id']
        and 'espn_game_id' in left_cols
        and 'pred_margin' in left_cols
        and 'h_game_no' in right_cols
        and 'd_adj_l7_margin' in right_cols
    ):
        left = self.copy()
        left['ncaa_game_id'] = left['game_id'].astype(str)
        left['game_id'] = pd.to_numeric(left['espn_game_id'], errors='coerce').astype('Int64').astype(str)
        return _prior_merge(left, right, *args, **kwargs)
    return _prior_merge(self, right, *args, **kwargs)


pd.DataFrame.merge = _identity_repaired_merge

if __name__ == '__main__':
    exp.main()
