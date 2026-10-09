from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

import prediction_first_15_gates as m


def pace_test(ctx, schedf, r2, pace):
    """Use the actual G10 pace artifact schema: one row per game.

    G10 pace_predictions.parquet columns are season,date,game_id,pace_pred,pace_70,pace_actual.
    This patch changes only artifact plumbing. The preregistered T17.03 features,
    chronological folds, acceptance hurdle and model class are unchanged.
    """
    required = {'game_id', 'pace_pred', 'pace_actual'}
    missing = required.difference(pace.columns)
    if missing:
        raise KeyError(f'missing G10 pace columns {sorted(missing)}; have {list(pace.columns)}')

    pg = pace[['game_id', 'pace_pred', 'pace_actual']].copy()
    pg['game_id'] = pg['game_id'].astype(str)
    pg = pg.drop_duplicates('game_id')

    x = (
        ctx[['game_id', 'season']]
        .drop_duplicates()
        .merge(pg, on='game_id', how='inner', validate='one_to_one')
        .merge(schedf, on='game_id', how='inner', validate='one_to_one')
    )
    x['pace_resid'] = x.pace_actual - x.pace_pred
    features = ['rest_diff', 'games7_diff', 'games14_diff']
    rows = []

    for sy in m.EVAL_SEASONS:
        tr = x[(x.season >= m.TRAIN_MIN_SEASON) & (x.season < sy)].dropna(
            subset=features + ['pace_resid']
        )
        te = x[x.season == sy].dropna(subset=features + ['pace_resid']).copy()
        model = LinearRegression().fit(tr[features], tr.pace_resid)
        te['pred_pace_rest'] = te.pace_pred + model.predict(te[features])
        base = float(np.mean(np.abs(te.pace_actual - te.pace_pred)))
        new = float(np.mean(np.abs(te.pace_actual - te.pred_pace_rest)))
        rows.append({
            'season': sy,
            'n': len(te),
            'base_pace_mae': base,
            'rest_pace_mae': new,
            'mae_gain': base - new,
            'coef_rest_diff': float(model.coef_[0]),
            'coef_games7_diff': float(model.coef_[1]),
            'coef_games14_diff': float(model.coef_[2]),
        })

    d = pd.DataFrame(rows)
    gain = float(d.mae_gain.mean())
    wins = int((d.mae_gain > 0).sum())
    return d, {
        'mean_mae_gain': gain,
        'season_wins': wins,
        'pass': bool(gain >= 0.03 and wins >= 2),
        'joined_games': int(len(x)),
    }


m.pace_test = pace_test
m.main()
