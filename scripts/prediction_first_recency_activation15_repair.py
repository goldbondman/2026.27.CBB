import numpy as np
import prediction_first_recency_activation15 as r

# Engineering-only namespace/date/bootstrap adapters. Scientific hypotheses, folds,
# threshold candidates, resample count and gate rules are unchanged.
r.c.b = r.c.t35.b
r.c.add_model_adjusted_form = r.c.t35.add_model_adjusted_form
r.c.calibrated_baseline = r.c.t35.calibrated_baseline

def repaired_prep():
    a, q = r.c.prep()
    q['win_y'] = (q.actual_margin > 0).astype(int)
    q['min_game_no'] = q[['h_game_no','a_game_no']].min(axis=1)
    q['max_game_no'] = q[['h_game_no','a_game_no']].max(axis=1)
    q['is_non_neutral'] = ~q.neutral.astype(bool)
    return q

# Mathematically identical date-block resampling to the original T37 implementation,
# but uses each date's sum/count sufficient statistics instead of repeatedly concatenating
# full DataFrames for every bootstrap draw.
def fast_date_boot(df, base='cal_margin', pred='pred', n=1000, seed=37):
    x = df.dropna(subset=['game_date',base,pred,'actual_margin']).copy()
    x['gain'] = np.abs(x.actual_margin-x[base]) - np.abs(x.actual_margin-x[pred])
    g = x.groupby('game_date')['gain'].agg(['sum','count']).reset_index(drop=True)
    sums = g['sum'].to_numpy(float); counts = g['count'].to_numpy(float)
    rng = np.random.default_rng(seed); vals=[]; k=len(g)
    for _ in range(n):
        idx=rng.integers(0,k,size=k)
        vals.append(float(sums[idx].sum()/counts[idx].sum()))
    a=np.asarray(vals)
    return float(a.mean()),float(np.quantile(a,.025)),float(np.quantile(a,.975)),float(np.mean(a>0))

r.prep = repaired_prep
r.date_boot = fast_date_boot

if __name__ == '__main__':
    r.main()
