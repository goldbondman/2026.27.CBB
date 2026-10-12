import prediction_first_recency_activation15 as r

# Engineering-only namespace/date adapter. T37 hypotheses, folds, thresholds and gate rules are unchanged.
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

r.prep = repaired_prep

if __name__ == '__main__':
    r.main()
