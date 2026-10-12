import prediction_first_recency_activation15 as r

# Engineering-only namespace adapter. T37 hypotheses, folds, thresholds and gate rules are unchanged.
r.c.b = r.c.t35.b
r.c.add_model_adjusted_form = r.c.t35.add_model_adjusted_form
r.c.calibrated_baseline = r.c.t35.calibrated_baseline

if __name__ == '__main__':
    r.main()
