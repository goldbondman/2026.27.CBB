# R031 | First-two-weeks matchup-pace robustness and total-score downstream evaluation

**Completed:** 2026-11-06 / Pacific. **Classification:** ROBUSTNESS/EXTENSION, CBB-EXP-0004, CBB-OBJ-003 → CBB-PROG-007 → CBB-Q-015. The original CBB-Q-008 certified pregame pace research remains CLOSED. R030 in the workbook is the distinct elite scoring diagnostic, not this research. No original game forecasts, production settings, or market inputs changed.

## First: archived OS evidence, don't repeat
- **R027** source-derived actual-pace oracle explains **~1.03 of 6.80** points of top50/bottom100 stronger-side margin bias, leaving ~5.77 unexplained. The oracle is postgame and never usable as a predictor.
- **R028** top50/bottom100 n163 historical: expected stronger-side score lower than actual by **6.789 points**, expected weaker-side score error near 0. The source does not prove a specific causal factor. Historical favorite-only correction and variants have been tried.
- **R029** prospective-looking shadow is **quarantined** because market exposure occurred before its freeze. Don't promote it or present as blind. **R022 remains separately frozen before such exposure**, with a 55-game independent challenger awaiting verified results.

## New, independently executed R031 pace robustness

Input sources: sportsdataverse ESPN hoopR NCAA team box-score Parquet 2024, 2025, 2026, fetched and parsed on sandbox. SHA256 of source files: [metrics JSON](r030_early_pace_robustness_metrics_20261106.json). Source hashes are pinned in the research run results. No live market data retrieved.

Development: first 14 days of 2024-25, 597 same-game home/away pairs. Previous 2023-24 team raw mean box possessions P475 as pregame pace, minimum 15 previous-year games. Test on first 14 days of 2025-26, 607 matched games, using 2024-25 priors. Historical evaluation is **REUSED/EXPLORATORY**, not untouched out-of-sample certification. The original champion pace formula has opponent adjustment and current-season partial updates; **our comparison is a stripped-down previous-year, team-pace-only baseline**, not exact R014/v14 parity. No roster PIT, D1 or complete OT certification.

Box possessions = FGA − ORB + TOV + 0.475×FTA. Actual game possessions = mean of two box estimates. The game total test uses one identical previous-year offense/defense adjustment for each pace method:
`expected_total = pace/100 × (home_prior_off + away_prior_def − league_off + away_prior_off + home_prior_def − league_off)`.
Neither model uses pregame market information. No season-specific source after tipoff is allowed into features.

### Screen performance, 607 historical test games

| Pre-game pace method | Pace MAE | Predicted minus actual pace bias | Same-game total MAE |
|---|---:|---:|---:|
| Equal mean of teams' previous-season paces | 4.7593 | −2.6965 | 15.7110 |
| Slower team 60% | 4.8576 | −2.9622 | 15.8124 |
| Slower team 70% | 4.9672 | −3.2279 | 15.9492 |
| Faster team 60% | 4.6741 | −2.4308 | 15.6398 |
| Faster team 70% | 4.6112 | −2.1652 | 15.5922 |
| Geometric mean | 4.7666 | −2.7168 | 15.7176 |
| Previous-cohort-trained constant (+1.432) | **4.4548** | −1.2645 | 15.4390 |
| Prior-trained intercept and pace-gap slope | 4.4815 | −1.2419 | **15.4277** |
| Prior-trained gap slope without intercept | 4.9908 | −3.2818 | 15.9800 |

Development 2024-25: observed mean 70.6955 vs projected 69.2635. Test 2025-26: observed 71.4955 vs projected 68.7990. Train-only learned constant **+1.4320**. Joint train-only regression intercept **+2.0399** and slope **−0.2203 per team pace gap**. Slow-team control conjecture is **not proven**: extra slow weight worsened both metrics; a gap-only slope worsened results, and a train-only constant was more successful than nonlinear interaction for pace. Confounders include changes in game composition, opponent quality, and OT.

### Matched test-game day-cluster bootstrap

Paired MAE gain is equal-mean absolute error minus challenger absolute error; positive favors challenger. 1,500 resamples across **14 dates**.

| Challenger | Pace MAE gain [95% CI] | Game total MAE gain [95% CI] |
|---|---|---|
| 70% faster-team weight | +0.1481 [0.0930, 0.1925] | +0.1188 [0.0380, 0.1817] |
| Prior-trained constant | **+0.3045 [0.1702, 0.4124]** | +0.2720 [0.0184, 0.4727] |
| Prior-trained gap plus intercept | +0.2778 [0.1354, 0.3887] | +0.2833 [−0.0293, 0.5349] |
| 60% slower-team | −0.0983 [−0.1255, −0.0670] | −0.1013 [−0.1324, −0.0578] |

CIs are not multiple-comparisons-adjusted, and only 14 independent day clusters constrain precision. Prior-season records include NCAA teams beyond independently verified D1, and box possessions are estimates, with no independent regulatory-length filter; subject to source definition/OT bias. No claim of clean causal effect or significant prospective edge.

## New questions and test decisions

1. **What causes the opening-season pace baseline shift?** Is it systematic season environment drift, non-D1 opponent mixture, 40-min vs OT measurement, actual team pace, or rate features? **Research priority 1.** Replicate on D1-only regulation games with source-verified opponents and team-season priors.
2. **Does faster/slower pace weight add signal after calibration?** Compare previous-cohort intercept-only against intercept+pace-gap plus opponent-adjusted and home/away lineup effects, nested folds. Current gap-only variant failed. **Research priority 2.**
3. **Does corrected pace consistently improve *team score*, *margin*, and *total*?** Use identical reigning v14/R014 opponent-adjusted scoring factors, no raw-only extrapolation. **Research priority 3.**
4. **Can past coach/player changes predict pace volatility?** Keep separately source-PIT-blocked under Q014; no guesswork.
5. **Can extreme mismatch underprediction be corrected via higher elite offensive efficiency vs weaker-opponent defensive suppression?** Existing R028 historical residual diagnostic suggests stronger-side under-scoring, but R022 needs complete unexposed prospective scoring; do not tune on Nov6 games.

### Decisions
- **Pace equal mean:** KEEP BASELINE pending independent fresh checks.
- **Simple slower-team control:** REJECT AS UNIVERSAL DEFAULT from this screen; still possible conditional effects.
- **Historical constant-offset or faster weighting:** CONTINUE TESTING, **not promote**. In a simplified model, both improve early-season MAE but risk generalization.
- **R022 strict pretip challenger:** Score eligible independently verified outcomes when source status is FINAL; no modification.
- **R029:** QUARANTINED and excluded from clean prospective confirmatory testing.

**No changes to production or immutable frozen predictions. No market research or odds in the empirical test.**

### Verified artifacts
- [Source code](../scripts/cbb_r030_early_pace_robustness.py)
- [Complete metrics, paired CIs and Parquet hashes](r030_early_pace_robustness_metrics_20261106.json)
- [R031 matched-game replay table](https://docs.google.com/spreadsheets/d/1a9FPD_oVL-ozVagzpuK1J_iQj8ssziIdCtl6JC0EUE8/edit#gid=20262700)
- [R031 decision gates](https://docs.google.com/spreadsheets/d/1a9FPD_oVL-ozVagzpuK1J_iQj8ssziIdCtl6JC0EUE8/edit#gid=20262701)
