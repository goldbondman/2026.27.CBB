# R047: November 6 archived slate completion and incremental error learning
Date (Pacific): 2026-11-12
Classification: EXTENSION / PROSPECTIVE SCOREKEEPING / REUSED THRESHOLD DIAGNOSTIC
Record status: APPENDED_RESEARCH_ONLY_NO_PRODUCTION_CHANGE
Research lineage:
- CBB-OBJ-007 / CBB-PROG-018 / CBB-Q-041 / CBB-RT-006 (prospective scorekeeping and frozen forecast provenance)
- CBB-OBJ-006 / CBB-PROG-016 / CBB-Q-037 (champion/challenger parity and promotion block)
- CBB-OBJ-008 / CBB-PROG-019 / CBB-Q-042 (market disagreement; exploratory)
- CBB-OBJ-008 / CBB-PROG-021 / CBB-Q-045 (postgame error taxonomy)
- Data quality protection CBB-OBJ-001.
Experiment ID: CBB-EXP-R047-20261112-001 (new append-only record; not attributed retrospectively to past research).
**No new model trained, selected, or promoted.** Frozen rows not modified. No 2026 outcome used in pregame feature design.

## Evidence / sources
Primary project spreadsheet: https://docs.google.com/spreadsheets/d/1a9FPD_oVL-ozVagzpuK1J_iQj8ssziIdCtl6JC0EUE8/edit
Input archived scorecard: NOV06_FULL58_SCORECARD (39 already-graded final rows; 19 pending at last snapshot).
Input frozen R022 challenger: R022_PROSPECTIVE_L4_FROZEN (55 distinct matching IDs, frozen 2026-11-06T22:43:54.641Z before tipoff).
Recovered additional final-score source: https://www.sharpdataos.com/cbb/scores?date=2026-11-06&season=2026
Secondary cross-check: https://statsnext.com/college-basketball/games/2026-11
Outputs (new append-only spreadsheet tabs):
- R047_NOV06_58_FINALS (58 graded forecasts, exact margin/total residuals, odds differences and source grades)
- R047_R022_PAIRED (55 same-game R022 lambda4 vs archived baseline outcome comparisons)
- R047_FINDINGS_GATES (source counts, findings, confidence and next gates).

The 19 newly recovered outcomes are from the **November 6 slate**, not games played since November 6. The synced master remains stale for November 7–11; no claim is made about newly completed post-November 6 games. New score observations are secondary/provider results, not independently school-certified.

## Dataset status / freshness gate
Workbook originally recorded 2024-25: 6,031 source-validated games; 2025-26: 6,082; 2026-27: 302 through November 6. D1 identity and point-in-time feature vintage are not fully certified. UPDATE_LOG and SYNC_CONTROL do not establish a successful unattended post-November 6 ingest. Repo main most recent observed commit prior to R047: 05a8060c655874f9907542e9f915c362385e4944 (2026-11-07T03:51:45Z). Do not confuse recent scores with refreshed team box/advanced stats or with prospectively scored sealed v14.

## Archived R014/R018 market-blind research forecasts: full 58
- Margin MAE 13.6939, total MAE 13.3066.
- Projected-favorite oriented margin residual (realized favorite differential minus forecasted favorite differential): **+8.2569** points. Simple game-iid bootstrap 95% CI [+4.15,+12.54]; one slate and overlapping team/program structure, not multi-date replicated evidence.
- Actual-total minus model-total mean +2.8117 points, bootstrap CI [-1.25,+7.09]; directional totals shift not established.
- 28 / 58 games with absolute margin miss >=10, 22 / 58 with absolute total miss >=15.
- Previous 39 finished games: margin MAE 12.3502, total MAE 12.6738, favorite residual +7.6052.
- New 19 finals: margin MAE 16.4520, total MAE 14.6055, favorite residual +9.5947.
- Favorite margin cohorts: 0-5: n13/+6.745; 5-10: n14/+12.780; 10-15: n10/+5.707; 15-20: n9/+3.225; 20-30: n11/+11.989; 30+: n1/-5.661. Insufficient 30+ sample.
- Largest examples: Detroit Mercy at Marquette model +9.82 actual +61, Queens at Oklahoma State model +9.45 actual +50, UAPB at Baylor +20.89 actual +56, Southern at TCU +20.59 actual +52. Check for matchup/identity/venue/forecast engineering failures before drawing structural causal conclusions.

## R022 lambda4 vs archive baseline: all 55 frozen matched games
- Margin MAE lambda4 12.8766 vs original baseline 13.4924; improvement **+0.6158** points, 39 wins / 16 losses, simple game-level bootstrap 95% CI [+0.0766,+1.1384].
- Total MAE lambda4 13.4579 vs original baseline 13.4841; improvement +0.0262 points, CI [-0.1057,+0.1558].
- 10 games with archived baseline expected abs margin >=20: lambda4 MAE 12.8822 vs archive 13.9305.
- Research baseline is **not** the immutable production v14. This is one date/small cohort, 2025 reused history already examined. No promotion.

## Edge threshold exploration, 58 archived forecasts, same recorded market snapshots
Assume flat stake and hypothetical -110 pricing. Market lines are pregame indicative (not verified executable prices nor certified closing line). Outcomes graded after the forecasts were frozen, but edge thresholds inspected repeatedly; strictly exploratory:
- abs edge >=2: 53 games, model 25-28, ROI -9.95%
- >=3: 48 games, 21-27, ROI -16.48%
- >=5: 42 games, 17-25, ROI -22.73%
- >=8: 26 games, 9-17, ROI -33.92%
- >=12: 16 games, 3-13, ROI -64.20%
- 2 <= edge < 12: 37 games, 22-15, ROI +13.51%.
- Market underdog baseline on the same 37 games: 23-14, ROI +18.67%.
- Market underdog baseline on same >=12 games: 4-12, ROI -52.27%.
Do not claim any statistically validated or deployable strategy. Failed 12+ disagreement more consistent with systematic mistrust of large model-market gaps, not evidence that larger gaps are better.

## Prior historical research (not newly retested)
- R024 elite-weak November 2023–2025 historical 240-game cohort: forecast 20.24, actual 27.90, stronger-team residual +7.66; historical reused and Division I season membership/proxies provisional.
- R045 high priority: restore immutable champion v14 and feature wiring; independently test separately regularized margin/total, total baseline intercept, strength-conditioned pace, offense/defense priors and D1/PIT certification.
- PARALLEL_SCORECARD: generic Four Factors had no reliable incremental gain, player skill signal small and roster prediction not validated, pace factor limited, champion unrecovered.

## Reproducibility and integrity
- Source forecast quantities are pre-tip records; observed outcomes appended on a new grading sheet. Market attributes were read only after forecasts were frozen.
- Mean absolute errors recomputed from each paired game score. Favorite residual defined sign(model home margin) * actual home margin - abs(model home margin).
- Bootstrap intervals reflect resampling of individual games with a fixed seed, not date-clustered or hierarchical uncertainty. One game date offers no multi-slate inference.
- All 55 R022 IDs matched archived baseline forecast IDs; all 58 R014/R018 IDs got final scores. Existing input sheets/forecasts left unchanged.
- No production code/weights/config touched; no champion parity assertion; old historical holdout not described as untouched. No new valid 2026-27 post-Nov6 data backfilled.

## Decision / next gates
1. DATA QA: get actual Nov07–Nov11 result/box-score/advanced-stat ingestion into original source master with official DI membership, timestamps and duplicate checks. Verify automatic run really writes and append UPDATE_LOG; do not overwrite historical scores.
2. GOVERNANCE/CHAMPION: recover exact sealed v14 executable, full config, hashes, and prediction file; re-score frozen champion and challengers on *identical* games.
3. MODEL: pre-register a margin-specific weaker-shrink shadow, independent offense/defense priors and dynamic early-season uncertainty on a later unseen multi-date cohort. Do not choose parameters on R047 games.
4. MARKET: keep 2–under12 and 12+ as locked hypotheses for later unseen bets; compare to same-games simple underdog, use actual time-stamped prices if available, and do not back-tune Stage A forecasts from outcomes.
5. Continue past Q041/CBB-Q-037 gates only with new primary-score certification, proper PIT priors, broader fresh samples and human promotion review.
