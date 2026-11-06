# R017 | Possession measurement, returning production and confidence (2026-11-06)

**Research-only, no production changes.** Linked to CBB-OBJ-001 / CBB-Q-004 (source semantics; not reopening certified legacy question), CBB-OBJ-003 / CBB-Q-012 and Q-014 (rotation and roster-source legality) and CBB-OBJ-005 / CBB-Q-025 (conditional reliability; legacy state preserved). Legacy CBB-Q-005 and Q-013 are already stopped. Historical 2025-26 holdout has previously been used and is NOT untouched.

## Input provenance

Downloaded **actual** ZSTD Parquet source bytes, parsed with PyArrow in an authorized network-enabled Vercel research sandbox. Source byte hashes are in [R013 possession metrics](r013_possession_metrics_20261106.json). Parsed 2027 NCAA: 259 distinct contests, 518 paired team rows, 6,258 player rows, 38,271 NCAA source possession events. Source through 2026-11-05. ESPN seasonal rosters and player boxes for 2024, 2025 and 2026 downloaded and parsed.

The original 2027 NCAA `espn_game_id` column is entirely NULL. Other R013 project ingestion matched 165 source games by exact dual IDs/scores and additional partial candidate matches; **this R017 source-level possession comparison does not rely on ESPN cross-source game joins**.

## P1: 0.44 versus 0.475 possession estimator

| Statistic (518 same team-games) | 0.44 | 0.475 |
|---|---:|---:|
| MAE versus NCAA presegmented possessions | 1.7880 | **1.4003** |
| Signed box-estimate minus event-count bias | -1.4319 | **-0.6794** |

* Paired MAE gain of 0.475 versus 0.44: **+0.3877 possessions/team-game**, game-cluster bootstrap 95% CI **+0.3251 to +0.4479** across 259 games.
* All 259 NCAA contests have exactly two team records.
* 0 duplicate game-period-possession identifiers across 38,271 event rows.
* NCAA player points reconcile **518/518** team rows.
* Sum of possession event points reconciles **452/518** team rows; 66 mismatches, summed absolute discrepancy 100 points, max 4.
* NCAA `o_poss` equals NCAA event counts in 504/518 rows.

**Interpretation:** 0.475 more closely reproduces this *source's* presegmented possession count. This is not independent adjudication and **does not establish lower pregame total MAE**. 66 team score discrepancies need investigation. No canonical possession metric changed.

## P2: Player-aware mean forecast, fixed-structure comparison

Retrospective training on first 14 days of 2024-25 (597 paired games) and a previously explored 2025-26 sample of first 14 days (611 paired games, 14 date clusters). Previous-season opponent- and box-derived strength plus pace, Ridge baseline with same training games; add returning-minutes, scoring, field-goal volume and minimum team continuity to the otherwise equivalent Ridge challenger. All evaluation is market-blind.

| Target | Team-only MAE | Player-aware MAE | Paired gain | Date-block bootstrap 95% CI |
|---|---:|---:|---:|---|
| Home margin | 12.0557 | 12.0528 | +0.0029 | -0.0255 to +0.0260 |
| Game total | 15.3016 | 15.1608 | +0.1409 | -0.0061 to +0.2485 |

**No robust incremental mean improvement established.** Total gain CI crosses 0, margin effect trivial. This test **does not compare the historical legacy champion**, and is not an independent prospective validation.

ESPN numeric athlete IDs must be normalized from floating-point Parquet values like `12345.0` to `12345` before matching season roster IDs. Without normalization, returning minutes falsely calculate to 0%. After normalization, mean prior minutes retained within same team were **9.58%** (2024->2025) versus **28.98%** (2025->2026). The large coverage gap is a **data-quality / historical-source-vintage warning**, not a demonstrated NCAA roster trend.

## P3: Roster features for predictive reliability

Same point forecast held fixed for U0 and U1. U1 adds roster continuity to a compact absolute-error risk estimator. Error targets for training risk models were **in-sample development residuals**, so these are exploratory diagnostics, not cross-fitted or certified probability estimates.

| Metric on 611 reused 2025-26 games | U0 team-only | U1 roster-aware |
|---|---:|---:|
| Margin predicted absolute-error MAE | 7.4201 | 7.4392 |
| Margin nominal 80% interval coverage | 77.58% | 77.91% |
| Margin low-risk 60% games' actual MAE | 11.9587 | 12.0256 |
| Total predicted absolute-error MAE | 9.3092 | 9.5212 |
| Total nominal 80% interval coverage | 73.49% | 68.58% |
| Total low-risk 60% games' actual MAE | 14.3080 | 14.9858 |
| Total risk-rank Spearman versus actual absolute error | +0.1517 | -0.0153 |

These **specific** roster-aware uncertainty features do not improve error ranking and worsen totals interval coverage. They **do not establish that all roster-informed uncertainty approaches fail**; source timing and better cross-fitted risk calibration are unresolved.

## Material limitations / gates

1. Historical roster snapshots have **no verified first-published as-of timestamps**. They can contain end-season information; retrospective roster overlap is **not pregame safe**.
2. 2025-26 screen is already consumed across other research. Confidence intervals do not solve selection/holdout reuse.
3. Team-only baseline is a modest separately reconstructed Ridge model, not original frozen M1-M5 vectors or sealed v14.
4. Venue is simplified by globally applying +2.7 home advantage. Season-specific Division I eligibility not independently certified; cohort may include lower divisions.
5. U0/U1 risk target built from in-sample development residuals, not full chronological cross-fitting, and interval half-width uses an approximate Gaussian scale.
6. The source's event-derived possessions are partly source-defined rather than independent possession play-by-play recoding.
7. The live project has concurrently created R013 2027 staging and R014 pre-tip current-pace forecasts. R017 is a *distinct* diagnostic registry and intentionally does not edit those artifacts.

**Decision:** Retain current production; do not change R2, R005, R007, R014 or sealed legacy v14. Source convention favors 0.475 descriptively; player-added mean results weak; proposed U1 confidence fails this exploratory screen. No betting odds or lines accessed.

## Reproducibility

* [Possession experiment](../scripts/cbb_r013_possession_comparison.py)
* [Returning-production and confidence experiment](../scripts/cbb_r013_continuity_research.py)
* [Full possession metrics](r013_possession_metrics_20261106.json)
* [Full continuity metrics](r013_continuity_metrics_20261106.json)
* [Live workbook possession sheet](https://docs.google.com/spreadsheets/d/1a9FPD_oVL-ozVagzpuK1J_iQj8ssziIdCtl6JC0EUE8/edit#gid=20261300)
* [Live workbook mean model sheet](https://docs.google.com/spreadsheets/d/1a9FPD_oVL-ozVagzpuK1J_iQj8ssziIdCtl6JC0EUE8/edit#gid=20261301)
* [Live workbook confidence sheet](https://docs.google.com/spreadsheets/d/1a9FPD_oVL-ozVagzpuK1J_iQj8ssziIdCtl6JC0EUE8/edit#gid=20261302)
* [Live workbook decision gates](https://docs.google.com/spreadsheets/d/1a9FPD_oVL-ozVagzpuK1J_iQj8ssziIdCtl6JC0EUE8/edit#gid=20261303)

**Next high-value gate:** Prove frozen pregame roster membership / previous player minutes, repair NCAA possession event scoring discrepancies, validate season-specific D1/venue, and preregister chronological cross-fitted risk-model tests on genuinely new prospective games. Stop if missing pregame roster provenance remains material.
