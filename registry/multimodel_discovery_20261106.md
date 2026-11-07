# Multi-model Discovery Lab | Initial execution, 2026-11-06

**Status:** HISTORICAL_EXPLORATORY / PRODUCTION_UNCHANGED. Four distinct executable models, **fifth player/roster model explicitly blocked** pending timestamp-verified historical rosters. This is a **research screening experiment**, not production model certification, not profitable-betting evidence, and not a new immutable prospective forecast.

## Executive finding

A meaningful, per-game multi-model comparison was produced and saved on **607 same-game early 2025-26 matchups**, with **597 first-14-day 2024-25 training games**. All four executable models use pregame inputs, including previous-season team-game data; the dynamic Elo model updates current-season state **only from prior calendar days**. This 2025-26 period has been repeatedly examined in the Research OS and cannot be described as a clean untouched holdout.

The models use distinct calculations and disagree on the winning team in **24.2% of games**. The mean range of projected margins is **11.26 points**. Substantial disagreement is also caused by poor calibration in some models. Model independence is **architectural, not statistical**: all four share basketball source data and correlated outcome targets.

| Model | Computation | N | Margin MAE | Total MAE | Team-score MAE | Winner accuracy | Decision |
|---|---|---:|---:|---:|---:|---:|---|
| M1 team strength | Previous-season raw team offense/defense per-possession with mean matchup pace | 607 | 13.3664 | 15.7110 | 10.4888 | 72.0% | Keep as *lightweight* comparator, NOT sealed v14 |
| M2 players/roster | Hierarchical player minutes, skill, transfers and replacements | 0 | — | — | — | — | BLOCKED: roster as-of |
| M3 generative possession | Geometric pace × expected turnover/shot/rebound/free-throw scoring chain | 607 | 13.8863 | 15.5589 | 10.6102 | 76.3% | Fail margin vs M1; improve opponent factor calibration |
| M4 dynamic/situational | Pregame Elo updates + development-trained margin/total Ridge; rest variable | 607 | **11.9445** | **15.1859** | **9.7477** | **80.1%** | Continue testing; NOT v14 head-to-head |
| M5 nonlinear score-first | Independent home and away depth-2 boosted regressors on prior strength/style | 607 | 12.4881 | 15.5123 | 9.9803 | 77.9% | Continue testing only |

M1's home-margin signed bias is **−7.54 points**, M3's **−8.80**, M4 **−0.57**, M5 **−0.91**. This indicates basic rate-mixing calibration is a larger concern than just model diversity. The observed M4 advantage over M1 does **not** imply an advantage over established **sealed v14**; the original champion is stronger than this simplistic M1 comparator.

Paired **14-date clustered** *descriptive* intervals for margin MAE gain versus M1: M3 −0.5199 [−0.9311, −0.0231], M4 +1.4218 [+0.8092, +2.0156], M5 +0.8783 [+0.2853, +1.4697]. These are not adjusted for multiple analyses, are not team-clustered, and are not confirmatory.

## Four Factors v1/v2/v3 screen

Historical matched 607 games; next-game predictions only, using prior-season input factors:

| Specification | Margin MAE | Evidence interpretation |
|---|---:|---|
| FF1: fixed relative 40% eFG / 25% turnovers / 20% ORB / 15% FT scoring index, standardized and train-calibrated | 12.0665 | Baseline *descriptive index*, not physical event-score weights |
| FF2: separately learned regularized Four Factor coefficients | 11.9900 | +0.0765 MAE gain, tiny, exploratory |
| FF3: FF2 + prior strength/pace + shooting-volume/2P/3P/FT splits | 11.9622 | +0.0279 gain vs FF2, could reflect extra structural strength not shot components |

FF3's baseline is *not otherwise identical*: it also includes prior strength and pace. Therefore **do not attribute its small gain to new shooting variables** without matching feature-control ablations. FF2 weights should not be assumed positive, stable, or summing to 100%. A **true Four Factors 3.0** as a genuinely new, independently validated construct remains proposed, NOT completed.

## Situational evidence

M4 adding a simple **rest-days difference** improved margin MAE only **0.00915 points** on 607 reused games (11.9537 no-rest versus 11.9445 with-rest). This is insufficient to justify a generic rest feature, let alone automatic rivalry, revenge, letdown, or motivational bonuses.

Among proposed external tests, an October 2026 published NCAA altitude association is testable with proper team/venue/travel evidence. It is an **association, not a ready-made +2 point betting adjustment**. All narrative factor effects require strong opponent/venue controls and negative controls. CBB-Q-018 already rejected generic rest features; do not re-open without a different, verifiable interaction hypothesis.

## Player shooting stability and replacement source gate

Retrospective longitudinal ESPN athlete identity matches, requiring ≥30 3PA in both seasons:

| Transition | Surviving matched players | Same team | Transfers | Prior-season vs next-season 3P% correlation | Raw prior 3P% next-year MAE | Prior 3P% with k=100 league-average pseudoattempts next-year MAE |
|---|---:|---:|---:|---:|---:|---:|
| 2023-24 → 2024-25 | 1430 | 867 | 563 | 0.2038 | 0.05792 | 0.04497 |
| 2024-25 → 2025-26 | 1311 | 629 | 682 | 0.1919 | 0.05672 | 0.04348 |

The k=100 rule is a **sensitivity example** and should not be called optimal or winner on fresh prospective data. Sample conditioning on **future ability to get 30 attempts** introduces survivorship/selection bias. For in-season historical projections, prior-season shot history is potentially available, but **current roster membership as of tipoff is NOT verified**, so M2 may not generate a certified historical score.

Next test: early prior seasons train hierarchical empirical Bayes shot-proficiency priors by attempt count, class, role, transfer direction, projected shot volume, and test next available season with **all rostered players**, including missingness indicators and novel freshmen, without conditioning on next-year attempts.

## Landscape research and ideas registry

**22 testable leads** with source, mechanism, data requirements, existing evidence, independent validation and priority are recorded in [MULTIMODEL_IDEAS](https://docs.google.com/spreadsheets/d/1a9FPD_oVL-ozVagzpuK1J_iQj8ssziIdCtl6JC0EUE8/edit#gid=20269003). They cover adjusted efficiency, Bayesian player progression, Empirical-Bayes shooting, shot quality, opponent pace style, possession-level event simulation, reweighted Four Factors, second-chance creation, three-heavy outcome variance, live-ball turnover conversion, altitude, rest/travel, coaching changes, cross-conference biases, neutral site, lineup RAPM, late-game possessions, model complementarity, market value, nonlinear specialists.

Source quality is heterogeneous:
- KenPom Four Factors: https://kenpom.com/blog/four-factors/
- KenPom stats: https://kenpom.com/blog/stats-explained/
- EvanMiya: https://blog.evanmiya.com/p/bayesian-performance-rating
- EvanMiya model-comparison: https://blog.evanmiya.com/p/which-cbb-supercomputer-should-you
- NCAA altitude research: https://doi.org/10.1177/15586235261496290
- RAPM data/methods: https://hoopr.sportsdataverse.org/articles/college-basketball-rapm-model.html
- Community ideas in Reddit are **hypotheses**, never validated evidence: https://www.reddit.com/r/sportsanalytics/comments/1v894os/a_college_basketball_model_worked_all_season_and/ and https://www.reddit.com/r/sportsbetting/comments/1ige280

A source URL or a Reddit user-reported ROI is NOT evidence that a feature reliably improves predictions or betting outcomes. Several proposed source URLs are leads rather than fully audited studies; verify provenance/license/coverage before ingestion.

## Governance and incomplete work

1. **R022** 55 clean earlier market-blind freezes remain intact. [R022_OUTCOME_RESULTS](https://docs.google.com/spreadsheets/d/1a9FPD_oVL-ozVagzpuK1J_iQj8ssziIdCtl6JC0EUE8/edit#gid=20261803) had no rows at this execution's verification. Some public media reported finals for specific games but a full *official final* source reconciliation was not completed. Do not invent tipoff/final verification nor claim all 55 graded.
2. **R029** is quarantined. Its later predictions were created after incidental market exposure and cannot be compared as clean prospective results.
3. R036/R037/R038 names were being used **concurrently by separate existing experiments**, so this lab was saved under the canonical `MULTIMODEL_` namespace to protect their distinct research.
4. No clean 2026-27 **new** predictions were issued by M1–M5 in this run; do not label historical 2025-26 rows as prospective or timestamp-frozen.
5. No model-specific ROI, expected value, closing-line value, win probability calibration or betting profitability has been established.
6. Does **not** include official D1 membership certification, all neutral-site maps, complete OT normalization, or audited pre-tip roster snapshots. M1 simplified raw PPP is not exact sealed v14; all model comparisons must be re-run against stronger exact production champion before any promotion.
7. Cross-model total and pace variance calibration for M5 remains unimplemented. The M4/M5 pace shown in scorecard is **M1 team-pace proxy**, not an independently generated M4/M5 pace forecast. It is *descriptive* only. M4 and M5 produce internally coherent individual scores, margins and totals.
8. Model diversity is evident but proper uncertainty benefit of disagreement has NOT been tested.

## Artifacts (verified native workbook and connected repo)

- [All 607 per-game outcome and independent predictions](https://docs.google.com/spreadsheets/d/1a9FPD_oVL-ozVagzpuK1J_iQj8ssziIdCtl6JC0EUE8/edit#gid=20269005)
- [Five model cards](https://docs.google.com/spreadsheets/d/1a9FPD_oVL-ozVagzpuK1J_iQj8ssziIdCtl6JC0EUE8/edit#gid=20269001)
- [Five model scorecard](https://docs.google.com/spreadsheets/d/1a9FPD_oVL-ozVagzpuK1J_iQj8ssziIdCtl6JC0EUE8/edit#gid=20269000)
- [Four Factors and player 3P/ rest ablations](https://docs.google.com/spreadsheets/d/1a9FPD_oVL-ozVagzpuK1J_iQj8ssziIdCtl6JC0EUE8/edit#gid=20269002)
- [22 ideas with source and status](https://docs.google.com/spreadsheets/d/1a9FPD_oVL-ozVagzpuK1J_iQj8ssziIdCtl6JC0EUE8/edit#gid=20269003)
- [Research gates](https://docs.google.com/spreadsheets/d/1a9FPD_oVL-ozVagzpuK1J_iQj8ssziIdCtl6JC0EUE8/edit#gid=20269004)
- [Python experiment](../scripts/cbb_multimodel_screen_20261106.py)
- [Player skill QA code](../scripts/cbb_multimodel_player_shooting_20261106.py)
- [Model metrics, correlation and paired boostrap](multimodel_screen_metrics_20261106.json)
- [Player skill summary and source hashes](multimodel_player_shooting_20261106.json)

Source raw files are sportsdataverse ESPN `hoopR-mbb-data/main/mbb/team_box/parquet/team_box_{2024,2025,2026}.parquet` and player analog. Fetch via existing project `scripts/cbb_r017_fetch_sources.py` or documented external raw path and run screen. Expect output `/tmp/cbb/multimodel/`. The existing executable uses local PyArrow/Pandas/scikit-learn source and needs those dependencies and the explicitly hashed source files.

## Ranked next experiments and research questions

**P0:** Reproduce *exact* sealed v14 score vectors on same historical games or prospective new games. Is M4's apparent lead merely because M1 was too simple and poorly calibrated? Need a true champion comparison.
**P0:** Score independently verified official R022 2026-27 finals append-only and compare λ4 and frozen base on same games. Focus margin and team score, separated by OT.
**P1:** Resolve archival 2024/25/26 pre-tip roster publications and player/team IDs. Does M2 improve even after strong v14, not raw-only M1?
**P1:** Re-run FF1/FF2/FF3 with *identical baseline prior strength + pace* and nested chronological weight learning. Are shooting component differences independently useful or only reintroducing team strength?
**P1:** Model calibrated pregame score uncertainty and held-out disagreement value: do four-model range and past residual correlation meaningfully predict future abs errors?
**P2:** Test altitude, specific rest/travel interactions, coach transition, opponent pace control and foul/shot mix only where source coverage, conditioning and negative controls support it.

## Decisions

M1 **KEEP_AS_LIGHTWEIGHT_COMPARATOR**; M2 **BLOCKED_DUE_TO_PIT**; M3 **CONTINUE_TESTING_EVENT_CALIBRATION**; M4 **CONTINUE_VALIDATION_WITH_V14_HEAD_TO_HEAD**; M5 **CONTINUE_VALIDATION_EXTRASEASON**. FF3 **NOT_PROMOTED**. Situational rest **NOT_PROMOTED**. Production models **NO_CHANGE**.

The deliverable required by the user is only partly complete: independent architecture prototypes and a reproducible retrospective ledger are real; 5 fully production-grade models, prospectively certified M2, profitability and any betting-edge proof remain unachieved.

## After-report R022 published-outcome preliminary check

To advance the already-frozen R022 prospective scorecard without contaminating the official-results ledger, a separate [MULTIMODEL_R022_EARLY_SCORE worksheet](https://docs.google.com/spreadsheets/d/1a9FPD_oVL-ozVagzpuK1J_iQj8ssziIdCtl6JC0EUE8/edit#gid=20269006) records **two independently reported final scores**, marked **NEWS_FINAL_VERIFIED_PRIMARY_OFFICIAL_PENDING**. These are descriptive interim scores, **not officially certified outcome-ledger rows and not a representative prospective sample**.

| R022 game, originally frozen before tipoff | Published final (home-away) | OT | λ4 margin absolute error | Rebuilt λ7 margin absolute error | λ4 total absolute error | Rebuilt λ7 total absolute error |
|---|---|---:|---:|---:|---:|---:|
| Valparaiso at Purdue | Purdue 80–83 Valpo | 2 | 24.7297 | 22.2502 | 11.4132 | 11.8183 |
| James Madison at Kentucky | Kentucky 89–80 JMU | 0 | 9.6070 | 6.8554 | 17.4918 | 17.6479 |

Sources: https://www.reuters.com/sports/basketball/valparaiso-shocks-purdue-buzzer-beater-2nd-ot--flm-2026-11-07/ ; https://www.reuters.com/sports/basketball/no-23-kentucky-gets-past-upset-minded-james-madison--flm-2026-11-07/ . R022's original 2026-11-06T22:43:54.641Z freeze remains unchanged. Purdue was an extreme upset in **second overtime**, so aggregate conclusion from these two would be especially misleading. Verify primary athletics final and actual tip-off before moving to the official R022 outcome table, and keep overtime diagnostics separate.
