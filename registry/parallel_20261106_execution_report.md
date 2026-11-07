# College Basketball Research OS | Six parallel workstreams
**Execution date:** 2026-11-06 Pacific (some source timestamps UTC November 7).  
**Decision:** **KEEP PRODUCTION / ALL EXPERIMENTS RESEARCH-ONLY**.  
**Status:** Four empirical tests executed, one independent frozen-champion provenance audit, and one independent live R022 scorekeeping pass. Complementarity was evaluated on matched 2024-25 games. No historical predictions were retroactively frozen. No live odds or market data entered forecast training.

## Research integrity and lineage

Governance: [AGENTS.md](../AGENTS.md), [Run registry](parallel_20261106_run_registry.csv). This is a new, named **PARALLEL_20261106** suite to avoid collision with existing R042–R045, previous R031 pace, the prior Five-Model Lab, and R022/R029 original records. Each empirical workstream has a separate registered experiment label and is linked to an existing open question. Prior solved or rejected questions (Q005, Q008, Q013, Q018, Q028 and others) are **not reopened** by these exploratory sensitivity checks.

**Data**: ESPN SportsDataverse men's basketball 2022-23, 2023-24, 2024-25 team box-score Parquet; player boxes 2022-23, 2023-24, 2024-25. Original source SHA256 files are pinned within each JSON metric artifact. Shared team-pregame feature builder is [controlled FF helper](../scripts/cbb_multimodel_fourfactors_controlled_20261106.py), itself SHA256-verified by experiment runner. Four historical models explicitly use *previous-season features* before target-season game. Score targets use current-game results **only as labels**. No 2025-26 sealed cohort was loaded for new empirical tests.

**Evaluation**: Team models developed on the **646** first-14-day games of 2023-24 (2022-23 source priors), then screened on **597** first-14-day games of 2024-25 (2023-24 priors). **These are previously explored historical periods**, not untouched out-of-sample certification. Game-date-clustered bootstrap confidence intervals have only 14 dates, are **not multiple-testing corrected**, and do not prove causal effects. Division I memberships, precise neutral-site assignments and overtime normalization remain provisional. Simplified structural source priors are **not the sealed v14 production champion**. The experiments were launched as separate processes simultaneously; three completed on first run, the player runner initially failed with a positional-index bootstrap error, then was fixed and independently rerun before any player result was accepted.

## Track 1: Why major favorites are underprojected

[Full same-games favorite data and bootstrap](parallel_20261106_favorite_metrics.json).

Separate corrections to stronger-team expected scoring were fitted only on the previous development season. The test varies stronger-team **prior offense**, weaker-team **prior defense allowed**, predicted strength gap, and their interaction. This does not identify a causal offensive or defensive mechanism because these variables correlate.

| Team-based variant, n=597 | Margin MAE | Gain vs original | Day-cluster bootstrap CI |
|---|---:|---:|---|
| Original prior team PPP / game scoring | 12.7927 | — | — |
| Previous-year two-sided intercept correction | 12.7251 | +0.0676 | [−0.1228,+0.2349] |
| Stronger-team offense only | 12.7736 | +0.0191 | [−0.1411,+0.1532] |
| **Weaker-team prior defense** | **12.4944** | **+0.2983** | **[+0.0906,+0.4992]** |
| Strength-gap only | 12.7118 | +0.0809 | [−0.1231,+0.2661] |
| Combined offense+defense+interaction | 12.6141 | +0.1785 | [−0.0288,+0.3862] |

The tested **20+ prior-margin cohort contained only 21 games**: baseline margin MAE **15.0774**, weaker-def-only **15.2191** (worse), stronger-off-only **15.1983** (worse), joint **15.9049** (worse). The original favorite-points forecast underpredicted actual favorite points on average **1.95 points** across the full screen and **4.99 points** in the 20+ subset.

**Conclusion:** A possible **ordinary-game calibration effect**, but not a validated extreme-favorite fix. The nominal CI above zero for the weaker-defense feature follows testing multiple variants, and the entire cohort was previously explored. Do not promote or add generic favorite points.

## Track 2: Is the pace problem a shared offset or true style interaction?

[Same-games pace and downstream totals](parallel_20261106_pace_metrics.json).

Using the source P475 box possession estimator, test prior-team mean, prior-development cohort global intercept, opponent schedule-adjusted pace, pace-gap interaction, strength-gap interaction, and both. Opponent adjustment is an approximate schedule correction using only teams' **previous-season** opponent pace, not current-game opponents' eventual statistics.

| Pace model, n=597 | Possession MAE | Game total MAE |
|---|---:|---:|
| Equal team mean | 4.2306 | **14.1139** |
| Prior-year offset (+1.7778 possessions) | 4.1152 | 14.1481 |
| **Opponent-adjusted + prior-year offset** | **4.0921** | 14.1043 |
| Pace-gap trained residual correction | 4.1172 | 14.1490 |
| Strength-gap trained residual correction | 4.1158 | 14.1491 |
| Combined gap and strength | 4.1177 | 14.1499 |

Opponent-adjusted pace improvement **+0.1385 possessions MAE**, 95% game-day-cluster CI **[−0.0734,+0.3227]**. Its total-points improvement was **+0.0097 MAE**, CI **[−0.3959,+0.3546]**. The season-offset alone slightly worsened totals.

**Conclusion:** No reliable predictive correction that improves the downstream scoring target. Do not simply increase pace for every matchup; rebuild with verified D1, opponent adjustment and regulation/OT before independent confirmation.

## Track 3: Four Factors v3 individual feature ablations

[Full same-baseline factor-by-factor results](parallel_20261106_factors_metrics.json).

Every candidate used **identical prior-only F0 strength, total, pace and rating baseline features**, same matched 646/597 games, same Ridge alpha=120, then added only the named factor group. Margin and total were trained independently. This corrects the prior FF3 comparison's unequal-baseline weakness.

- Baseline F0: **11.6316 margin MAE** and **13.7316 total MAE**.
- Best-looking *exploratory* margin group was opponent fouls, MAE **11.5949**, gain **+0.0366**, CI **[−0.0796,+0.1310]**. Neither this nor any other isolated group provided reliable margin improvement.
- FTR alone **worsened total MAE** to **13.8166** (gain −0.0850, nominal CI [−0.1411,−0.0141]).
- Opponent fouls alone **worsened total MAE** to **13.8410** (gain −0.1094, nominal CI [−0.1775,−0.0364]).
- The compact original four-factor add-on worsened total MAE to **13.8721** (gain −0.1405, nominal CI [−0.2381,−0.0337]).

**Conclusion:** Different targets need independently chosen factor sets; more inputs did **not** reliably improve results. No generic Four Factors v3 promotion. Because multiple variations were tested, even nominally significant negative contrasts are exploratory until replicated independently.

## Track 4: Player progression and shooting-persistence study

[Full player K-grid, identity coverage and per-cohort errors](parallel_20261106_players_metrics.json) and [league-only comparison](parallel_20261106_player_league_comparison.json).

Players with **at least 10 three-point attempts in the prior season** were eligible for the longitudinal study. Dev: 2022-23 to 2023-24, 2,175 players with at least one next-season 3PA. Historical screen: 2023-24 to 2024-25, 2,175 evaluable players with **192,062 next-season attempts**, from 3,371 prior eligible players. **1,196 prior eligible players lacked usable next-season shot observations.** This is survival/selection conditioning, not a known historical pregame roster.

Shot skill prior: $(3PM_{prior} + K\times 3P\%_{league,prior})/(3PA_{prior}+K)$. The K grid was selected using **development-season next-shot binomial log loss only**, then evaluated on the later season. Extended sensitivity grid **K = 0,15,30,50,75,100,150,200,300,500,1000,2000,1000000** establishes **K=300** was an interior optimum in the training grid.

| Historical next-year screen | Per-shot negative log loss |
|---|---:|
| Raw prior-year player 3P% (K0) | 0.653485 |
| **Train-selected K300** | **0.643608** |
| League-only limiting prior (K1,000,000) | 0.644027 |

K300 gain vs raw **+0.009877** log loss per next-season 3PA, school-cluster bootstrap 95% CI **[+0.008221,+0.011575]**. Crucially, gain beyond merely using the league prior is much smaller: **+0.0004185** per shot, 405 prior-school clusters, 95% bootstrap CI **[+0.000258,+0.000586]**. Repeated-player selections, shot volume and next-season survival remain potential applicability limits.

Of 2,175 evaluated players, 1,328 remained with the prior-program team and 847 transferred **as observed retrospectively**, which is not a legal historical pregame transfer feature without a posted-as-of roster. No independent M2 team-score forecast was constructed or certified.

**Conclusion:** Player shooting needs strong shrinkage, and a small but measurable **individual player-specific** prior can beat flat league average for observed returning shooters. Source and expected-minutes timing still block M2 as a full five-model independent scorer.

## Does model diversity produce complementary errors?

[Matched game error correlation and simple ensemble diagnostics](parallel_20261106_complementarity_metrics.json) and [reproducible script](../scripts/cbb_parallel_complementarity_20261106.py). Same 597 2024-25 historical screen; 50/50 blends were not optimized on test outcomes.

| Comparison | Error correlation | Best individual MAE | Equal blend MAE | Gain versus best |
|---|---:|---:|---:|---:|
| Weaker-defense margin adjustment vs FF F0 | **0.9856** | 11.6316 | 11.9021 | **−0.2706** |
| Corrected pace total vs FF F0 | **0.9649** | 13.7316 | 13.6809 | **+0.0507** |
| Weaker-defense total vs FF F0 | **0.9714** | 13.7316 | 13.7196 | **+0.0119** |

**Interpretation:** These particular variants contain **highly overlapping errors**. The margin blend is worse than the better model, and the total blend improvements are very small and were examined post-hoc; do not promote an ensemble. This supports prioritizing calibration and genuinely distinct player/event mechanisms. It does **not** show that all five desired model architectures have correlated errors or that no ensemble could ever work.

## Independent audit 5: v14 exact champion provenance

**Status: BLOCKED.** Verified [main repo tree](https://github.com/goldbondman/2026.27.CBB), [sealed workflow](https://github.com/goldbondman/2026.27.CBB/blob/main/.github/workflows/sealed-evaluation.yml), [sealing governance](https://github.com/goldbondman/2026.27.CBB/blob/main/src/cbb/governance/sealed.py), latest R041 audit, source question map and Project file search. The sealed workflow explicitly says **"Protected hook only; no holdout evaluation is implemented or invoked in v0.1."** There is no recovered immutable v14 commit+config+game-level prediction set with deterministic historical run parity. A simplified M1/FF F0 comparator is **not v14**, and existing aggregate metrics do not substitute for identical-game champion vectors.

**Decision:** Do not claim M4, M5 or any new track beats v14. No sealed holdout access or market-informed model tuning occurred. Exact champion artifact and authorized evaluation are still critical path before promotion.

## Independent audit 6: R022 prospective freeze scoring

A fresh ESPN scoreboard source snapshot at **2026-11-07T03:38:47.622Z** (November 6, 7:38 pm Pacific), source SHA256 **982ff622945c6fd642caef67822e2b38583fe69fe4f02269d750d5faf690f9b2**:

- **39 of 55 R022 forecasts** matched final provider results, all with verified pregame freeze; 16 were not FINAL.
- All 55 event identities still matched; original frozen forecasts were not changed. **Three additional provider-final rows were appended** to the existing independent [R022 ESPN final sheet](https://docs.google.com/spreadsheets/d/1a9FPD_oVL-ozVagzpuK1J_iQj8ssziIdCtl6JC0EUE8/edit#gid=20269007). The original official-outcome ledger remains untouched because primary team score certification was not independently obtained.
- Comparing the λ4 challenger against the **original separately frozen baseline** (not the rebuilt λ7 surrogate) on the same 39 games:

| Target | λ4 MAE | Frozen-original MAE | Paired MAE gain | 95% game bootstrap CI |
|---|---:|---:|---:|---|
| Margin | **11.6449** | 11.9821 | **+0.3373** | [−0.3354,+1.0274] |
| Total | **12.4225** | 12.4987 | **+0.0762** | [−0.0530,+0.2034] |

Both CIs include zero, and 16 outstanding games could change the comparison. **No promotion.** Quarantined R029 shadow predictions remain excluded as non-blind.

## Artifacts and verification

- [Four-track historical experiment script](../scripts/cbb_parallel_hypothesis_lab_20261106.py)
- [Historical favorite source and metrics](parallel_20261106_favorite_metrics.json)
- [Historical pace source and metrics](parallel_20261106_pace_metrics.json)
- [Individual Four Factors source and metrics](parallel_20261106_factors_metrics.json)
- [Player K-grid metrics and source hashes](parallel_20261106_players_metrics.json)
- [Player versus league-only nested comparison](parallel_20261106_player_league_comparison.json)
- [Error-correlation / ensemble metrics](parallel_20261106_complementarity_metrics.json)
- [Append-only experiment tracking](parallel_20261106_run_registry.csv)
- [Six-track live workbook summary](https://docs.google.com/spreadsheets/d/1a9FPD_oVL-ozVagzpuK1J_iQj8ssziIdCtl6JC0EUE8/edit#gid=20269240)
- [82 individual ablation rows](https://docs.google.com/spreadsheets/d/1a9FPD_oVL-ozVagzpuK1J_iQj8ssziIdCtl6JC0EUE8/edit#gid=20269241)
- [39 individual player-shrinkage subgroup rows](https://docs.google.com/spreadsheets/d/1a9FPD_oVL-ozVagzpuK1J_iQj8ssziIdCtl6JC0EUE8/edit#gid=20269242)
- [R022 provider-final paired summary](https://docs.google.com/spreadsheets/d/1a9FPD_oVL-ozVagzpuK1J_iQj8ssziIdCtl6JC0EUE8/edit#gid=20269243)
- [Audit and promotion gates](https://docs.google.com/spreadsheets/d/1a9FPD_oVL-ozVagzpuK1J_iQj8ssziIdCtl6JC0EUE8/edit#gid=20269244)
- [Complementarity test results](https://docs.google.com/spreadsheets/d/1a9FPD_oVL-ozVagzpuK1J_iQj8ssziIdCtl6JC0EUE8/edit#gid=20269245)

Game-level CSVs for the four tracks were emitted to the research sandbox by the reproducible source script. Their sample-level aggregates, underlying source hashes and comparison intervals are persisted in GitHub and Sheets. Rerun the script from the repo against the hash-pinned inputs to reproduce the full historical per-game vectors. No retroactive timestamp freezes were made.

## Priority next independent tests (without changing production)

1. **v14 exact provenance**: Obtain champion commit, config, game-level records, frozen snapshots and run parity. The last barrier to legitimate model promotion. If impossible, maintain explicit `CHAMPION_COMPARATOR_UNAVAILABLE` instead of a fake head-to-head.
2. **D1 + regulation/neutral certified extreme cohorts**: Extend verified large-favorite sample (only 21 in the new screen), isolate favorite offense from opposing defense, and only then prospectively freeze candidate adjustments.
3. **Player skill into full scoring**: Verify published-before-tipoff rosters and expected minutes. Test K300 player 3P priors in a complete M2 team-score model rather than claiming next-year shooting improvements prove margin improvement.
4. **Target-specific factor selection**: Test turnover/shot factors with nested historical folds and compare separately for margin and total on a new game set; record all failed candidates.
5. **R022 full slate**: Continue independent source-first official scoring of remaining 16 and recompute paired full-slate uncertainty; preserve freeze and market firewall.
6. **Truly independent residuals**: Compare player-aware score and generative event models prospectively; current simple shared-source ensembles have residual correlations above 0.96.

**Final decision: KEEP FROZEN PRODUCTION. CONTINUE RESEARCH. NO CONFIRMED PROFITABILITY OR PRODUCTION-GRADE MODEL IMPROVEMENT.**
