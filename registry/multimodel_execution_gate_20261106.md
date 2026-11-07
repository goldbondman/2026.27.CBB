# Five-model advancement | Executed research gate 2026-11-06

**Status: PARTIALLY EXECUTED; production baseline preserved.** User directive: five independent architectures against sealed v14, prospective M2, equal-baseline Four Factors 3.0, situational factors, verified R022 prospective scores. See [prior model lab](multimodel_discovery_20261106.md). This is an **extension/controlled ablation**, not an entirely new validation of the previously examined 2025-26 holdout. No licensed sportsbook market inputs used to train new challenger coefficients.

## 1. Exact v14 champion: HARD BLOCKER, not substituted

Source audit of `AGENTS.md`, the code in `src/cbb/governance/sealed.py`, `.github/workflows/sealed-evaluation.yml`, and existing [R041_MODEL_AUDIT](https://docs.google.com/spreadsheets/d/1a9FPD_oVL-ozVagzpuK1J_iQj8ssziIdCtl6JC0EUE8/edit#gid=20263500):

- **No precise v14 champion commit/config, executable scorer, original historical per-game predictions, or feature snapshots have been recovered.** M1 in the Multi-Model Lab is a **simplified prior-season raw-efficiency comparator**, never the v14 champion.
- The sealed-evaluation workflow file currently contains a **protected placeholder**, not an implemented scorer. Governance explicitly prohibits ordinary development access to the sealed 2026 holdout. Do not run ad hoc code against that holdout.
- Existing 4,357-game legacy aggregates are NOT a matched per-game v14 comparison, and prior M2/M4/M5 labels refer to different historic architectures. Keep versions and labels unambiguous.
- **Decision:** V14_CHAMPION_COMPARISON_BLOCKED. Obtain exact immutable champion model, environment, run configuration, and archived game vectors and then authorize a separated, access-controlled comparison. Neither existing M4 nor M5 can be called superior.

## 2. Prospective R022: 36 source-verified completed games, partial score

The source is the ESPN men's basketball API November 6 slate, `groups=50&limit=500`. Saved response timestamp **2026-11-07T03:14:56.001677Z** (Nov 6, 7:14 pm Pacific); raw response SHA256 **3a98d8799e475270924433998b52a925170a1b7b36e08f0df7d322130bc6c6b9**.

- 72 event IDs in the API response; 50 ESPN marked FINAL then.
- **All 55 original R022 forecasts matched** on ESPN IDs (50 R018) or uniquely grounded home/away names (5 R014); no identity failures.
- 36/55 R022 forecasts had FINAL scores; 19 were still live/scheduled.
- **All 36 passed** `freeze_utc < frozen_scheduled_tipoff_utc` and `freeze_utc < ESPN_scheduled_tipoff_utc`.
- **No old frozen R022 row or official R022 results ledger was edited.** An independent append-only provider-scored worksheet preserves all 36 outcomes and exact provider-source hash, with explicit `ESPN_SCORES_NOT_PRIMARY_SCHOOL` status. ESPN final is authoritative *provider* evidence but not a source of independently verified school athletics box scores.

| Matched 36 provider FINAL games | Lambda 4 | Rebuilt Lambda 7 |
|---|---:|---:|
| Margin MAE | **11.6576** | 11.9753 |
| Total MAE | **12.7933** | 12.8614 |

Paired game-bootstrap fixed RNG (5,000 resamples; exploratory):
- Lambda4 margin MAE gain **+0.3177** points, 95% CI **[-0.4084,+1.0005]**; wins/losses against λ7 23/13.
- Lambda4 total MAE gain **+0.0680**, 95% CI **[-0.0483,+0.1786]**; wins/losses 24/12.
- Average actual-minus-frozen-rebuilt-λ7 **favorite margin residual +6.8215 points**. This is a descriptive first-36 early-game selection, **not causal shrinkage evidence**, no blanket favorite boost. Favorite projected 20-30 n=5, 30+ n=1; insufficient.
- ESPN identifies Purdue–Valparaiso as **double overtime**; other early outcomes not independently normalized to 40 minutes.
- **Decision:** KEEP BOTH FROZEN, DO NOT PROMOTE. Finish remaining finals, reconcile exact regulation/OT and independently verify official primary box scores.

Saved:
- [Provider-verified per-game partial results](https://docs.google.com/spreadsheets/d/1a9FPD_oVL-ozVagzpuK1J_iQj8ssziIdCtl6JC0EUE8/edit#gid=20269007)
- [R022 bootstrap and mismatch summary](multimodel_r022_partial_diagnostics_20261106.json)
- [Original immutable forecasts](https://docs.google.com/spreadsheets/d/1a9FPD_oVL-ozVagzpuK1J_iQj8ssziIdCtl6JC0EUE8/edit#gid=20261800)

## 3. Four Factors 3.0 controlled *same-input* experiment: negative evidence

Implemented [reproducible, versioned Python ablation](../scripts/cbb_multimodel_fourfactors_controlled_20261106.py) and [full 4-specification metrics and paired intervals](multimodel_ff_controlled_metrics_20261106.json). No data from sealed 2025-26 holdout were loaded.

- Prior 2022-23 team box scores -> 2023-24 first 14 days **646 development games**.
- Prior 2023-24 team box scores -> 2024-25 first 14 days **597 historical screening games**.
- Four experiments share exact historical team strength+pace baseline, exact games, target definitions, ridge regularization α=120 and training budget. This is a simplification, **not v14 exact source parity**.
- F0: structural margin/total/pace/strength features.
- F1: F0 + home/away predicted eFG, turnover, offensive rebounding and FTR relative features.
- F2: F1 + pregame shot mix, 2P%, 3P%, FT%.
- F3: F2 + steals, fouls, FTM per possession and TS%.
- All added variables use **previous-season team and opposing-allowed rates**. A first-run column collision between prior-season `steals` and current target-game `steals` produced a hard KeyError, no predictions. Code was corrected with explicit `p_h_` and `p_a_` **prior-only feature prefixes** and the successful run re-executed. The code used for reported results has an explicit pregame collision guard.

| Specification | Margin MAE | Total MAE |
|---|---:|---:|
| F0 strength+pace | 11.6316 | **13.7316** |
| F1 + Four Factors | 11.6279 | 13.8721 |
| F2 + shooting split | 11.5900 | 13.9963 |
| F3 + event factors | **11.5781** | 14.0576 |

F3−F0 margin MAE gain **+0.0534**, 14-date clustered bootstrap CI **[-0.1497,+0.2130]**. No reliable margin gain.
F3−F0 total MAE gain **-0.3260**, 14-date clustered bootstrap CI **[-0.5107,-0.1436]**. Expanded factors **worsened totals** in this simplified historical screen.

**Decision:** F3 NOT PROMOTED for margin or total. Penalize unnecessary complexity, and **test margin and total feature families separately**. Mathematical factor-component relationships are not causal effect estimates, and shot-level feature quality remains a constraint.

Saved:
- [Controlled factor comparison](https://docs.google.com/spreadsheets/d/1a9FPD_oVL-ozVagzpuK1J_iQj8ssziIdCtl6JC0EUE8/edit#gid=20269008)
- [Live decision gates](https://docs.google.com/spreadsheets/d/1a9FPD_oVL-ozVagzpuK1J_iQj8ssziIdCtl6JC0EUE8/edit#gid=20269009)

## 4. M2 pregame roster access: promising snapshot, NOT operational scoring

New timestamped public source snapshot captured **2026-11-07T03:21:43.226555Z**, Nov6 7:21 pm Pacific; these inputs can be candidates **only for games after this retrieval**.

- 2026-27 ESPN season roster: **12,563 rows; 743 team identifiers**.
- ESPN 2026-27 player game boxes: **9,723 rows, 303 games through 2026-11-05**.
- ESPN 2026-27 per-game roster: **9,723 rows / 7,557 unique team-player pairs across 303 games**.
- Season roster contains **7,446 of those 7,557** past game-roster pairs. **111 missing pairs (1.47%)**, showing why retrospective current-roster membership is not complete historical truth.
- All raw Parquet SHA256 hashes and snapshot timestamp are [stored](multimodel_m2_prospective_source_snapshot_20261106.json). [Cross-source join QA](multimodel_m2_roster_join_qa_20261106.json).
- Current source snapshot does NOT establish historical first-published roster membership; it also does not supply independently approved expected-minutes, freshmen and translated-transfer predictions.
- **Decision: M2 RESEARCH-ONLY / PREGAME SOURCE STAGED / FORECAST BLOCKED.** Build an explicit current pregame roster snapshot gate, no historical roster backfill. Do not touch sealed 2025-26 data for model development without authorization.

## 5. Situational research and fresh five-model forecasts

- Reviewed existing R041 audit and initial same-game rest ablations. Generic rest effects were tiny and previously stopped. No altitude, trip, coaching and injury source with full pregame as-of and controls was available for a fresh, independent reliable test in this batch.
- **No new Nov6 five-model forecasts were created.** The 19:12–19:21 PT research run happened after many games began. Public score research also showed sportsbook-market lines incidentally. The strict market-blind firewall prohibits retroactively representing later forecasts as uncontaminated pregame outputs.
- Reserve next clean upcoming slate for independent source-snapshotted predictions in a fresh Stage A workflow, with no sportsbook or result lookup before freeze. M2 remains `NO FORECAST` until minutes/roster model is actually executable. All existing forecasts are preserved.

## Model readiness decision

| Model | Historical implementation | v14 same-game parity | New pretip independent stream | Decision |
|---|---|---|---|---|
| M1 structural | Reconstructed simple prior PPP | **UNVERIFIED** | Not issued | RESEARCH_ONLY |
| M2 player/roster | Source pilot only | **BLOCKED** | Not issued | PIT_AND_MINUTES_BLOCKED |
| M3 possession events | Historical screen | **UNVERIFIED** | Not issued | RESEARCH_ONLY |
| M4 dynamic Elo | Historical screen | **UNVERIFIED** | Not issued | CONTINUE_VALIDATION |
| M5 nonlinear score model | Historical screen | **UNVERIFIED** | Not issued | CONTINUE_VALIDATION |
| v14 exact champion | Not recovered | **BLOCKED** | Existing original forecasts intact | KEEP_PRODUCTION |

No prediction ROI/profitability is asserted or evaluated from these research outputs.

## High-value open questions

1. **Exact champion provenance**: Where are the production v14 commit, runtime config, roster/ratings/pace source snapshots, frozen game-level vectors, and authorized comparator? Existing sealed GitHub workflow does not implement a scoring job. Without this, M4's low MAE against a weak baseline is not a valid champion challenge.
2. **Are large-favorite errors prior-offense or opponent-defense specific?** The 36 R022 finals show +6.82 favorite residual, but λ4 vs λ7 CI includes zero. Expand the full clean future sample; do not tune on initial postgame residuals.
3. **Does a factor specialist add signal conditional on strong opponent-adjusted ratings?** This F0 used raw prior PPP; F1-F3 barely affected margins and worsened totals. Evaluate via correctly nested source-PIT crossfit and separate target selection.
4. **Can complete verified player game-availability histories be built prospectively?** 111 roster/game-roster mismatches already show snapshot limitations. Add official roster and confirmed injury as-of metadata, expected-minutes simplex and partial-data reliability.
5. **Do altitude, coach changes and travel affect efficiency independently after verified venue, roster and strength controls?** Data still missing for causal observational tests.
6. **Does cross-model disagreement predict prospective error?** Historical M1–M5 margins disagree, but no valid prospective conditional reliability comparison exists.

**Final research gate: KEEP BASELINE; CONTINUE_TESTING; NOTHING PROMOTED.** Source and calculation records are written and independently verified. Historical holdout safety, market-blind pregame ordering, and original freeze immutability remain the constraints.
