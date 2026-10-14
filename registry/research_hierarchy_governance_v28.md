# CBB Research Objective Hierarchy Governance v28

As of: 2026-10-14

This is an OS architecture/governance layer over the existing College Basketball research estate. It does not restart, renumber, or replace historical research, models, gates, decisions, the 47-agent / 10-team / 19-route CBB agent architecture, or the A18 market firewall.

## Permanent hierarchy

Project Objective → Research Domain → Research Program → Research Question → Research Ticket → Experiment → Finding → Decision / Certification.

Historical gate/execution records remain gate records. Do not fabricate retroactive Experiment_ID values when none existed. Future substantive empirical tests must receive Experiment_ID and link to Ticket_ID / Question_ID / Program_ID / Objective_ID.

## Project objectives

- CBB-OBJ-001 — Trustworthy point-in-time basketball data and reconstruction.
- CBB-OBJ-002 — Accurate market-blind team strength, pace, and score forecasting.
- CBB-OBJ-003 — Reliable personnel and early-season state estimation.
- CBB-OBJ-004 — Context-sensitive forecasting without double counting.
- CBB-OBJ-005 — Calibrated score/outcome distributions and target-specific uncertainty.
- CBB-OBJ-006 — Discover incremental predictive information and govern challengers.
- CBB-OBJ-007 — Production reliability, reproducibility, and immutable blind forecasting.
- CBB-OBJ-008 — Separate market/betting value from forecast quality and learn postgame.

## Research domains

Data Foundation; Core Forecasting; Personnel & State; Context & Environment; Distribution & Reliability; Incremental Information; Model Governance; Production; Market & Betting; Postgame Learning.

## Programs

- CBB-PROG-001 / OBJ-001 / Data Foundation — Canonical corpus, semantics & PIT integrity.
- CBB-PROG-002 / OBJ-001 / Data Foundation — Possession & event reconstruction.
- CBB-PROG-003 / OBJ-002 / Core Forecasting — Structural offense/defense state.
- CBB-PROG-004 / OBJ-002 / Core Forecasting — Pace and possession forecasting.
- CBB-PROG-005 / OBJ-002 / Core Forecasting — Coherent team-score architecture.
- CBB-PROG-006 / OBJ-003 / Personnel & State — Availability, rotation & roster legality.
- CBB-PROG-007 / OBJ-003 / Personnel & State — Roster/transfer priors & early adaptation.
- CBB-PROG-008 / OBJ-004 / Context & Environment — Home court, road and neutral context.
- CBB-PROG-009 / OBJ-004 / Context & Environment — Rest, compression, travel & schedule context.
- CBB-PROG-010 / OBJ-004 / Context & Environment — Tournament transfer, upset & special context.
- CBB-PROG-011 / OBJ-005 / Distribution & Reliability — Joint score, margin & total distributions.
- CBB-PROG-012 / OBJ-005 / Distribution & Reliability — Calibration, uncertainty & selective prediction.
- CBB-PROG-013 / OBJ-006 / Incremental Information — Component/event information.
- CBB-PROG-014 / OBJ-006 / Incremental Information — Recency, adaptive state & calibration-controlled form.
- CBB-PROG-015 / OBJ-006 / Incremental Information — Residual ML, interactions & challenger combination.
- CBB-PROG-016 / OBJ-006 / Model Governance — Champion/challenger certification & freeze/reopen.
- CBB-PROG-017 / OBJ-007 / Production — Operational readiness & hardening.
- CBB-PROG-018 / OBJ-007 / Production — Blind lock, launch operations & monitoring.
- CBB-PROG-019 / OBJ-008 / Market & Betting — Market value & disagreement.
- CBB-PROG-020 / OBJ-008 / Market & Betting — Bet expression, thresholds & staking.
- CBB-PROG-021 / OBJ-008 / Postgame Learning — Results grading & error-trigger loop.

## Ticket creation QA

Every new predictive/research ticket must contain Objective_ID, Program_ID, and Question_ID. Exceptions are OS_INFRASTRUCTURE, DATA_INTEGRITY, SOURCE_QA, GOVERNANCE, INCIDENT_RESPONSE, and PRODUCTION_RELIABILITY, but these should still reference protected objectives where applicable.

Before opening a Question, Ticket, or Experiment, search the Objective Map, Question ledger, Master Plan, Search Space, existing gates/tickets, Findings, Decisions, and Knowledge Packages. Classify proposed work NEW, REPLICATION, ROBUSTNESS, ABLATION, EXTENSION, NEGATIVE_CONTROL, DUPLICATE, or SUPERSEDED.

## Question stopping states

Use SUFFICIENT_EVIDENCE_FOR_CURRENT_DECISION, REJECTED_WITH_SUFFICIENT_EVIDENCE, BLOCKED_PENDING_NEW_INFORMATION, PROSPECTIVE_VALIDATION_REQUIRED, or CONTINUE_RESEARCH. Do not continue a saturated branch just because another variation can be tested.

## Routing priority

Prioritize executable work using Expected Research Value × Objective Importance × Decision Relevance × Uncertainty Reduction × Critical-Path Leverage. Dependencies and executable state are hard constraints. A high raw ERV inside a saturated low-value question does not outrank a major project blocker.

## Findings and decisions

Every material Finding and Decision must map upward to Question → Program → Objective. Decisions must distinguish research conclusion from production implication, identify evidence maturity (retrospective / replicated / OOS / prospective), and state revisit conditions.

## Forecast / market separation

CBB-OBJ-001 through CBB-OBJ-007 primarily govern market-blind forecasting and production. Market/betting work under CBB-OBJ-008 occurs only after immutable Stage A forecast lock and A18 authorization. Profitable betting outcomes cannot retroactively justify forecast leakage, weak PIT methodology, or an uncertified model change.

## Preserved agent architecture

The existing CBB Agent/Team OS remains authoritative: A00–A46 (47 agents), 10 teams, 19 routes. Forecast chain remains A21 Team State → A22 Availability/Player State → A23 Lineup/Rotation → A24 Matchup/Components → A25 Game Simulator/Score Distribution → A26 Independent Challenger → A27 Prediction Adjudicator → A18 Market Firewall. Discovery specialists A40–A46 remain subject to hierarchy, anti-duplication, saturation and ERV controls.

## Current v28 control state

- Production forecast: v14 FROZEN.
- 2026 outcomes used for model construction: FALSE.
- Activated non-neutral adjusted-L7 challenger: formal production-reopen tournament earned, not promoted.
- Question map: 47 questions.
- Existing gate records mapped: 330.
- Existing material decisions mapped: 186.
- Existing evidence/finding rows mapped: 180.
- Historical first-class CBB Research Ticket registry found in inspected repo/tracker: 0; no legacy Ticket IDs fabricated.
- New lineage-valid current tickets: CBB-RT-001 through CBB-RT-007.
- Saturated/rejected branches explicitly stopped: 15.
- Orphaned meaningful gate/decision/evidence records after backfill: 0.

## Critical-path queue

1. CBB-RT-001 / CBB-Q-032 — freeze prospective 2026 adjusted-L7 activation-selection protocol using historical data only.
2. CBB-RT-002 / CBB-Q-029 — clean activated L7 vs L7+ORB / L7+DRB / L7+ORB+DRB test without failed TS.
3. CBB-RT-003 / CBB-Q-037 — formal immutable-v14 common-fold production-reopen tournament after RT-001 and RT-002.
4. CBB-RT-006 / CBB-Q-041 — prospective 2026 blind scorecard/monitoring when forward outcomes are legitimately available.
5. CBB-RT-005 / CBB-Q-014 — timestamp-valid historical roster/recruit source resolution; blocked pending new legal source.

## Autonomous cycle rule

Load Objective Map → assess objective uncertainty → inspect open Questions → dependencies/blockers → objective-aware ERV → select highest-leverage executable Ticket → route through existing CBB agents/teams → execute Experiment → audit PIT/leakage/reproducibility → persist Finding/Decision → update Question → Program → Objective → rerank portfolio → continue to next meaningful gate.

Do not stop merely because one Ticket completes. Do not continue indefinitely when the Question’s current decision threshold has already been reached.
