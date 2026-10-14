# CBB Laboratory Governance

1. Never use sealed 2026 data during ordinary development.
2. Never leak current or future information into pregame features; dates update in batches.
3. Never silently change canonical data semantics.
4. Frozen R1/R2 outputs are regression targets, not tuning objectives.
5. Keep market data isolated from blind modeling.
6. Challengers cannot promote themselves; status changes require human review.
7. Register every experiment and never overwrite a record.
8. Failed experiments remain recorded.
9. Large searches require nested chronological validation.
10. Prefer robust parameter regions over isolated optima.
11. Do not introduce unrelated feature families during an assigned gate.
12. Any leakage discovery is a **RED integrity event**: stop, preserve evidence, and report it.

## Research Objective Hierarchy — v28

13. Every new predictive/research ticket must reference a valid `Objective_ID`, `Program_ID`, and `Question_ID`. Infrastructure/governance/data-integrity tickets may use their explicit exception class but should still reference the objective(s) they protect when applicable.
14. Before opening a Question, Ticket, or Experiment, search existing Objective Map, Questions, gates/tickets, Findings and Decisions; classify the work as NEW, REPLICATION, ROBUSTNESS, ABLATION, EXTENSION, NEGATIVE_CONTROL, DUPLICATE, or SUPERSEDED.
15. Every Question must carry a stopping state. Do not keep testing branches marked `SUFFICIENT_EVIDENCE_FOR_CURRENT_DECISION`, `REJECTED_WITH_SUFFICIENT_EVIDENCE`, `BLOCKED_PENDING_NEW_INFORMATION`, or `PROSPECTIVE_VALIDATION_REQUIRED` unless the stated reopen condition occurs.
16. Route research by Expected Research Value × Objective Importance × Decision Relevance × Uncertainty Reduction × Critical-Path Leverage, subject to dependencies and executable state. Raw isolated ERV is not enough.
17. Every future substantive empirical test receives an `Experiment_ID` linked to its Ticket/Question/Program/Objective. Do not fabricate retroactive Experiment IDs for historical gate records that did not have one.
18. Every material Finding and Decision must map upward to Question → Program → Objective and distinguish research conclusion from production implication.
19. Preserve the existing CBB Agent/Team OS IDs and routes. The objective hierarchy is a supervisory control layer; it does not renumber or replace A00–A46, the 10 teams, the 19 routes, or A18 Market Firewall.
20. Discovery agents A40–A46 may create a new branch only after anti-duplication, decision-relevance, objective lineage, and ERV checks pass.
21. Stage A forecast research remains market-blind. No market/betting result may retroactively justify a forecast feature, coefficient, calibration choice, or certification.
22. After each material Finding or Decision, update Question state → Program roll-up → Objective roll-up → dependencies → objective-aware queue ranking before selecting the next meaningful gate.

See `registry/research_hierarchy_governance_v28.md`, `registry/research_question_map_v28.csv`, and `registry/research_queue_v28.json` for the current control layer.
