# Universal 2026-27 matchup-pace, roster-turnover and shooting-performance research protocol
Date: 2026-11-06. **Classification: NEW ROBUSTNESS/EXTENSION protocol, not an experiment result.** Linked to CBB-OBJ-001/002/003/007; CBB-Q-014 (blocked roster source provenance), CBB-Q-041 (prospective blind workflow). Do not reopen legacy Q004, Q005, Q013, Q025 stop states simply to repeat old tests. No production model modifications. Preserve R2/R005/R007/R014/R018/R019/R022 frozen predictions.

## Verified status / why this is universal
* R014 pace is **game-specific**, not fixed: BC 66.3872 and UMass 70.8822 adjusted team paces, arithmetic blend **68.6347 possessions per team**. K8 previous-pace pseudogames plus current 2026-27 games; `0.475` is a **fixed possession-estimator free-throw coefficient**, not a fixed number of projected possessions.
* BC–UMass final Nov 6: 71–55 (neutral event). BC 3/27 3PT, 24/26 FT, 3 turnovers; UMass 5/18 3PT, 8/14 FT, 13 turnovers. BC projected 71.1461, UMass 71.0564. Estimated game possessions = average(59−11+3+.475×26, 54−12+13+.475×14) = **62.5**, roughly **6.1 below projection**. Final margin under-predicted by 15.91, total over-predicted by 16.20. Do NOT fit to this single game.
* Illustrative, **not causal or fair-score**: at identical 33% 3PT on fixed attempts, net margin changes +14.91 toward BC; at identical 75% FT, −7.00 toward BC; net +7.91 vs actual margin 16. Player/shot quality and altered possession chains make this a toy computation, NOT a predicted deserved 24-point outcome.
* Oct 6 BC coach says 13 new faces, 11 transfers, 2 freshmen, 2 returning players; earlier note of 10 transfers was inaccurate. Source: https://asaptext.com/orgs/acctipoff/browse_file.php?browse_file_name=transcripts%2F171813.html . BC athletics Nov2 has a slightly different count; retain dated provenance.
* R024 historical prior-top50 vs prior-bottom100 November cohort: 240 games, mean predicted stronger-team margin 20.24 vs actual 27.90, 7.66 underprediction. Same-game `lambda8` MAE 13.466, `lambda4` 12.041, `lambda0` 11.285. 2023-24/2024-25/2025-26 **historically explored, provisional D1**, not untouched. R022 already froze prospective less-shrinkage forecasts. **First score R022 without modifying it**, not posthoc changing every spread.
* R017 roster-aware mean margin gain 0.003 points and totals +0.141 with CI crossing zero; roster-aware risk error model worse. 2025-26 retrospective cohort reused, roster source publication timestamps not verified; no production credibility.

## Workstreams with strict pregame features
**P: Matchup tempo (NOW).** Compare P0 actual R014/R018 equal mean with P1 opponent-adjusted team pace, P2 learned slower/faster or difference interaction, P3 pace prior conditioned on pregame coach/team continuity if available. Include pace per 40, regulation/OT adjustments, neutral, opponent adjustment, mismatch under-pace bias. Never use final-game pace as predictor. Evaluate pregame possession MAE + total MAE on identical games; no new P0 tuning from Nov6 finals.

**S: Prior season replacement (PIT BLOCKED).** Old-team efficiency is unreliable when the roster is radically replaced. Do **not** mechanically downgrade transfers or regress a reconstructed team to league mean. Build prior-year team contribution *retained by current verified roster*; outgoing players; incoming transfers translated by previous-school opponent, conference, role and minutes; freshmen/historical uncertainty; new head coach/pace separately. Train hierarchical weights and a legal pregame minutes distribution across all programs. Keep stable team prior if continuity is high; use roster-derived prior when trustworthy, otherwise league/conference fallback with wider uncertainty. R017 numeric ESPN player ID normalization must hold (float 12345.0 -> canonical 12345). The absence of archived pregame roster-as-of evidence prevents certified backtest and promotion. Require roster completeness, identity confidence, and source-published-before-tipoff proof.

**F: Opportunity vs shooting variance (NOW for box data).** Separate lagged opponent-adjusted projected turnover rate, opponent-forced turnovers, offensive-rebound rate, 3PA rate, 3P%, 2P%, FT attempts per possession, FT%. Produce expected offense and joint margin/total variance. Models must **not** add Four-Factor terms again to already-adjusted efficiency without residual ablations that prove independent information. After a game, score process components separately from actual points. Smooth observed 3-for-27 through player skill priors rather than treating poor one-game 3P% as new true team strength. Never treat normalized actual makes as a pregame input.

**M: Existing mismatch shrinkage (SCORE R022 FIRST).** Less shrinkage may reduce predicted-margin compression for elite-vs-weak opponents. Test targeted strength-prior specification predeclared for 2026-27 future games, **not** a universal unvalidated +5 points to favorites or any BC-specific correction.

## Same-games test grid
A: frozen team baseline.
B: A + pregame opponent-style pace.
C: A + roster replacement when source PIT is verified.
D: A + independently incremental shot/foul/TO/ORB residual.
E: A with identical means + new predictive variance (proper score, coverage/risk ranking).
F: combination only after ablation evidence. Equal cohorts, model training budget, chronological as-of cuts and identical D1/venue eligibility.

Metrics: home-margin MAE and signed bias, total MAE, team-score MAE, projected-pace MAE, 50/80/95% intervals and proper distributional scores. Break out 0-3 games played, elite/bottom mismatches, neutral, large 3PA, new coaches, high/low returning production, and OT. Paired bootstrap clustered by day/team and chronological cross-fitting. Already-examined 2025-26 is *exploratory only*. Reserve 2026-27 **new** pregame market-blind frozen prospective rows for decision. Never view odds before freezing.

## Stop / promote
- Stop feature family on no robust matched-game gain, PIT uncertainty, identity leakage, double count or sample-coverage illusion.
- Score R022 already-frozen λ challengers once real finals verified. Protect past snapshots and hash.
- Roster correction and error-confidence changes **not production-ready**.
- Document reproducibility, source hashes and failures. Human review and independent prospective improvement required before promotion.

Source BC box: https://www.cbssports.com/college-basketball/gametracker/recap/NCAAB_20261106_UMASS%40BC/
