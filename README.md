# 2026.27.CBB — Model Laboratory v0.1

A deterministic, market-blind laboratory for chronological NCAA men's basketball research.
The **2026 season is sealed**: normal data and backtest commands reject it. This repository stops
at the Milestone 5 gate; it does not perform feature search or inspect the holdout.

## Where are the project files?

The laboratory files are committed on the feature branch used by the pull request. GitHub's
default-branch view will continue to show only the original `.gitkeep` until that pull request is
merged. Review the pull request's **Files changed** tab to inspect the package, tests, configs, and
workflows before merging it.

For a normal local clone, fetch and check out the pull-request branch (replace `<branch>` with the
branch name shown by GitHub):

```bash
git fetch origin
git switch <branch>
git status --short --branch
git ls-tree -r --name-only HEAD
```

After review, merge the pull request in GitHub. The files will then appear on the default branch;
large downloaded NCAA datasets will intentionally remain absent because `.gitignore` keeps the
data cache out of Git history.

## Install and verify

```bash
python -m pip install -r requirements.lock
python -m pip install --no-deps -e .
ruff check src tests
pytest --cov=cbb
```

## What to do after merging

1. Open the repository's **Actions** tab and confirm the latest **CI** run is green. Do not start
   historical data work while the foundation tests are failing.
2. Clone the default branch and run `make install`, then `make lint` and `make test`. This uses the
   committed dependency lock so local development and CI install the same versions.
3. Configure and validate authoritative SportsDataverse 2021--2025 source assets before running a
   frozen backtest. The current sync command accepts one explicit source URL at a time; it is not
   yet the complete multi-family canonical data builder described by the data architecture
   addendum.

Do not run search/P1 work, publish `CBB-DATA-v1`, or use Supabase as the historical model source
until the canonical corpus passes integrity checks and reproduces the frozen R1/R2 targets.

The latest attempted Milestone 5 execution is recorded in
[`results/summaries/milestone5_blocker.md`](results/summaries/milestone5_blocker.md). It records
the precise network and dependency-access failures; it is not a successful reproduction report.

## Data flow

Raw SportsDataverse files are downloaded once, checksummed into `data/manifests`, validated,
then compiled to canonical team-game rows. Large datasets remain in the cache, not Git. Sync a
URL explicitly (the upstream path is intentionally configurable):

```bash
python -m cbb.data.sync --dataset team_box --season 2025 --url URL --output data/cache/team_box_2025.csv
```

Each manifest records URL, retrieval time, byte hash, row count, and schema version. Canonical
rules use `totalTurnovers` (never add `teamTurnovers`) and the frozen conservative possession
formula `FGA - ORB + TOV + 0.44*FTA`. Invalid rows fail before model-ready output.

## Frozen backtest

```bash
python -m cbb.backtest.run --config configs/production/r1_r2.yaml --input canonical.parquet --output results/run
```

The runner emits predictions, metrics (overall, season, and maturity buckets), hashes, and an
append-only experiment JSON. Without the independently synchronized historical input, regression
targets are documented but not falsely claimed as reproduced.

## Governance and GitHub

Blind predictions precede any future market reveal. Same-date games are predicted before any
same-date update. Workflows under `.github/workflows` run CI and provide manual backtest,
research, search-stub, regression, and protected sealed-evaluation entry points. The sealed job
requires an explicit environment and acknowledgement; ordinary code still refuses season 2026.

## Supabase boundary

No Supabase client or schema exists in v0.1: this is an offline reproducibility foundation.
If persistence is added, use separate raw/canonical tables, append-only versioned predictions,
foreign-key indexes, RLS on every exposed table, owner-only policies for bets, and server-only
service-role credentials. Market storage must remain outside blind feature construction.
