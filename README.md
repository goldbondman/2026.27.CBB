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
python -m pip install -e '.[dev]'
pytest
```

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
