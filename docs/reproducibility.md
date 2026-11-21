# Reproducibility

This page describes how to reproduce every number, table and figure in this
repository, and the mechanisms that make that possible.

## What is controlled

| Source of variation | Control |
|---|---|
| Python and library versions | `requirements.lock` (133 pinned packages, generated inside `python:3.12.8-slim-bookworm` by `make lock`); the image is built from it |
| OS and base image | Dockerfile pins `python:3.12.8-slim-bookworm`; compose pins `postgres:16.4-alpine` |
| Input data | Yahoo data is revised over time (dividends change adjusted prices; corrections happen). Every ingest writes `data/snapshots/<ticker>.csv` in a canonical format, with its sha256 recorded in `ingestion_log`. `--from-snapshot` rebuilds the database from those exact files without network access |
| Randomness | a master `seed`, and one `SeedSequence` per (ticker, t2, window). Independent of `n_jobs` and execution order (tested) |
| Configuration | YAML under git. The full effective config plus git SHA plus package version are stored per run in `fit_runs` |
| Code | `git_sha` in `fit_runs`; commit before running experiments you want to cite |

**Tested.** For this repository the `case:tsla2021` run (2,106 fits) was deleted and re-run
from scratch in the container. `lppl verify` then matched all 2,751 reference rows
(maximum difference 4.6e-11, which is CSV float formatting).

Floating-point results are bit-identical for the same image on the same CPU architecture.
Across architectures (x86-64 vs arm64), or with different BLAS builds, tiny differences in the
last bits can occasionally flip a qualification decision at a filter boundary. `lppl verify`
reports any such differences explicitly.

## Procedure A: reproduce from snapshots (exact)

Requirements: Docker, make, and the `data/snapshots/` directory plus `reports/reference.csv`
from the original run (archive them alongside the repository; `data/` is git-ignored by default
because of its size and licensing).

```bash
git checkout <commit>            # the git_sha recorded in fit_runs
cp .env.example .env
make clean-db                    # optional: start from an empty database
make reproduce                   # build → up → ingest-snapshot → scan → case-studies
                                 #   → report → notebooks → verify
```

`make verify` compares every `(label, ticker, date)` row of `pos_conf` and `neg_conf` with
`reports/reference.csv` and fails on any difference larger than `1e-9`:

```
reference rows: 2751, missing in DB: 0, differing (> 1e-09): 0, max abs diff: 4.62e-11
OK: results reproduced
```

To check the snapshot files themselves before running:

```bash
shasum -a 256 data/snapshots/*.csv
docker compose run --rm --entrypoint "" app python -c \
  "from lppl.data import db; import pandas as pd; \
   print(pd.read_sql('select ticker, sha256, fetched_at from ingestion_log order by id', db.get_engine()))"
```

## Procedure B: re-run on fresh data (same method, new data)

```bash
make build up ingest scan case-studies report notebooks
docker compose run --rm app export --out reports/reference.csv   # new reference
```

The numbers will differ from the published ones wherever Yahoo has revised history, and
because there are new dates. To keep the result reproducible, archive `data/snapshots/`,
`reports/reference.csv` and the commit SHA together.

## Procedure C: host Python instead of the app container

```bash
python3.12 -m venv .venv && source .venv/bin/activate
pip install -r requirements.lock && pip install --no-deps -e .
python -m ipykernel install --sys-prefix --name lppl
make up
lppl ingest --from-snapshot
lppl scan --label default
# the case-studies commands from the Makefile, using `lppl` directly
lppl verify
```

## Run bookkeeping

```bash
lppl runs                        # all runs with label, status, seed, git SHA
lppl report --label default      # latest default run
lppl report --run-id <uuid>      # any specific run
lppl delete-run <uuid>           # remove a run (cascades to fits and indicator)
```

Notebook 03 uses the latest finished `default` run and all finished `case:*` runs.
To pin a notebook to a specific run, set `RUN = "<uuid>"` in its first code cell.

## Archive checklist for a published result

- [ ] commit SHA (`git rev-parse HEAD`), with a clean working tree
- [ ] `data/snapshots/*.csv`
- [ ] `reports/reference.csv` (`lppl export`)
- [ ] `requirements.lock` (in git)
- [ ] the executed notebooks (in git)
- [ ] the output of `lppl runs`
