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

## Validation reports

`lppl validate` is deterministic for a given scan and `config/validation.yaml` (seeded bootstrap,
placebo, null paths and Optuna). Its outputs are saved in `reports/validation/<label>_<run>/`
(CSV tables, `summary.json`, `settings.yaml`, figures) and logged to MLflow (`lppl-validation`).
Archive this folder with the scan it validates.

## Full-period pipeline

```bash
make pipeline-full                            # all steps, PIPE_ID = today's date
PIPE_ID=20261122 bash scripts/full_pipeline.sh validate   # resume from a step
```

Steps: `ingest base tune adopt tuned validate compare short combine`. Scans, validation and the
combined model run on the host (`.venv`, all cores). The tuning studies and `adopt` run in the
`lppl-tune` image, so they are logged to MLflow and Optuna in Postgres. Validation runs with
`--no-mlflow`, because the host environment has no MLflow client. The script writes a timestamped
line per step into the log, so an interrupted run can be resumed from the failed step with the same
`PIPE_ID` (the run of 2026-11-22 was resumed at the tuned validation).

## Robust variant

```bash
bash scripts/robust_variant.sh            # scan (~4.5 h on 4 cores), validate, compare, combine, analyse
bash scripts/robust_variant.sh validate   # resume from a step
```

The extra seeds use their own RNG streams (the window key plus the seed number), so seed 0
reproduces the ordinary scan exactly and the variant is as deterministic as the main scan.
`scripts/robust_compare.py` writes the pre-registered comparison to
`reports/robust_variant_analysis.{md,json}`.

## Short-scale scan and combined model

Both are the last two steps of `make pipeline-full`; `make combine` reruns only the model.
The burst-bubble universe of validation.md §6 used `config/bubbles.yaml` and
`config/bubbles_short.yaml`, retired on 2026-11-23 (`git show ce9d40e:config/<file>`).

* **Scans** are deterministic in the same way as the main scan: the per-window seeds depend only on
  the seed, ticker, date and window length. The short scan's analysis dates are aligned with the
  long scan (both are counted back from the last day with the same step).
* **`lppl combine`** is deterministic for the given pair of runs and `config/combined.yaml`. The
  logistic fits are deterministic and the bootstrap is seeded. Outputs go to
  `reports/combined/<label>_<run>/`, including the full feature table (`features.csv`), so
  results can be re-analysed without the database.
* **Ingestion overwrites snapshots.** Ingesting `config/full.yaml` rewrites the snapshot CSVs of
  the main universe with the longer history from 1985. Scans of `config/default.yaml` still start from their own `data.start`, so their results
  are unaffected. To archive the exact inputs, keep `data/snapshots/` together with the
  `ingestion_log` sha256 values.

Run labels used in the docs:

| label | config | contents |
|---|---|---|
| `default` | `config/default.yaml` | reference scan, 12 instruments |
| `pipeline-base` / `pipeline-tuned` / `pipeline-short` | `pipeline*.yaml` (retired, in commit `ce9d40e`) | 2019+ pipeline, validation.md §5 |
| `bubbles` / `bubbles-short` | `bubbles*.yaml` (retired, in commit `ce9d40e`) | burst-bubble universe, validation.md §6 |
| `full-base` / `full-tuned` / `full-short` | `config/full.yaml` / `config/full_tuned.yaml` / `config/full_short.yaml` | `make pipeline-full`: the 12 instruments from 1987 |
| `full-robust` | `config/full_robust.yaml` | robust variant (3 seeds, persistence), validation.md §8 |

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
