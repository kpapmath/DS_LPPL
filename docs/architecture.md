# Architecture

## Components

```
┌──────────────────────────── docker compose (project "lppl") ───────────────────────────┐
│                                                                                         │
│  db  postgres:16.4-alpine ── volume pgdata ── host port ${POSTGRES_PORT:-5442}           │
│   ▲                                                                                     │
│   │ SQLAlchemy 2 / psycopg 3                                                            │
│   │                                                                                     │
│  app  lppl-app:0.1.0  (python 3.12, requirements.lock)      profile: app                │
│   │   entrypoint `lppl` → ingest · scan · report · export · verify · fit-one            │
│   │   bind mounts: ./data  ./config(ro)  ./reports  ./notebooks                         │
│   │                                                                                     │
│  jupyter  same image, JupyterLab :8888, kernel "lppl"        profile: notebooks         │
│  adminer  DB web UI :8080                                    profile: tools             │
│                                                                                         │
│  mlflow   lppl-tune:0.1.0, tracking server :5050 (DBs mlflow + optuna)  profile: tune   │
│  tune     lppl-tune:0.1.0, `lppl tune …` (Optuna / Ray Tune), shm 2 GB  profile: tune   │
│  optuna-dashboard  :8081                                               profile: tune   │
└─────────────────────────────────────────────────────────────────────────────────────────┘
```

## Package structure (`src/lppl`)

| Module | Responsibility | Depends on |
|---|---|---|
| `model/lppls.py` | time conversion, the LPPLS function, vectorised linear solve, `LPPLSFit` | numpy, pandas |
| `optim/ga.py` | generic batched real-coded GA (`minimize`) and its operators | numpy, scipy (polish) |
| `optim/baseline.py` | differential evolution and multistart Nelder–Mead benchmarks | scipy |
| `model/calibrate.py` | `SearchBounds`, objective, `fit_window(method=ga|de|nm)` | lppls, optim |
| `model/filters.py` | qualification conditions | lppls |
| `model/indicator.py` | window grid, deterministic per-window RNG, `scan_date`, `aggregate`, `scan_series` (joblib) | calibrate, filters |
| `data/yahoo.py` | yfinance download with retry, normalisation, validation, CSV snapshots and sha256 | yfinance, pandas |
| `data/db.py` | schema (SQLAlchemy Core), idempotent upserts, loaders | SQLAlchemy |
| `config.py` | pydantic models for the YAML config; DB settings from env and `.env` | pydantic(-settings) |
| `pipeline.py` | orchestration: `ingest`, `scan`, `report`, `resolve_run` | everything above |
| `cli.py` | Typer CLI | pipeline |
| `viz.py` | matplotlib style and chart helpers for the notebooks | matplotlib |
| `validation.py` | validation suite (`lppl validate`, notebook 05): skill, calibration, block bootstrap, placebo, robustness grid, timing, walk-forward, null model, backtest | tuning.detection (labels, AP), joblib; optuna optional |
| `tuning/*` | hyperparameter studies, see [tuning.md](tuning.md) | optuna, ray[tune], mlflow (lazy imports; only in the `lppl-tune` image) |

The model and optimiser layers have **no I/O dependencies**. They can be unit-tested and
reused without a database.

## Data flow

1. `lppl ingest`: `yahoo.download` → `normalize` → `validate` → `save_snapshot`
   (`data/snapshots/<ticker>.csv` plus its sha256) → `db.upsert_prices` → `db.log_ingestion`.
   With `--from-snapshot`, the download step is replaced by `load_snapshot`.
2. `lppl scan`: `db.create_run` (the config JSON, seed, git SHA and version are frozen) → for each
   ticker: `db.load_prices` → `indicator.scan_series` (joblib parallel over analysis
   dates; each date fits every window length) → `db.write_fits`, `db.write_indicator` →
   `db.finish_run` (or `failed` if an exception occurs).
3. Notebooks and `lppl report` read `bubble_indicator` and `lppl_fits` for one run
   (default: the latest finished run labelled `default`).

## Design decisions

| Decision | Alternatives | Reason |
|---|---|---|
| Own NumPy GA | DEAP, PyGAD | transparent, about 150 lines, fully testable, batched fitness, no extra dependency |
| Slaved linear parameters | fitting all 7 by GA | reduces the search to 3D and gives a much smoother landscape (Filimonov & Sornette 2013) |
| Normal equations built from sums plus scaling | `lstsq` / QR per candidate | about 2× faster per generation; accuracy checked against `lstsq` in tests |
| Calendar decimal years | trading-day index | `tc` maps directly to a calendar date |
| Schema in SQLAlchemy metadata (`init-db`) | SQL files in `docker-entrypoint-initdb.d` | one source of truth; works the same for host, container and tests (isolated schema) |
| Upsert everywhere | insert-only | ingest and scan can be re-run safely; Yahoo revisions overwrite old values |
| Full re-download on ingest | incremental | about 80k rows is small, and adjusted prices change retroactively after dividends |
| Run table with labels | separate tables per experiment | many experiments (default, quick, case studies) live side by side and are always traceable |
| Per-window `SeedSequence` | one global RNG | determinism regardless of parallelism and execution order |
| Separate `lppl` Jupyter kernel | the default `python3` kernel | a user-level `python3` kernelspec on the host would otherwise shadow the venv |

## Performance

On a 2-CPU Docker VM, one window fit takes about 90 ms with defaults (about 1,300 objective
evaluations). The default scan has 12 tickers, about 200 analysis dates each and 26 windows,
roughly 60,000 fits, and takes 1–1.5 h. Runtime scales close to linearly with cores (`scan.n_jobs`).
The main levers are `scan.step`, `windows.step` and `ga.population_size`.
