# LPPL Bubble Detection with a Genetic Algorithm

Detect speculative bubbles in equity prices with the **Log-Periodic Power Law
Singularity (LPPLS)** model of Johansen, Ledoit and Sornette. The model is calibrated
with a **genetic algorithm (GA)**, and the data and results are stored in
**PostgreSQL**, which runs in Docker.

Universe (12 instruments, daily data from Yahoo Finance since 1995):

| Group | Tickers |
|---|---|
| Indices | `^GSPC` (S&P 500), `^IXIC` (NASDAQ Composite) |
| Magnificent Seven | `AAPL` `MSFT` `GOOGL` `AMZN` `NVDA` `META` `TSLA` |
| Extra | `AVGO` (Broadcom), `JPM` (JPMorgan Chase, a non-tech control), `NFLX` (Netflix) |

> **Disclaimer.** This is a research and teaching tool. LPPLS signals are
> probabilistic, often wrong about timing, and are **not investment advice**.

---

## What it does

```
Yahoo Finance ──► lppl ingest ──► PostgreSQL ◄── lppl scan ◄── GA + LPPLS + filters
                     │               │  ▲
                     ▼               │  └── fit_runs / lppl_fits / bubble_indicator
             data/snapshots/*.csv    ▼
             (sha256-hashed)     notebooks/ · lppl report · lppl verify
```

1. **Ingest**: download daily OHLCV and adjusted close, validate the data, upsert it into
   Postgres, and save a hashed CSV snapshot of each ticker so the database can be rebuilt offline.
2. **Fit**: for each analysis date `t2` and each window length from 125 to 750 trading
   days, find `(tc, m, ω)` with the GA and solve `(A, B, C1, C2)` exactly by least squares.
3. **Qualify**: keep only fits whose parameters fall in the stylised bubble ranges
   (Sornette et al. 2015).
4. **Indicate**: the *positive* (bubble) and *negative* (anti-bubble) **confidence**
   is the share of windows with a qualified fit of that sign. The spread of qualified `tc` values
   gives a window for the critical time.
5. **Explore** the results in four notebooks, or with `lppl report`.
6. **Tune** the GA and the detector with Optuna, Ray Tune and MLflow, all in Docker
   ([docs/tuning.md](docs/tuning.md)).
7. **Validate** accuracy with `lppl validate`: significance tests, a placebo, walk-forward,
   bubble-free null model and a backtest ([docs/validation.md](docs/validation.md)).

## Repository layout

| Path | Contents |
|---|---|
| `src/lppl/model/` | `lppls.py` (model and linear solve), `filters.py`, `calibrate.py`, `indicator.py` |
| `src/lppl/optim/` | `ga.py` (the genetic algorithm), `baseline.py` (differential evolution and Nelder–Mead benchmarks) |
| `src/lppl/data/` | `yahoo.py` (download, validation, snapshots), `db.py` (schema and data access) |
| `src/lppl/` | `pipeline.py`, `cli.py` (`lppl …`), `config.py`, `viz.py`, `validation.py` (`lppl validate`) |
| `src/lppl/tuning/` | tuning studies: `settings.py` (search spaces), `ga_benchmark.py`, `detection.py`, `runners.py` (Optuna / Ray), `tracking.py` (MLflow), `cli.py` (`lppl tune …`) |
| `config/` | `default.yaml` (full experiment), `quick.yaml` (a few minutes), `tuning.yaml` (tuning studies), `validation.yaml` (validation suite) |
| `notebooks/` | 01 data overview · 02 single-fit walkthrough · 03 dashboard · 04 GA benchmark · 05 validation |
| `tests/` | unit tests (no network or DB needed) and integration tests (`-m integration`) |
| `docs/` | methodology, model presentation, architecture, data dictionary, configuration, reproducibility |
| `docker/`, `docker-compose.yml`, `Makefile` | containers (app, tuning, MLflow, optuna-dashboard) and one-command workflows |

## Getting started

See **[QUICKSTART.md](QUICKSTART.md)**. The short version:

```bash
cp .env.example .env
make build up ingest scan-quick report
make jupyter            # http://localhost:8888 (token: lppl)
```

## Documentation

- [docs/how_the_model_works.md](docs/how_the_model_works.md): **start here**, a plain-language guide to how the model works, how it is validated and its prediction horizon, with examples from the current run.
- [docs/methodology.md](docs/methodology.md): extended review of LPPLS, its estimation, the GA, and the criticisms.
- [docs/validation.md](docs/validation.md): how accuracy is validated (`lppl validate`, notebook 05), and the results.
- [docs/tuning.md](docs/tuning.md): hyperparameter-tuning methodology (Optuna, Ray Tune, MLflow), the procedure and the pitfalls.
- [docs/model_presentation.md](docs/model_presentation.md): presentation of the model and the results.
- [docs/architecture.md](docs/architecture.md): components, data flow and design decisions.
- [docs/data_dictionary.md](docs/data_dictionary.md): database tables and columns.
- [docs/configuration.md](docs/configuration.md): every setting in `config/*.yaml` and `.env`.
- [docs/reproducibility.md](docs/reproducibility.md): how to reproduce every number exactly.
- [docs/PLAN.md](docs/PLAN.md): the original implementation plan.

## Command line

```text
lppl init-db                        create tables (idempotent)
lppl ingest [--from-snapshot]       Yahoo (or local snapshots) -> Postgres
lppl scan  [-c cfg] [-t T1,T2] [--start --end --step --label]
lppl report [--label default] [--csv path]
lppl requalify [-c cfg] [--run-id ID]     re-apply filters to stored fits
lppl runs | delete-run ID
lppl export --out reports/reference.csv
lppl verify --reference reports/reference.csv
lppl validate [--label default] [--null-paths N] [--no-walk-forward]
lppl fit-one NVDA --end 2024-06-18 --window 500

lppl tune init-stores | build-benchmark
lppl tune run ga|indicator --backend optuna|ray --trials N --jobs J [--name STUDY]
lppl tune best STUDY | export-best STUDY --out config/tuned.yaml
lppl tune evaluate-config -c config/tuned.yaml
```

Inside Docker, use `docker compose run --rm app <command>`, or the `make` targets (`make help`).

## Tests

```bash
make test               # 104 unit tests, host venv, no DB or network
make test-integration   # DB tests against the compose Postgres
make test-docker        # everything, inside the container
```

## License

MIT.
