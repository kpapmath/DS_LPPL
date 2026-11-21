# Quickstart

From an empty checkout to a bubble dashboard. You only need **Docker**
(Docker Desktop, or Engine with Compose v2) and **make**. A local Python install is
optional.

## 1. Configure

```bash
cp .env.example .env      # DB credentials, host port (default 5442), Jupyter token
```

If port 5442 is taken on your machine, change `POSTGRES_PORT` in `.env`.

## 2. Build and start the database

```bash
make build                # builds lppl-app:0.1.0 from requirements.lock
make up                   # starts postgres:16.4 and waits until it is healthy
```

## 3. Load the data

```bash
make ingest               # Yahoo Finance -> Postgres, plus data/snapshots/*.csv
```

This takes about 30 s and loads about 80,000 daily rows for 12 tickers. If you already have
`data/snapshots/` (from a colleague or an archive), use `make ingest-snapshot`
to rebuild the database offline from exactly those files.

## 4. Run the model

```bash
make scan-quick           # ~5 min on 2 CPUs: monthly analysis dates since mid-2024
make report               # latest reading per ticker, also written to reports/latest.csv
```

For the full experiment (an analysis date every 2 weeks since 2019, 26 window lengths)
and the historical case studies (dot-com 2000, the 2007 peak, Tesla 2021, NVIDIA 2024):

```bash
make scan                 # ~1 h on 2 CPUs (scales with cores: n_jobs=-1)
make case-studies         # ~15 min
```

## 5. Look at the results

```bash
make jupyter              # open http://localhost:8888, token from .env (default: lppl)
```

| Notebook | Shows |
|---|---|
| `01_data_overview` | coverage, provenance hashes, prices, return statistics |
| `02_single_fit_walkthrough` | one GA calibration step by step, fit, residuals, filter checklist |
| `03_bubble_indicator_dashboard` | current readings, signed-confidence heatmap, per-ticker charts, case studies |
| `04_ga_diagnostics_and_benchmark` | GA vs. differential evolution vs. Nelder–Mead, seed stability, hyperparameters |

To execute all notebooks headless (inside the container): `make notebooks`.

## 6. Tune the model (optional)

```bash
make tune-build tune-up             # MLflow http://localhost:5050 · Optuna dashboard http://localhost:8081
make tune-indicator ITRIALS=300     # detector study on the stored default scan (~3 min)
make tune-ga-ray TRIALS=30 JOBS=2   # GA study with Ray Tune + ASHA (benchmark built on first use)
```

Each study writes `data/tuning/<study>/tuned_config.yaml` and logs everything to MLflow.
**Read the validation metrics before adopting anything**: see [docs/tuning.md](docs/tuning.md) §9.

## Without Docker for the app (host Python)

Postgres still runs in Docker, and the code runs in a local virtual environment:

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.lock && pip install --no-deps -e .
python -m ipykernel install --sys-prefix --name lppl     # kernel used by the notebooks
make up
lppl ingest && lppl scan -c config/quick.yaml --label quick && lppl report --label quick
pytest                     # unit tests
```

The CLI reads `.env`, so `localhost:5442` is used automatically.

## Single fit, no scan

```bash
docker compose run --rm app fit-one NVDA --end 2024-06-18 --window 500
```

This prints the fitted parameters, the critical-time date, and every qualification check.

## Clean up

```bash
make down                  # stop containers, keep the data volume
make clean-db              # stop and DELETE the database volume
```

Next: [docs/reproducibility.md](docs/reproducibility.md) explains how to reproduce the
published numbers exactly.
