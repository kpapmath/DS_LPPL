# Implementation Plan: LPPL Bubble Detection with Genetic Algorithm Calibration

Status: **implemented** (phases 0–8; phase 9, CI, is not set up). This is the original plan; the
places where the implementation differs from it are listed directly below and documented in
`docs/methodology.md` and `docs/architecture.md`.

### Deviations from the plan during implementation

| Plan | Implemented | Why |
|---|---|---|
| History from 2000-01-01 | **1995-01-01** | the dot-com case study needs up to 3 years of history before the March 2000 peak |
| `tc` search `[t2 − 0.2·dt, t2 + 0.2·dt]` | search `(t2 + 1 day, t2 + 0.5·dt]`; *filter* `[t2 − 0.05·dt, t2 + 0.2·dt]` | the model is undefined for `tc ≤ t` within the window; the search is wider than the filter so that fits stuck at a bound are rejected |
| Oscillation formula with `(t2 − t1)` | `O = (ω/π)·ln((tc − t1)/(tc − t2)) ≥ 2.5` | the correct form (Sornette et al. 2015) |
| `ω` search `[2, 25]` | search `[1, 50]`, filter `[2, 25]` | same reason as for `tc` |
| Schema in `docker/initdb/01_schema.sql` | SQLAlchemy metadata + `lppl init-db` | one source of truth for the host, the container and the test schema |
| Incremental ingestion | full re-download plus upsert | adjusted prices change retroactively; the data is small |
| Outputs stripped with `nbstripout` | notebooks are committed **executed** | they are the summary deliverable; `make notebooks` regenerates them |
| GA population and generations to be decided | 40 / 30 + Nelder–Mead polish | tuning in `docs/methodology.md` §5 and notebook 04 |
| Host DB port 5432 | **5442** | 5432 and 5433 are commonly taken on developer machines |
| Filter `0.01 ≤ m ≤ 0.99` | `0.1 ≤ m ≤ 0.9` plus `tc > t2 + 3 days` | 24 % of qualified fits were pinned at the search bounds (methodology §6) |
| — | `lppl requalify` | re-applies filters to stored fits without refitting |
| — (added later) | hyperparameter tuning with Optuna, Ray Tune and MLflow, dockerised | see `docs/tuning.md` |
| — (added later) | `lppl validate` + notebook 05: significance, placebo, walk-forward, null model, backtest | see `docs/validation.md` |
| CI (phase 9) | not set up | optional in the plan; `make test-docker` runs the full suite in the container |

---

## 1. Objective

Build a reproducible pipeline that:

1. Downloads daily price data from Yahoo Finance for:
   - **Indices:** S&P 500 (`^GSPC`), NASDAQ Composite (`^IXIC`). NASDAQ-100 (`^NDX`) is an optional extra.
   - **Magnificent Seven:** `AAPL`, `MSFT`, `GOOGL`, `AMZN`, `NVDA`, `META`, `TSLA`
   - **Three additional names** (proposed, easy to change in config): `AVGO` (AI/semis), `JPM` (financials, a non-tech control), `NFLX` (a growth stock with a past boom-and-bust)
2. Stores the data in **PostgreSQL** running in a **Docker container**.
3. Fits the **Log-Periodic Power Law (LPPL / LPPLS)** model with a **Genetic Algorithm (GA)** to find the nonlinear parameters, and solves the linear parameters in closed form.
4. Produces bubble indicators (critical time `t_c`, confidence over many windows, filtered "qualified" fits).
5. Ships documentation, notebooks, unit tests, and a step-by-step reproducibility guide.

---

## 2. Model Summary (what gets implemented)

### 2.1 LPPLS equation (Filimonov & Sornette 2013 form)

```
ln p(t) = A + B (t_c − t)^m + C1 (t_c − t)^m cos(ω ln(t_c − t)) + C2 (t_c − t)^m sin(ω ln(t_c − t))
```

- **Nonlinear parameters:** `t_c` (critical time), `m` (power-law exponent), `ω` (log-periodic angular frequency). The GA searches over these.
- **Linear parameters:** `A, B, C1, C2`. For fixed (`t_c, m, ω`) they are found exactly by ordinary least squares (a 4×4 normal-equation solve). This "slaving" of the linear parameters turns a 7-parameter problem into a 3-parameter search, which makes the GA fast and stable.
- Recover `C = sqrt(C1² + C2²)` and `φ` for reporting.

### 2.2 Search bounds and qualification filters (from the literature, configurable)

| Parameter | Search range | Qualification filter |
|---|---|---|
| `t_c` | `[t2 − 0.2·dt, t2 + 0.2·dt]` (dt = window length) | same |
| `m` | `[0.01, 1.0]` | `0.01 ≤ m ≤ 0.99` (or tighter `0.1–0.9`) |
| `ω` | `[2, 25]` | `2 ≤ ω ≤ 25` |
| Damping | — | `D = m·|B| / (ω·|C|) ≥ 0.5` |
| Oscillations | — | `O = (ω/π)·ln((t_c − t1)/(t2 − t1)) ≥ 2.5` |
| Sign of B | — | `B < 0` for a positive (upward) bubble, `B > 0` for a negative bubble |
| Residuals | — | Relative error of the fit below a threshold; optional Lomb periodogram / unit-root check |

### 2.3 Genetic Algorithm design

- **Chromosome:** real-valued vector `(t_c, m, ω)`.
- **Fitness:** sum of squared residuals (SSE) of `ln p` after the linear OLS solve. Lower is better. Singular or ill-conditioned solves get a penalty.
- **Initialisation:** uniform inside the bounds (Latin hypercube as an option).
- **Selection:** tournament selection (k = 3).
- **Crossover:** blend / BLX-α or simulated binary crossover (SBX).
- **Mutation:** Gaussian mutation with a step size that shrinks over generations; clip results to the bounds.
- **Elitism:** keep the top N individuals each generation.
- **Stopping rule:** maximum number of generations, or no improvement for K generations.
- **Polish (optional):** a final Nelder–Mead / L-BFGS-B refinement from the best individual (a hybrid GA).
- **Reproducibility:** every run takes an explicit `numpy.random.Generator` seed.
- **Implementation:** a small NumPy GA in our own code (transparent and easy to test). Fitness is vectorised across the population. Windows run in parallel with `joblib`. As a benchmark, compare against `scipy.optimize.differential_evolution`.

### 2.4 Bubble indicator (multi-window scanning)

For each analysis date `t2`:
- Fit the model over many windows, with `t1` ranging from `t2 − 750` to `t2 − 125` trading days in steps of about 5–10 days.
- **LPPLS confidence indicator** = the fraction of windows whose fit passes all qualification filters, given a sign: positive for bubbles, negative for anti-bubbles.
- Also report the distribution of `t_c` across the qualified fits (median and interquartile range).

Rolling this over time gives a confidence time series per ticker. Those series go into the database and into the notebooks.

---

## 3. Architecture and Repository Layout

```
DS_LPPL/
├── README.md                    # overview, features, architecture diagram, links
├── QUICKSTART.md                # running in about 5 minutes
├── pyproject.toml               # package + dependencies (pinned) + tool config
├── requirements.lock            # fully pinned for reproducibility (pip-tools / uv)
├── .env.example                 # DB credentials template
├── Makefile                     # make up / ingest / fit / test / notebooks
├── docker-compose.yml           # postgres + app (+ optional jupyter, adminer)
├── docker/
│   ├── app.Dockerfile
│   └── initdb/01_schema.sql     # schema created when the container starts
├── config/
│   └── default.yaml             # tickers, dates, GA hyperparams, filters, seeds
├── src/lppl/
│   ├── __init__.py
│   ├── config.py                # pydantic settings (YAML + env)
│   ├── data/
│   │   ├── yahoo.py             # yfinance download, retry, validation
│   │   └── db.py                # SQLAlchemy engine, upsert, read helpers
│   ├── model/
│   │   ├── lppls.py             # model function, linear OLS solve, SSE
│   │   ├── filters.py           # qualification conditions (damping, oscillations…)
│   │   └── indicator.py         # multi-window scan + confidence indicator
│   ├── optim/
│   │   ├── ga.py                # genetic algorithm
│   │   └── baseline.py          # differential evolution / multi-start NM benchmark
│   ├── pipeline.py              # orchestration: ingest → fit → store
│   └── cli.py                   # `lppl ingest`, `lppl fit`, `lppl scan`, `lppl report`
├── notebooks/
│   ├── 01_data_overview.ipynb
│   ├── 02_single_fit_walkthrough.ipynb
│   ├── 03_bubble_indicator_dashboard.ipynb
│   └── 04_ga_diagnostics_and_benchmark.ipynb
├── tests/
│   ├── conftest.py
│   ├── test_lppls.py
│   ├── test_filters.py
│   ├── test_ga.py
│   ├── test_indicator.py
│   ├── test_yahoo.py            # mocked network
│   └── test_db.py               # integration (testcontainers or the compose DB)
└── docs/
    ├── PLAN.md                  # this file
    ├── methodology.md           # extended review (see §7)
    ├── model_presentation.md    # presentation of the model and results
    ├── architecture.md
    ├── data_dictionary.md       # DB schema + column meanings
    ├── configuration.md
    ├── reproducibility.md       # full procedure (see §9)
    └── references.bib
```

**Stack:** Python 3.12 (in the container; the local virtual environment is 3.13 and is also supported), NumPy, pandas, SciPy, yfinance, SQLAlchemy 2 + psycopg 3, pydantic-settings, joblib, matplotlib/plotly, Typer (CLI), pytest + hypothesis + pytest-cov, ruff, mypy.

---

## 4. Data Layer

### 4.1 Ingestion (`src/lppl/data/yahoo.py`)
- Use `yfinance.download(tickers, start, end, auto_adjust=False, actions=True)`. Store both `Close` and `Adj Close`, and fit on `Adj Close`.
- Default history: **2000-01-01 → today**. This covers the dot-com bubble, 2008, the 2020 crash, and the 2021 and 2023–25 AI rallies.
- Validate the data: no duplicate dates, positive prices, gaps flagged, prices that do not move over time flagged.
- Retry with backoff. Cache raw CSV/Parquet snapshots in `data/raw/` with a SHA-256 hash, so a run can be reproduced even if Yahoo revises its data.
- Ingestion is incremental: only dates after the latest stored date are fetched.

### 4.2 PostgreSQL schema (`docker/initdb/01_schema.sql`)

| Table | Purpose | Key columns |
|---|---|---|
| `instruments` | ticker metadata | `ticker` PK, `name`, `asset_type` (index/equity), `group` (index/mag7/extra) |
| `prices` | daily OHLCV | (`ticker`, `date`) PK, `open, high, low, close, adj_close, volume` |
| `ingestion_log` | data provenance | `run_id`, `ticker`, `fetched_at`, `rows`, `source_hash`, `yfinance_version` |
| `fit_runs` | one row per scan or experiment | `run_id` PK, `config_json`, `git_sha`, `seed`, `started_at`, `finished_at` |
| `lppl_fits` | one row per window fit | `run_id`, `ticker`, `t1`, `t2`, `tc`, `m`, `omega`, `A, B, C1, C2`, `sse`, `rmse`, `damping`, `n_osc`, `qualified`, `sign`, `generations`, `elapsed_ms` |
| `bubble_indicator` | aggregated per date | `run_id`, `ticker`, `date`, `pos_conf`, `neg_conf`, `tc_median`, `tc_q25`, `tc_q75`, `n_windows` |

Indexes go on (`ticker`, `date`) and (`run_id`, `ticker`, `t2`).

### 4.3 Containers (`docker-compose.yml`)
- `db`: `postgres:16-alpine` with a named volume, a healthcheck, and the init scripts mounted.
- `app`: the project image. It runs the CLI (`lppl ingest`, `lppl scan`) and starts only after `db` is healthy.
- `jupyter` (profile `notebooks`): JupyterLab with the package installed and a connection to `db`.
- `adminer` (optional profile): a web UI for browsing the database.
- Credentials come from `.env`. `.env.example` is committed; `.env` is git-ignored.

---

## 5. Implementation Phases and Milestones

| # | Phase | Deliverables | Done when |
|---|---|---|---|
| 0 | Scaffolding | `pyproject.toml`, `.gitignore` (including `.idea/`, `.venv/`, `.env`, `data/`), ruff/mypy/pytest config, Makefile, pre-commit | `make lint test` runs green on an empty package |
| 1 | Containers + DB | compose file, Dockerfile, schema SQL, `db.py` | `make up` starts the database; the schema exists; the connection test passes |
| 2 | Ingestion | `yahoo.py`, CLI `lppl ingest` | All 12 tickers are in `prices`; the provenance log is filled; re-running inserts no duplicates |
| 3 | LPPLS core | `lppls.py`, `filters.py` | Unit tests recover known parameters from synthetic series |
| 4 | Genetic algorithm | `ga.py`, `baseline.py` | GA recovers synthetic `t_c` within a tolerance in ≥ 95 % of seeded trials; same seed gives an identical result |
| 5 | Indicator + pipeline | `indicator.py`, `pipeline.py`, CLI `lppl scan` | Confidence series are stored for all tickers; parallel runs work |
| 6 | Notebooks | 4 notebooks (§8) | Each runs top to bottom against the database with no manual steps (`nbmake` in CI) |
| 7 | Documentation | README, QUICKSTART, docs/* | A newcomer can reproduce the results using only the docs |
| 8 | Validation study | historical case studies and false-positive analysis, in `model_presentation.md` | Results are documented with figures |
| 9 | CI (optional) | GitHub Actions: lint, unit tests, integration tests with a Postgres service, nbmake | Pipeline passes |

---

## 6. Testing Strategy (`tests/`)

**Unit tests (fast, no network, no DB):**
- `test_lppls.py`: the model output matches a hand calculation; the linear OLS solve recovers `A, B, C1, C2` exactly on noise-free synthetic data; the function handles `t ≥ t_c` (masking or NaN) safely; SSE is non-negative.
- `test_filters.py`: each qualification condition is tested at its boundary values; sign classification is correct.
- `test_ga.py`: seeded determinism; results always stay inside the bounds; the best fitness never gets worse with elitism; it converges on a convex toy function (sphere) and on synthetic LPPLS data with noise; crossover and mutation operators are tested on their own.
- `test_indicator.py`: the window grid is generated correctly; confidence stays within [0, 1]; a synthetic bubble gives high positive confidence near `t_c`; a random walk gives low confidence.
- Property-based tests with `hypothesis` for the operators and bounds handling.

**Integration tests (marked `@pytest.mark.integration`):**
- `test_yahoo.py`: network calls are mocked with recorded fixtures (and one optional live smoke test).
- `test_db.py`: upsert idempotency and round-trip reads, using `testcontainers[postgres]` or the compose database.

**Coverage goal:** ≥ 85 % on `src/lppl/model` and `src/lppl/optim`.

---

## 7. Extended Methodology Review (`docs/methodology.md`)

Planned outline:
1. **Background:** rational expectation bubbles (Blanchard–Watson), the Johansen–Ledoit–Sornette (JLS) model, herding and imitation, positive feedback, and the hazard rate.
2. **Derivation:** from the hazard rate to the LPPL price dynamics; the role of discrete scale invariance and log-periodicity.
3. **Formulations:** the original 7-parameter LPPL, the Filimonov–Sornette linearisation (LPPLS), and the extensions MPL/LPPLS-OR and the Lagrange regularisation of `t1` (Demos & Sornette).
4. **Estimation:** why the cost surface has many local minima; slaving of the linear parameters; comparison of optimisers (Levenberg–Marquardt, Nelder–Mead/tabu search, GA, differential evolution, CMA-ES); why we use a GA and how its hyperparameters were chosen.
5. **Qualification filters and the confidence indicator** (Sornette et al., Financial Crisis Observatory).
6. **Statistical validity and criticism:** sloppy parameters, the poorly identified `t_c`, sensitivity to the window, multiple testing and data snooping, false positives, critiques such as Feigenbaum (2001) and Bree & Joseph (2013), and how this project addresses them (a profile likelihood for `t_c`, bootstrap confidence intervals, random-walk null tests).
7. **Limitations and ethical caveats.** This is not investment advice.
8. **References** (`references.bib`).

**Model presentation (`docs/model_presentation.md`):** a short presentation-style document covering intuition, the equation, the GA flowchart, case studies (dot-com 2000 on NASDAQ; 2007 on the S&P 500; TSLA 2021; NVDA 2024), the current readings for all 12 tickers, and limitations. An optional Marp or reveal.js export turns it into slides.

---

## 8. Notebooks (`notebooks/`)

1. **01_data_overview:** coverage per ticker, normalised price chart, log-price chart, returns, and a data-quality report read from the database.
2. **02_single_fit_walkthrough:** one window step by step, including GA convergence, the fitted curve over the log price, residuals, parameter values, and a filter checklist.
3. **03_bubble_indicator_dashboard:** for each ticker, price with the positive and negative confidence bands overlaid, the `t_c` distribution, and a summary heatmap of current confidence across all 12 tickers.
4. **04_ga_diagnostics_and_benchmark:** the GA compared with differential evolution and multi-start Nelder–Mead on SSE, runtime, `t_c` stability across seeds, and hyperparameter sensitivity.

The notebooks import from `src/lppl`, which contains all the logic, and read results from Postgres. They are executed in CI with `nbmake`; outputs are stripped with `nbstripout`.

---

## 9. Reproducibility Procedure (`docs/reproducibility.md`)

- **Pinned environment:** `requirements.lock` plus a Docker image tag, and the Postgres image pinned by digest.
- **Config-driven runs:** everything (tickers, dates, windows, GA hyperparameters, filters, seed) lives in `config/default.yaml`. The full config and the `git_sha` are stored in `fit_runs` for every run.
- **Seeds:** a master seed generates child seeds per (`ticker`, window) through `numpy.random.SeedSequence`, so parallel runs are deterministic.
- **Data snapshots:** raw downloads are hashed and saved, and `lppl ingest --from-snapshot` rebuilds the database without network access. This matters because Yahoo data can be revised.
- **One-command rebuild:** `make reproduce`, which runs `up` → `ingest` → `scan` → `notebooks` → `report`.
- **Verification:** a script compares the new outputs with the reference results, within tolerances.

---

## 10. Risks and Mitigations

| Risk | Mitigation |
|---|---|
| Yahoo rate limits, API changes, data revisions | retry and backoff, pinned `yfinance` version, local snapshots |
| GA is slow over thousands of windows | vectorised fitness, slaved linear parameters, joblib parallelism, a configurable window step, caching of fits in the DB |
| Unstable or meaningless `t_c` | report distributions instead of point estimates, apply filters, run null-model tests |
| Results misread as trading signals | prominent disclaimers in the README and docs |

---

## 11. Open Decisions (defaults proposed)

1. **Three extra tickers:** `AVGO`, `JPM`, `NFLX`. Other options are `AMD`, `BRK-B`, `TSM`, `PLTR`.
2. **NASDAQ series:** Composite `^IXIC` as the main series, with `^NDX` added optionally.
3. **History start:** 2000-01-01.
4. **GA:** our own NumPy implementation, benchmarked against SciPy differential evolution (rather than using DEAP or PyGAD).
5. **Python in the container:** 3.12.
