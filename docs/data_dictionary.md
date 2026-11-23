# Data dictionary

The schema is defined in `src/lppl/data/db.py` and created by `lppl init-db`, which is
also called by `ingest` and `scan`. Connect with any Postgres client:

```bash
psql "postgresql://lppl:lppl@localhost:5442/lppl"
# or: docker compose --profile tools up -d adminer   -> http://localhost:8080 (server: db)
```

## `instruments`

| Column | Type | Description |
|---|---|---|
| `ticker` | varchar PK | Yahoo symbol (`^GSPC`, `AAPL`, …) |
| `name` | text | display name |
| `grp` | varchar | `index` \| `mag7` \| `extra` \| `bubble` |
| `asset_type` | varchar | `index` \| `equity` \| `etf` \| `crypto` \| `commodity` |

## `prices`

Primary key `(ticker, date)`. One row per trading day.

| Column | Type | Description |
|---|---|---|
| `ticker` | varchar FK | → `instruments.ticker` |
| `date` | date | trading date (exchange-local, timezone removed) |
| `open, high, low, close` | float | **unadjusted** OHLC |
| `adj_close` | float | close adjusted for splits and dividends. **The model is fitted on this** (`data.price_field`) |
| `volume` | bigint | shares traded (0 for indices where it is unavailable) |

## `ingestion_log`

One row per ticker per ingestion: the provenance of every price.

| Column | Type | Description |
|---|---|---|
| `id` | serial PK | |
| `ticker` | varchar | |
| `fetched_at` | timestamptz | UTC |
| `source` | varchar | `yahoo` or `snapshot` |
| `n_rows` | int | rows upserted |
| `first_date`, `last_date` | date | coverage |
| `sha256` | char(64) | hash of the canonical CSV snapshot (`data/snapshots/<ticker>.csv`) |
| `yfinance_version` | varchar | library version used |
| `issues` | text | non-fatal validation warnings (gaps > 7 days, stale prices, high < low) |

## `fit_runs`

One row per `lppl scan` invocation.

| Column | Type | Description |
|---|---|---|
| `run_id` | uuid (text) PK | |
| `created_at`, `finished_at` | timestamptz | |
| `status` | varchar | `running` → `finished` \| `failed` |
| `label` | text | e.g. `default`, `quick`, `case:dotcom`, `pipeline-base`, `bubbles`, `bubbles-short` (the labels used in the docs are listed in reproducibility.md) |
| `seed` | int | master seed |
| `git_sha` | varchar | commit of the code (null if not a git checkout) |
| `package_version` | varchar | `lppl.__version__` |
| `config` | jsonb | **full** effective configuration (YAML plus CLI overrides) and the ticker list |

## `lppl_fits`

One row per (run, ticker, analysis date, window). Unique `(run_id, ticker, t2, window_len)`.
Deleting a run cascades to these rows.

| Column | Type | Description |
|---|---|---|
| `t1`, `t2` | date | window start and end (`t2` = analysis date) |
| `window_len` | int | number of trading days in the window |
| `tc` | float | critical time, decimal years |
| `tc_date` | date | `tc` as a calendar date |
| `m`, `omega` | float | nonlinear parameters |
| `a`, `b`, `c1`, `c2` | float | linear parameters (`A, B, C1, C2`) |
| `sse`, `rmse` | float | fit error on the log price |
| `damping` | float | `m·|B| / (ω·|C|)` |
| `n_osc` | float | `(ω/π)·ln((tc−t1)/(tc−t2))` |
| `qualified` | bool | passes all filters (`docs/methodology.md` §6) |
| `sign` | smallint | `+1` if B<0 (bubble), `−1` if B>0 (anti-bubble) |
| `generations`, `n_evals`, `elapsed_ms` | | optimiser diagnostics (`n_evals`, `elapsed_ms` summed over all seeds in a robust run) |
| `n_seeds_ok` | smallint, nullable | robust variant only: number of seeds in the agreeing group (`qualified` = at least `robust.min_agree`); NULL for ordinary runs |

## `bubble_indicator`

Primary key `(run_id, ticker, date)`. The aggregation of `lppl_fits` per analysis date.

| Column | Type | Description |
|---|---|---|
| `date` | date | analysis date `t2` |
| `pos_conf` | float [0,1] | share of windows qualified with `sign = +1` |
| `neg_conf` | float [0,1] | share of windows qualified with `sign = −1` |
| `n_windows`, `n_qualified` | int | denominators |
| `tc_median_date`, `tc_q25_date`, `tc_q75_date` | date | quartiles of `tc` over the qualified fits (null if there are none) |

## Useful queries

```sql
-- latest default run
select run_id, created_at, git_sha from fit_runs
where status = 'finished' and label = 'default' order by created_at desc limit 1;

-- strongest bubble readings in the last 90 days of that run
select ticker, date, pos_conf, tc_median_date
from bubble_indicator where run_id = :run and date > now() - interval '90 days'
order by pos_conf desc limit 20;

-- qualification rate per ticker
select ticker, avg(qualified::int) as q_rate, count(*) from lppl_fits
where run_id = :run group by ticker order by q_rate desc;
```

## Report files (not in the database)

`lppl validate` and `lppl combine` write their results as files, not tables:

| folder | written by | main files |
|---|---|---|
| `reports/validation/<label>_<run>/` | `lppl validate` | `summary.json`, `skill.csv`, `labels.csv`, `tc_window.csv`, `events.csv`, … ([validation.md](validation.md) §3) |
| `reports/combined/<label>_<run>/` | `lppl combine` | `features.csv` (one row per ticker × date: LPPLS signals, market features, labels), `oos.csv`, `metrics.csv`, `increments.csv`, `per_year.csv`, `standalone.csv`, `coefficients.csv`, `summary.json` ([combined_model.md](combined_model.md) §3) |

