# Configuration

There are two sources:

1. **Experiment config** (YAML, `config/default.yaml` or `-c path`): the data universe,
   scan schedule, windows, search bounds, filters and GA. It is validated by pydantic models in
   `src/lppl/config.py`. The *effective* config is stored with every run in `fit_runs.config`.
2. **Environment** (`.env` or real environment variables): database connection and Jupyter
   token. These are not part of the experiment.

## Environment variables

| Variable | Default | Used by |
|---|---|---|
| `POSTGRES_USER` / `POSTGRES_PASSWORD` / `POSTGRES_DB` | `lppl` | db container, CLI |
| `POSTGRES_HOST` | `localhost` (host), `db` (inside compose) | CLI |
| `POSTGRES_PORT` | `5442` on the host (mapped to 5432 in the container) | compose port mapping, CLI |
| `LPPL_DATABASE_URL` | unset | overrides all of the above, e.g. `postgresql+psycopg://u:p@h:5432/db` |
| `LPPL_TEST_DATABASE_URL` | unset | integration tests (they always use an isolated `lppl_test` schema) |
| `JUPYTER_TOKEN` | `lppl` | the jupyter service |

## YAML reference

### top level

| Key | Default | Meaning |
|---|---|---|
| `seed` | `42` | master seed. Each window's RNG is `SeedSequence([seed, crc32(ticker), ordinal(t2), window_len])` |

### `data`

| Key | Default | Meaning |
|---|---|---|
| `start` / `end` | `1995-01-01` / `null` (today) | download range; `end` is inclusive |
| `price_field` | `adj_close` | `adj_close` or `close` |
| `snapshot_dir` | `data/snapshots` | where the CSV snapshots are written and read |
| `instruments` | 12 entries | `{ticker, name, group: index|mag7|extra, asset_type: index|equity}` |

To add a ticker, append an entry and re-run `lppl ingest -t NEWTICKER`.

### `scan`

| Key | default.yaml | quick.yaml | Meaning |
|---|---|---|---|
| `start` / `end` | `2019-01-01` / `null` | `2024-06-01` / `null` | range of analysis dates `t2` |
| `step` | `10` | `21` | trading days between analysis dates (counted back from the last date) |
| `n_jobs` | `-1` | `-1` | joblib workers (`-1` = all cores) |
| `method` | `ga` | `ga` | `ga`, `de` or `nm` (`de` and `nm` are benchmarks) |

CLI overrides: `lppl scan --start --end --step --n-jobs -t T1,T2 --label NAME`.

### `windows`

| Key | default.yaml | quick.yaml | Meaning |
|---|---|---|---|
| `min_len` / `max_len` | `125` / `750` | same | window lengths in trading days |
| `step` | `25` (26 windows) | `50` (13 windows) | grid spacing |

### `bounds` (GA search space)

| Key | Default | Meaning |
|---|---|---|
| `m` | `[0.01, 1.0]` | |
| `omega` | `[1.0, 50.0]` | deliberately wider than the filter range |
| `tc_after` | `0.5` | `tc ≤ t2 + tc_after·dt` |
| `tc_min_days` | `1.0` | `tc ≥ t2 + tc_min_days` (the model must be defined on the whole window) |

### `filters` (qualification)

| Key | Default |
|---|---|
| `m_min`, `m_max` | `0.1`, `0.9` |
| `omega_min`, `omega_max` | `2`, `25` |
| `tc_before`, `tc_after` | `0.05`, `0.2` (fractions of `dt` around `t2`) |
| `tc_margin_days` | `3.0`: reject `tc ≤ t2 + 3 days` (fits pinned at the search bound) |
| `damping_min` | `0.5` |
| `oscillations_min` | `2.5` |
| `max_rmse` | `null` (disabled) |

### `ga`

| Key | Default | Meaning |
|---|---|---|
| `population_size` | `40` | individuals per generation |
| `generations` | `30` | maximum number of generations |
| `elite` | `3` | individuals copied unchanged |
| `tournament_size` | `3` | selection pressure |
| `crossover_rate`, `blx_alpha` | `0.9`, `0.3` | BLX-α crossover |
| `mutation_rate` | `0.25` | per-gene probability |
| `mutation_sigma`, `mutation_decay`, `mutation_sigma_min` | `0.15`, `0.95`, `0.005` | step size in unit-cube coordinates, decaying geometrically |
| `patience`, `tol` | `10`, `1e-6` | early stopping on relative improvement |
| `polish`, `polish_maxiter` | `true`, `200` | Nelder–Mead refinement of the best individual |

Filters are applied after fitting. To change them for existing runs without refitting, use
`lppl requalify -c config/default.yaml [--run-id ID]`. The run's stored config is updated.

Changing any value changes the results. Use a new `--label` so that runs stay
comparable, and keep the YAML under version control.
