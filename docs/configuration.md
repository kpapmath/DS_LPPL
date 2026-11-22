# Configuration

There are two sources:

1. **Experiment config** (YAML, `config/default.yaml` or `-c path`): the data universe,
   scan schedule, windows, search bounds, filters and GA. It is validated by pydantic models in
   `src/lppl/config.py`. The *effective* config is stored with every run in `fit_runs.config`.
2. **Environment** (`.env` or real environment variables): database connection and Jupyter
   token. These are not part of the experiment.

## Config files

| File | Purpose |
|---|---|
| `config/default.yaml` | the reference experiment: 12 instruments, scan from 2019 |
| `config/quick.yaml` | a few minutes, for a smoke test |
| `config/pipeline.yaml` | `make pipeline`: `default.yaml` at the standard density (analysis date every 10 days, windows every 25) |
| `config/pipeline_tuned.yaml` (+ `.decisions.yaml`) | written by `lppl tune adopt`: the GA settings adopted by the pipeline |
| `config/bubbles.yaml` | the **burst-bubble universe**: 16 assets with documented bubbles, data from 1985, scan from 1987, using the pipeline's adopted model settings |
| `config/bubbles_short.yaml`, `config/pipeline_short.yaml` | the **short-scale** scans: identical to their base config except for the windows (8 windows, 40–180 days) |
| `config/full.yaml`, `config/full_tuning.yaml`, `config/full_validation.yaml` | `make pipeline-full`: the 12 instruments over their whole history (data from 1985, scan from 1987), the tuning studies on that scan (GA benchmark windows from 1987), and validation with the walk-forward from 1995 |
| `config/full_tuned.yaml` (+ `.decisions.yaml`), `config/full_short.yaml` | written by `make pipeline-full`: the adopted settings and their short-scale variant |
| `config/tuning.yaml`, `config/pipeline_tuning.yaml` | tuning studies ([tuning.md](tuning.md)) |
| `config/validation.yaml` | validation suite (`lppl validate`); see below |
| `config/combined.yaml` | refined signals and combined model (`lppl combine`); see below |

Instrument fields: `ticker`, `name`, `group` (`index`, `mag7`, `extra` or `bubble`) and
`asset_type` (`index`, `equity`, `etf`, `crypto` or `commodity`).

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

## `config/validation.yaml` (`lppl validate`)

Explained test by test in [validation.md](validation.md) §2. Key settings:

| Key | Default | Meaning |
|---|---|---|
| `alarm_threshold` | `0.3` | confidence counted as an alarm |
| `events.horizon`, `events.k_sigma`, `events.vol_window` | `63`, `1.5`, `750` | the main event: a drawdown ≥ k σ √h within h days |
| `split_date`, `embargo_days` | `2024-01-01`, `63` | train/validation split |
| `bootstrap`, `placebo`, `placebo_min_shift` | `1000`, `1000`, `126` | significance tests |
| `walk_forward.first_test_year`, `.trials` | `2021`, `100` | yearly re-tuning folds |
| `null_model.paths`, `.block_days`, `.scan_step`, `.window_step` | `1`, `20`, `21`, `50` | bubble-free rescans |
| `backtest.cut`, `.cost_bps`, `.placebo_shifts` | `0.5`, `10`, `200` | de-risking backtest |
| `labels.names` | the 4 labels of `lppl/labels.py` | pre-registered label suite |
| `labels.k_sigma`, `labels.top_lookback` | `1.5`, `252` | major-top definition |
| `labels.reps`, `labels.tc_test`, `labels.tc_reps` | `1000`, `true`, `2000` | placebo, bootstrap and `tc`-window permutations |

## `config/combined.yaml` (`lppl combine`)

Explained in [combined_model.md](combined_model.md). All values were fixed before the first run.

| Key | Default | Meaning |
|---|---|---|
| `labels` | `drawdown_63`, `drawdown_126`, `major_top_126` | the labels predicted |
| `embargo_days` | 63 / 126 / 252 | per label: drop training dates whose label still looks into the test year |
| `first_test_year`, `min_train` | `1995`, `1000` | first walk-forward year; labelled rows needed to score a fold |
| `l2` | `1.0` | ridge penalty on standardised slopes |
| `momentum_knots` | `[0.1, 0.25, 0.5, 0.75, 0.9]` | hinge terms at these training quantiles of 126-day momentum |
| `alarm_share` | `0.02` | alarms = the top 2 % of predictions (threshold from the training fit) |
| `bootstrap`, `placebo` | `1000`, `1000` | month-block bootstrap; standalone time-shift placebo |
| `signals.cluster_rel`, `signals.cluster_abs_days` | `0.25`, `5` | `tc` agreement tolerance |
| `signals.adf_crit` | `-2.86` | residual test (Dickey–Fuller 5 %, with a constant) |
| `signals.n_params` | `7` | Lagrange regularisation: χ² = SSE / (N − 7) |
| `signals.tc_scale_days` | `63` | decay of `tc_proximity` |
