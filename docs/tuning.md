# Hyperparameter tuning: methodology and procedure

This project tunes the LPPLS bubble detector with **Optuna** (the search algorithm),
**Ray Tune** (parallel execution with early stopping) and **MLflow** (experiment tracking).
Everything runs in Docker next to the existing PostgreSQL. This page explains *what* is tuned,
*against what*, *how overfitting is controlled*, and *how to reproduce* a study.

Contents

1. What can be tuned, and why it is two separate studies
2. Study 1: genetic-algorithm hyperparameters
3. Study 2: the detector (filters and window grid)
4. Search, early stopping and execution
5. Tracking: what is logged where
6. Infrastructure (Docker)
7. Procedure: step by step
8. Reproducibility
9. Interpreting results, and the pitfalls
10. First results (2026-11-21)

---

## 1. What can be tuned, and why it is two separate studies

The detector has three groups of settings. They have very different costs and objectives:

| Group | Examples | Changes the GA fits? | Cost of one evaluation | Objective |
|---|---|---|---|---|
| **GA hyperparameters** | population, generations, mutation, polish | yes | seconds to minutes (refits windows) | quality of the *optimisation* |
| **Qualification filters** | `m`, `ω`, `tc` ranges, damping, oscillations | no | milliseconds | quality of the *detection* |
| **Window grid** | min/max window length, spacing | no (if a subset of the stored grid) | milliseconds | quality of the *detection* |

The GA solves a numerical problem: find the minimum SSE. Its tuning has an objective truth
(the best SSE achievable) and needs no market labels. The filters and windows decide *what
counts as a bubble*, and can only be judged against what happened next. Mixing the two in one
objective would make every detection trial pay the GA's refitting cost and would blur two
different questions. So there are two studies:

* **Study 1, `ga`:** reach the oracle SSE with the fewest objective evaluations.
* **Study 2, `indicator`:** rank analysis dates so that upcoming drawdowns score high.

The key observation that makes study 2 cheap: **the GA result of a window does not depend on
the filters**, a fact tested in `tests/test_indicator.py`. Study 2 therefore re-scores the fits a
scan has already stored (`lppl_fits`, 62,088 rows for the default run) instead of refitting.
A trial takes about 4 ms.

## 2. Study 1: genetic-algorithm hyperparameters

**Benchmark** (`lppl tune build-benchmark`, cached in `data/tuning/ga_benchmark.pkl`):

* 40 real windows: random (ticker, end date ≥ 2019, length ∈ {125, 250, 500, 750}) from the
  database, plus 8 synthetic LPPLS series with known parameters and 1 % noise. The sample is
  deterministic (`benchmark.seed`) and identified by a content digest, which is logged to MLflow.
* **Oracle reference** per window: the best SSE found by a large GA (population 150,
  150 generations, 3 restarts), differential evolution (population size 30, 300 iterations) and
  40-start Nelder–Mead. This is the "best known" SSE against which every trial is measured.

**Trial evaluation** (`ga_benchmark.evaluate_ga`): the candidate GA runs on every window
`seeds_per_window` times. The seeds are the same for every trial (*common random numbers*), so
differences between trials reflect the hyperparameters and not luck.

| Metric | Definition |
|---|---|
| `mean_excess_pct` | mean of `max(SSE / SSE_oracle − 1, 0)` × 100 |
| `p90_excess_pct`, `hit_rate` | tail of the excess, and the share of fits within 0.01 % of the oracle |
| `mean_evals` | objective evaluations per fit: the **cost**, independent of hardware and parallel load |
| `mean_ms` | wall-clock time, logged for information only (noisy when trials run in parallel) |
| `median_tc_err_days` | distance of the fitted `tc` from the oracle's `tc` |
| **`loss`** | `mean_excess_pct + cost_weight × mean_evals / 1000` (minimised; `cost_weight = 0.5`) |

With `cost_weight = 0.5`, an extra 1,000 evaluations per fit (about 70 ms) must buy at least
0.5 percentage points of excess SSE. Raise it to favour speed, or lower it to favour accuracy.
The metrics are reported after each of `chunks = 4` window batches, which enables early
stopping (§4).

**Search space** (`config/tuning.yaml → ga.space`): population 10–120, generations 5–80,
elite 0–5, tournament 2–6, crossover rate 0.5–1, BLX-α 0–0.6, mutation rate 0.05–0.6, mutation
σ 0.01–0.4 (log), decay 0.85–1, patience 3–20, polish on/off, and polish iterations 50–400
*(sampled only when polish is on: a conditional parameter)*.

## 3. Study 2: the detector (filters and window grid)

### 3.1 What counts as an event

There is no ground-truth list of "bubbles". We use an objective, asset-adaptive event
definition: at analysis date `t2` the label is **1** if, within the next `H = 63` trading days
(3 months), the price falls from its running peak by at least

```
k_sigma × σ(t2) × √H        (log terms; k_sigma = 1.5)
```

Here `σ(t2)` is the daily log-return volatility over the **trailing** 750 days. The threshold
uses only information available at `t2`, which is tested. It adapts to each asset: median
thresholds are 12 % for the S&P 500, 19 % for MSFT, 32 % for NVDA and 37 % for TSLA. Dates whose horizon runs past the end of the data have no
label and are excluded. In the default scan, 14.7 % of training dates and 11.6 % of validation
dates are events.

### 3.2 Score and objective

The positive confidence `pos_conf(t2)` is treated as a **ranking score** for "an event is
coming". It is evaluated with **average precision (AP)**, the area under the precision–recall
curve. AP is the standard metric for rare events and needs no alarm threshold. A random score
has AP equal to the base rate, so **lift = AP / base rate** is reported alongside it.
For interpretation, precision and recall at the alarm threshold `pos_conf ≥ 0.3` are also logged.

As a reference, a **naive momentum score** is computed: the trailing 126-day log return in σ units.
If LPPLS does not beat this baseline, its complexity is not paying off.

### 3.3 Overfitting control: time split with an embargo

```
2019 ─────────── train ───────────┤ embargo ├──────── validation ────────► 2026
                                   63 days   split_date = 2024-01-01
```

* The optimiser only sees **train** AP (`objective = ap_train`).
* **Validation** AP is computed for every trial and logged, but never used to select.
* The **embargo** removes training dates whose 63-day label horizon would reach into the
  validation period ("purging", in López de Prado's terminology). Without it, the train labels
  would contain validation information.
* Trials that leave fewer than `min_windows = 5` windows per date are invalid (objective 0).

The honest reading of a study is the **validation AP of the selected configuration, compared
with the status quo and with the momentum baseline**. This comparison is the bar chart
`ap_train_vs_valid.png` that every study logs.

### 3.4 Search space

`m_min` 0–0.3, `m_max` 0.6–1, `ω_min` 1–6, `ω_max` 10–50, `tc_after` 0.05–0.5,
`tc_margin_days` 0–15, `damping_min` 0–1.5, `oscillations_min` 0–5, and the window grid:
min length 125–375, max length 400–750, stride 1 or 2 (every k-th stored length). All ranges lie
inside what the stored fits can express: the GA searched `m ∈ [0.01, 1]`, `ω ∈ [1, 50]` and
`tc ≤ t2 + 0.5·dt`.

> The window grid can only be tuned within the grid the source scan stored. The grid spacing
> is read from the source run itself (`window_grid_step`, logged). A scan with
> `windows.step: 5` would let the study choose among 5-day steps.

## 4. Search, early stopping and execution

| | Optuna backend (`--backend optuna`) | Ray backend (`--backend ray`) |
|---|---|---|
| Sampler | TPE (multivariate, 10 random start-up trials, seeded) | the same TPE, via `ray.tune.search.optuna.OptunaSearch` |
| Search space | `settings.suggest` (define-by-run; supports conditionals) | the same function (`DefineByRun`) |
| Study storage | PostgreSQL database `optuna` (resumable, shown in optuna-dashboard) | the same storage, passed to `OptunaSearch` |
| Early stopping | `MedianPruner` after each benchmark chunk (GA study) | **ASHA** (`max_t = chunks`, reduction factor 2) |
| Parallelism | threads (`--jobs`); exactly reproducible with `--jobs 1` | one Ray actor per trial, `ConcurrencyLimiter(--jobs)`, actors reused |
| Trial 0 | the current configuration (enqueued) | the same (`points_to_evaluate`) |
| Tracking | each trial is an MLflow child run (logged by `lppl`) | `MLflowLoggerCallback`, child runs under the same parent |

**Which backend to use.** Use **Ray for the GA study**: trials are expensive, so parallel
actors and ASHA's aggressive early stopping save real time. Use **Optuna for the detector study**:
a trial takes 4 ms, so Ray's per-trial scheduling overhead dominates. Measured: 20 trials took
about 7 minutes with Ray, while 300 trials took about 2.5 minutes with Optuna, most of it
MLflow logging. Both are supported for both studies.

**ASHA and the Optuna study.** When ASHA stops a Ray trial early, `OptunaSearch` tells the
study the *last reported* (partial-run) loss and marks the trial COMPLETE. TPE and the logged
parameter importances therefore see some partial values. They are unbiased running means over
random window chunks, but noisier. `lppl` selects the **best configuration only among fully
evaluated trials** (`training_iteration == chunks`), and labels stopped trials `STOPPED` in
`trials.csv`. With the Optuna backend, pruned trials are marked PRUNED and excluded everywhere.

**Why TPE.** The spaces are small (11–12 parameters), mixed (int, float, categorical, conditional)
and the objective is noisy. TPE handles all of this, and the first 10 random trials give it a
start. With a few hundred trials, random search is a reasonable alternative and easy to swap in
(`runners.run_optuna`, `sampler=`).

## 5. Tracking: what is logged where

| Where | What |
|---|---|
| MLflow experiment `lppl-ga` / `lppl-indicator` | one **parent run** per study (run name = study name) |
| parent run, params | backend, trials, jobs, seed, the metric and its direction, the full study settings (`study.*`), and the best parameters (`best.*`) |
| parent run, metrics | `best_*`, `baseline_*` (status-quo trial), `importance.*` (fANOVA), `n_trials` |
| parent run, tags | `git_sha`, `lppl_version`, `backend`, the benchmark digest or source scan run id, base rates, and the baseline AP |
| parent run, artifacts | `study_definition.json`, `trials.csv`, `best.json`, **`tuned_config.yaml`**, `param_importances.json`, `history.png`, `quality_vs_cost.png` (GA) or `ap_train_vs_valid.png` (detector) |
| child runs (one per trial) | the parameters, final metrics, the intermediate metric history (step = chunk), and the trial state (COMPLETE / PRUNED) |
| PostgreSQL `optuna` | the Optuna study: every trial with its parameters, values and intermediate values |
| `data/tuning/<study>/` | local copies: `best.json`, `tuned_config.yaml`, `trials.csv`, plots |
| `data/tuning/ray_results/` | Ray Tune experiment state (Ray backend) |

## 6. Infrastructure (Docker)

```
docker compose --profile tune
┌─────────┐     ┌────────────────────────────┐     ┌────────────────────────────┐
│   db    │◄────┤ mlflow  (lppl-tune image)  │     │ optuna-dashboard           │
│ pg 16   │     │ creates DBs mlflow, optuna │     │ http://localhost:8081      │
│ lppl    │     │ http://localhost:5050      │     └────────────┬───────────────┘
│ mlflow  │◄────┤ artifacts: volume          │                  │
│ optuna  │◄────┴──────────────▲─────────────┘◄─────────────────┘
└─────────┘                    │ MLFLOW_TRACKING_URI=http://mlflow:5000
     ▲                ┌────────┴────────────────────────┐
     └────────────────┤ tune (lppl-tune, `lppl tune …`) │ shm 2 GB for Ray
                      │ ./data ./config mounted          │
                      └──────────────────────────────────┘
```

* **Image `lppl-tune:0.1.0`** (`docker/tune.Dockerfile`) = `lppl-app` + `requirements-tune.lock`
  (Ray 2.60, Optuna 5.1, MLflow 3.18, optuna-dashboard 0.21, plus their dependencies; pinned and
  resolved against `requirements.lock` without changing any base pin).
* MLflow uses Postgres as its backend store (`postgresql+psycopg`) and serves artifacts from the
  `mlartifacts` volume (`--serve-artifacts`), so clients need no shared filesystem.
* MLflow 3 validates the Host header, so `--allowed-hosts` lists `mlflow:5000` (containers) and
  `localhost:*` (browser).
* Ports: MLflow **5050** (5000 is often taken by macOS AirPlay), optuna-dashboard **8081**.

## 7. Procedure: step by step

```bash
make tune-build                    # lppl-app + lppl-tune images
make tune-up                       # MLflow http://localhost:5050, dashboard http://localhost:8081

# Study 2 (detector): needs a finished `default` scan in the DB (make scan)
make tune-indicator ITRIALS=300    # Optuna, about 1-2 min

# Study 1 (GA)
make tune-benchmark                # one-off: sample windows + oracle (cached)
make tune-ga-ray TRIALS=60 JOBS=2  # Ray Tune + ASHA, parallel
#   or: make tune-ga TRIALS=60 JOBS=1   (Optuna + median pruner, exactly reproducible)

# Inspect
docker compose --profile tune run --rm tune best <study-name>
open http://localhost:5050         # compare runs, artifacts, tuned_config.yaml

# Adopt a result
docker compose --profile tune run --rm tune export-best <study-name> --out config/tuned.yaml
docker compose --profile tune run --rm tune evaluate-config -c config/tuned.yaml   # detector only
make scan CONFIG=config/tuned.yaml # full rescan with the tuned settings (new run label)

# Or everything end to end: base scan → both studies → adopt only if validated → tuned scan
# → validation of both → comparison report (see validation.md §7)
make pipeline-full
```

`lppl tune run ... --name <existing study>` **resumes** a study. Optuna loads its trials from
Postgres and continues sampling.

## 8. Reproducibility

* **Pinned stack:** `requirements.lock` plus `requirements-tune.lock`, and fixed image tags.
* **Fixed inputs:** the GA benchmark digest and the detector's source scan `run_id` are logged on
  every parent run. Snapshots (`data/snapshots`) and `data/tuning/ga_benchmark.pkl` should be
  archived with the result.
* **Seeds:** the sampler seed (`seed` in `tuning.yaml`), the benchmark seed, and common random
  numbers per window.
* **Determinism:** `--backend optuna --jobs 1` replays the same trial sequence. With parallel
  trials (threads or Ray actors), the order in which results reach TPE depends on timing, so
  the sequence of suggested configurations is not reproducible. Each *individual* trial
  still is: its metrics depend only on its parameters and the fixed seeds.
* **The study itself is data:** `study_definition.json` (space, defaults, data identifiers) plus
  `config/tuning.yaml` under git describe the whole experiment.

## 9. Interpreting results, and the pitfalls

* **Train vs validation.** A higher train AP with a lower validation AP is overfitting, not progress.
  The first smoke test showed exactly this (train AP 0.156 → 0.180, validation 0.120 → 0.113). Adopt
  a tuned detector only if it beats the status quo **and** the momentum baseline on validation.
* **Only one validation period.** 2024–2026 contains few independent events, and the labels of
  neighbouring dates overlap heavily (63-day horizons, 10-day steps). AP differences of a
  few hundredths are within noise. For a decision, use several splits (walk-forward) or block
  bootstrap the validation AP; neither is implemented yet.
* **Multiple testing.** Hundreds of trials will find a configuration that looks good by chance on
  any fixed sample. This is the reason for the validation period, and the reason to read it only once
  per decision.
* **The event definition is a modelling choice.** `horizon` and `k_sigma` change the base rate and
  what "success" means. Changing them is a different study. Keep them fixed while comparing
  configurations.
* **GA study scope.** The GA is tuned on 48 windows. Check that a tuned GA does not
  degrade on the full scan before adopting it (rescan, then compare `lppl_fits.sse` with the
  default run on the same windows).

## 10. First results (2026-11-21)

Both studies were run in the Docker stack. They are visible in MLflow (`lppl-ga`, `lppl-indicator`)
and in optuna-dashboard. The local copies are in `data/tuning/<study>/`.

### GA study `ga-ray-v1`: Ray Tune + ASHA, 30 trials, 2 in parallel, 6.5 minutes

| | current GA (40 × 30, polish 200) | tuned GA |
|---|---|---|
| loss | 1.121 | **0.576** |
| mean excess SSE over the oracle | 0.448 % | **0.150 %** |
| hit rate (within 0.01 % of the oracle) | 89.6 % | 89.6 % |
| objective evaluations per fit | 1,346 | **853 (−37 %)** |
| median `tc` distance from the oracle | 0 days | 0 days |

![GA quality vs cost](figures/tuning_ga_quality_vs_cost.png)

* ASHA stopped 16 of the 30 trials early: 10 after the first benchmark chunk, 6 after the second.
* Tuned parameters: population 91, **generations 6**, elite 1, tournament 2, crossover 0.98,
  BLX-α 0.51, mutation rate 0.34, σ 0.22, decay 0.93, patience 16, polish on with **375
  iterations**. The search moved work from many GA generations into one broad generation plus a
  longer Nelder–Mead polish. This is consistent with notebook 04, which found the polish to be
  the dominant factor.
* Most important parameters (fANOVA): generations 0.35, elite 0.15, patience 0.13, BLX-α 0.11.
* **Status:** promising, but not yet adopted. Before switching `config/default.yaml`, rescan the
  case studies with `data/tuning/ga-ray-v1/tuned_config.yaml` and compare the SSE per window with
  the default run (§9, GA study scope).

### Detector study `indicator-optuna-v1`: Optuna TPE, 300 trials, about 2.5 minutes

| average precision (event: 3-month drawdown ≥ 1.5 σ√H) | train 2019–2023 | validation 2024– |
|---|---|---|
| base rate (random score) | 0.147 | 0.116 |
| naive momentum baseline | 0.147 | **0.132** |
| current detector | 0.156 | 0.120 |
| tuned detector | **0.230** | 0.115 |

![train vs validation](figures/tuning_indicator_ap.png)

* The tuned filters (`m ∈ [0.30, 0.88]`, `ω ∈ [1.7, 10.9]`, ≥ 4.65 oscillations, `tc` margin
  14 days, window 150–750) raise **train** AP by 47 %, but do **not** transfer to the
  validation period. There they fall below the base rate.
* Neither the current nor the tuned LPPLS detector beats the naive momentum score out of
  sample for this event definition.
* **Conclusion:** do not adopt the tuned filters. As a *forward 3-month drawdown predictor*,
  the LPPLS confidence has no demonstrable out-of-sample skill on this universe and period (on
  a burst-bubble universe it does; see validation.md §6). This
  agrees with the published criticism (`methodology.md` §9) and with the case studies, where the
  signal was informative about *regimes* and `tc` windows more than about the timing of
  drawdowns. The tuning framework is what makes this conclusion measurable and reproducible. Next
  steps: walk-forward splits, other event definitions (for example peak-to-trough over 6 months), and
  the denser scan (`windows.step: 5`) that is currently running.

### Pipeline run (2026-11-21)

The 2019+ pipeline (configs retired, in commit `ce9d40e`) repeated both studies on a fresh baseline scan. The results are in
[validation.md](validation.md) §5.

* **GA study** (`ga-pipeline-20261121`): loss fell from 1.121 to 0.576, with 37 % fewer
  evaluations. It was **adopted**.
* **Detector study** (`indicator-pipeline-20261121`): it raised train AP from 0.156 to 0.230,
  but 2024+ AP fell from 0.120 to 0.115. It was **rejected** by the validated-adoption rule. This
  is the same over-fitting pattern as `indicator-optuna-v1`.

### Full-period pipeline (2026-11-22)

`make pipeline-full` repeated both studies on the 12 instruments over 1987–2026
([validation.md](validation.md) §7).

* **GA:** loss fell from 1.754 to 0.897 on a benchmark drawn from the whole period. It was
  **adopted**, with the same settings as in the 2019+ run.
* **Detector:** the 2024+ AP fell from 0.120 to 0.113 (train 1987–2023). It was **rejected**: a
  third over-fitting result out of three.

> **Correction (2026-11-22).** `n_windows_min`, the check that a detector setting keeps at least
> `min_windows` windows, used to take the minimum over *all* dates. On long histories, the
> earliest dates have only a few stored windows, so every setting was rejected. It now considers
> only dates where the full stored window grid fits (`detection.min_windows_used`). The studies
> above all used scans from 2019 with all 26 windows on every date, so they are unaffected.

