# Validating the model's accuracy

`lppl validate` runs a battery of tests on a stored scan and answers one question with
evidence: **is the LPPLS bubble confidence better than luck at anticipating large falls?**
Notebook `05_validation` presents the same report graphically. This page explains each test,
how to run the suite, and what it found for the current scan.

Contents

1. What is being validated
2. The tests
3. Running it
4. Results for the current scan (run `8264c9f6`, data to 2026-11-20)
5. End-to-end pipeline: tuned vs baseline (`make pipeline`, 2026-11-21)
6. Limitations and next steps

---

## 1. What is being validated

The model makes three kinds of claim, and each needs a different check:

| Claim | Validated by |
|---|---|
| the GA finds the right curve | unit tests (synthetic recovery), oracle benchmark ([tuning.md](tuning.md)) |
| **high confidence comes before large falls** | **`lppl validate` (this page)** |
| acting on the signal is useful | **`lppl validate`, backtest section** |

**Event definition** (`config/validation.yaml → events`): an analysis date `t2` is followed by an
*event* if, within the next **63 trading days**, the price falls from its running peak by at least
**1.5 × σ × √63**. Here σ is the trailing daily volatility over 750 days, with no look-ahead. The
threshold is about 12 % for the S&P 500 and 32 % for NVDA. The last 63 days of data have no label.

**Fair references.** Every score is compared with:

* the **base rate**: the score of a random guess;
* a **momentum score**: the trailing 126-day log return in σ units ("it went up a lot recently");
* **time-shifted placebos** of the confidence itself;
* the detector's behaviour on **bubble-free simulated prices**.

**Time order.** The confidence at `t2` uses only prices up to `t2`, by construction of the scan.
Only the labels look forward. The *validation* period is 2024 onwards, and was never used to choose
any setting. Training dates within 63 days of it are dropped (the embargo).

## 2. The tests

| # | Test | Method | What a real signal looks like |
|---|---|---|---|
| 1 | **Skill** | Average precision (AP) of the confidence for events, with precision and recall at the alarm level (≥ 0.3), on all dates and on train and validation separately | AP clearly above the base rate and the momentum AP |
| 2 | **Calibration** | event rate per confidence band (0, ≤ 0.1, < 0.3, < 0.6, ≥ 0.6) | the event rate rises band by band |
| 3 | **Block bootstrap** | 1,000 resamples of whole calendar months (all tickers of a month together), giving 95 % intervals for the AP gaps to the base rate and to momentum | intervals entirely above 0 |
| 4 | **Placebo** | 1,000 random circular time shifts (≥ 6 months) of each ticker's confidence; labels fixed; one-sided p-value | p < 0.05: the *timing* matters |
| 5 | **Robustness grid** | the same AP for horizons of 21, 63 and 126 days × thresholds of 1, 1.5 and 2 σ; each cell shows its event count | lift > 1 across the grid, not in one lucky cell |
| 6 | **Timing** | alarm episodes: lead time from first alarm to the 6-month top, rise to the top, fall after it, and whether the top is inside the `tc` window | short lead, top inside the window |
| 7 | **Walk-forward** | for each year from 2021: the detector as scanned, and the detector **re-tuned with Optuna on earlier years only**, both scored on that year | re-tuned beats as-scanned on the *next* year, and both beat momentum |
| 8 | **Null model** | each ticker's returns resampled by a stationary block bootstrap (mean block 20 days: keeps volatility clustering, destroys bubble structure); the detector re-run with the same window grid on real and simulated prices; one-sided two-proportion test | significantly more alarms on real prices |
| 9 | **Backtest** | equal-weight portfolio; halve a ticker's exposure from the day *after* an alarm to the next analysis date; 10 bp per change. Compared with buy & hold, momentum at the **same alarm rate**, and the time-shifted LPPLS placebo (200 shifts) | better risk-adjusted return than all three references |

All randomness is seeded (`seed` in the config), so a report is reproducible for a given scan.

## 3. Running it

```bash
make validate-quick        # ~20 s, app image: tests 1–6 and 9 (no null-model rescans, no MLflow)
make validate              # ~18 min, tuning image: all 9 tests, logged to MLflow (lppl-validation)
make validate LABEL=quick  # validate another scan label

# or directly
lppl validate --label default [--run-id ID] [--null-paths N] [--no-walk-forward] [--no-mlflow]
```

**Outputs** go to `reports/validation/<label>_<run>/`: one CSV per test (`skill.csv`,
`calibration.csv`, `bootstrap.csv`, `placebo.csv`, `grid.csv`, `timing.csv`, `walk_forward.csv`,
`null_model.csv`, `backtest.csv`, `backtest_curves.csv`, `events.csv`), plus `summary.json` (key
numbers and verdict), `settings.yaml` and the figures. With MLflow, the same files are artifacts of a
run in the `lppl-validation` experiment, with every summary number as a metric.

**Cost.** Everything except the null model takes seconds. The null model re-runs the GA scan on
simulated prices: about 1 minute per ticker and path on 2 CPUs (`null_model.paths`). Walk-forward
re-tuning takes about 100 Optuna trials × 4 ms per fold.

## 4. Results for the current scan

Scan `8264c9f6` (`default`): 12 instruments, an analysis date every 10 trading days from 2019 to
2026-11-20, 26 windows per date. There are 2,304 labelled dates, 304 of them followed by an event.

### Verdict

| Test (5 % level) | Result |
|---|---|
| better than a random guess, all dates | ✘ no |
| better than a random guess, 2024+ (unseen) | ✘ no |
| better than momentum, all dates | ✘ no |
| better than momentum, 2024+ | ✘ no |
| timing matters (beats the time-shifted placebo) | ✘ no (p = 0.40) |
| higher confidence = more events | ✘ no |
| fires more on real than on bubble-free prices | ✘ no (4.5 % vs 3.9 %, p = 0.18) |

### 1 · Skill

| | all dates | train 2019–23 | validation 2024+ |
|---|---|---|---|
| base rate | 0.132 | 0.147 | 0.116 |
| AP, LPPLS | 0.135 | 0.156 | 0.120 |
| AP, momentum | 0.137 | 0.147 | **0.132** |
| precision at alarm (≥ 0.3) | 0.095 | 0.154 | 0.045 |
| recall at alarm | 0.026 | 0.028 | 0.022 |

An alarm was followed by an event in 9.5 % of cases, *less* often than on a random date (13.2 %).
Alarms caught 2.6 % of events.

### 2 · Calibration

![calibration](figures/validation_calibration.png)

| confidence | dates | event rate |
|---|---|---|
| 0 | 1,740 | 13.3 % |
| (0, 0.1] | 281 | 12.8 % |
| (0.1, 0.3) | 199 | 14.6 % |
| [0.3, 0.6) | 77 | **7.8 %** |
| ≥ 0.6 | 7 | 28.6 % (2 of 7, too few to judge) |

### 3–4 · Significance

![bootstrap](figures/validation_bootstrap.png)

* LPPLS − base rate: +0.003, 95 % interval [−0.007, +0.025] on all dates, and [−0.006, +0.025] on
  validation.
* LPPLS − momentum: −0.002 [−0.053, +0.033] on all dates, and −0.012 [−0.090, +0.024] on validation.
* Placebo: p = **0.40** for LPPLS (and 0.58 for momentum). Shifting the signal randomly in time does as
  well as the real timing.

### 5 · Robustness

![grid](figures/validation_grid.png)

The lift is between 0.92 and 1.14 in every cell with a reasonable number of events. The one striking
cell (126 days, 2 σ, validation: lift 2.82) rests on **11 events** and is noise.

### 6 · Timing

20 completed alarm episodes:

* the price topped a median **110 trading days (about 5 months)** after the first alarm;
* the top fell inside the `tc` window in **1 of 20** episodes (Broadcom, spring 2024);
* the full episode list is in the notebook and in [how_the_model_works.md](how_the_model_works.md) §7.5.

### 7 · Walk-forward

![walk forward](figures/validation_walk_forward.png)

| test year | base rate | momentum | LPPLS as scanned | LPPLS re-tuned on past |
|---|---|---|---|---|
| 2021 | 0.077 | 0.143 | 0.223 | 0.123 |
| 2022 | 0.310 | 0.323 | 0.311 | 0.310 |
| 2023 | 0.003 | 0.004 | 0.003 | 0.003 |
| 2024 | 0.050 | 0.140 | 0.070 | 0.058 |
| 2025 | 0.180 | 0.176 | 0.187 | 0.203 |
| 2026 | 0.120 | 0.212 | 0.115 | 0.115 |
| **mean** | 0.123 | **0.166** | 0.151 | 0.135 |

Re-tuning on past years improved the following year only once (2025). On average it made things worse,
which confirms the over-fitting seen in the tuning study. The scanned detector's best year (2021) is the
NVIDIA episode. Momentum wins on average.

### 8 · Null model

![null model](figures/validation_null_model.png)

| | alarm rate |
|---|---|
| real prices | 4.5 % |
| bubble-free simulated prices | 3.9 % |
| one-sided p-value | 0.18 |

The detector fires about as often on prices that *cannot* contain a bubble. NVDA (13.6 % real vs 1.1 %
simulated) and AVGO (9.5 % vs 2.1 %) are the exceptions where real alarms are much more frequent.
MSFT, Tesla, Meta and both indices fired *more* on simulated paths. With one simulated path per ticker,
per-ticker numbers are noisy. Raise `null_model.paths` for a firmer estimate.

### 9 · Backtest (2019 – 2026)

![backtest](figures/validation_backtest.png)

| strategy | annual return | Sharpe | max drawdown | average exposure |
|---|---|---|---|---|
| buy & hold | 30.0 % | 1.11 | −40.7 % | 100 % |
| LPPLS de-risk | 29.1 % | 1.10 | −40.0 % | 98.1 % |
| momentum de-risk (same alarm rate) | 30.0 % | 1.13 | −40.7 % | 98.2 % |
| LPPLS placebo (time-shifted) | 29.3 % | 1.11 | −40.1 % | – |

De-risking on LPPLS alarms ended about 5 % behind buy & hold and slightly behind its own
time-shifted placebo. It barely changed the worst drawdown (−40.0 % vs −40.7 %, in the 2020 COVID
crash, when only Microsoft had an alarm beforehand).

### Conclusion

On this universe and period, **the LPPLS confidence shows no statistically significant ability to
anticipate 3-month drawdowns**. It is not better than chance, not better than a momentum rule, its
timing is not better than a random shift, and it fires about as often on bubble-free simulated prices.
What it does capture is descriptive: alarms coincide with strong, accelerating rallies that typically
run for months before a larger-than-usual correction ([how_the_model_works.md](how_the_model_works.md)
§6.6). This agrees with the published criticism of LPPLS as a timing tool
([methodology.md](methodology.md) §9).

## 5. End-to-end pipeline: tuned vs baseline

`make pipeline` runs every stage in one go:

1. ingest prices;
2. baseline scan (`pipeline-base`);
3. GA study (Ray Tune + ASHA) and detector study (Optuna);
4. `lppl tune adopt`;
5. tuned scan (`pipeline-tuned`);
6. full validation of both scans;
7. `lppl compare-reports`, written to `reports/pipeline_<id>.md`.

```bash
make pipeline                       # PIPE_ID defaults to today's date; ~2 h 15 min on 2 CPUs
make pipeline TRIALS=60 JOBS=2      # more GA trials
```

**Configuration.** The scan uses `config/pipeline.yaml`: the defaults with an analysis date every
10 trading days and a window every 25 days, so the scans match run `8264c9f6`. The studies use
`config/pipeline_tuning.yaml`. The source label is `pipeline-base` and the window search step is 25.

**Adoption only if validated.** `lppl tune adopt --policy validated` adopts a study's parameters
only when they beat the current setting:

* GA: on benchmark loss;
* detector: on the **2024+ AP**, the period never used for tuning.

The tuned config is written to `config/pipeline_tuned.yaml`, with a header recording the
decisions, and the reasons are saved in `config/pipeline_tuned.decisions.yaml`. `--policy all`
adopts everything regardless.

### Run of 2026-11-21

The run took 2 h 12 min and had no errors. The baseline scan is `0df8f184`; the tuned scan is
`bb29a715`.

| Study | Baseline | Best tuned | Adopted |
|---|---|---|---|
| GA (`ga-pipeline-20261121`, Ray, 40 trials, 20 completed, the rest stopped by ASHA) | loss 1.121 | **0.576** | ✔ |
| Detector (`indicator-pipeline-20261121`, Optuna, 300 trials) | AP 2024+ 0.1197 | 0.1153 | ✘ |

* **GA, adopted.** The adopted settings are a population of 91, 6 generations, elite 1,
  tournament 2, BLX-α 0.51, mutation rate 0.34 and σ 0.22, and a Nelder–Mead polish of up to 375
  iterations.
  * Fits are closer to the oracle: 0.15 % excess SSE, against 0.45 %.
  * Each fit needs **37 % fewer objective evaluations**: 852, against 1,345.
  * The hit rate is the same (90 %).
  * In practice, a small population is explored briefly and the local polish does the rest.
* **Detector, rejected.** On train (2019–23), AP rose from 0.156 to 0.230. On 2024+ it fell
  from 0.120 to 0.115, which is over-fitting. The filters stay at their defaults, so the tuned scan
  differs from the baseline only in its GA settings.

**Validation, baseline vs tuned** (full suite, null model included):

| | baseline `0df8f184` | tuned `bb29a715` | reference |
|---|---|---|---|
| AP, all dates | 0.135 | 0.137 | base rate 0.132 |
| AP, 2024+ | 0.120 | 0.121 | base rate 0.116 · momentum **0.132** |
| precision at alarm (≥ 0.3) | 9.5 % | 11.1 % | base rate 13.2 % |
| alarm rate | 3.7 % | 3.9 % | |
| placebo p-value | 0.39 | 0.33 | |
| walk-forward mean AP | 0.151 | 0.153 | |
| null model: real vs bubble-free alarm rate (p) | 4.5 % vs 3.9 % (0.18) | 4.6 % vs 3.8 % (0.12) | |
| median lead, first alarm to top | 110 days | 98.5 days | |
| tops inside the `tc` window | 1 of 20 | 2 of 22 | |
| backtest annual return / Sharpe | 29.1 % / 1.10 | 29.0 % / 1.10 | buy & hold 30.0 % / 1.11 |

**Every verdict is "no" for both scans** at the 5 % level. Tuning the GA makes the fits cheaper
and more accurate, and it nudges every indicator in the right direction, but by amounts well inside
the bootstrap intervals. For example, the 95 % interval for the 2024+ AP is [0.059, 0.204] for
the tuned scan.

The conclusion of §4 is unchanged. Better optimisation does not create predictive skill: the
limitation is the signal, not the fitting. The baseline scan reproduces run `8264c9f6` exactly
(same AP, same verdict), which confirms that a scan is deterministic for a given config.

## 6. Limitations and next steps

* **Small sample.** There are 304 events but only about 20 independent alarm episodes. The block
  bootstrap accounts for overlapping labels, but the intervals stay wide, so "not significant" also
  means "a small effect cannot be ruled out".
* **Universe.** These are 12 large-cap US names that survived and grew (survivorship bias), in one
  period (2019–2026). A stronger test adds burst bubbles: the Nikkei in 1989, Shanghai in 2007 and
  2015, Bitcoin in 2017 and 2021, oil in 2008, and dot-com stocks.
* **The event definition is a choice.** The grid (test 5) covers 9 variants. Peak-to-trough
  definitions or "regime change" labels are untested.
* **Momentum threshold.** To match alarm rates, the momentum threshold is a quantile over the whole
  sample, a mild look-ahead that *favours* the baseline in the backtest only. The AP comparisons need no
  threshold.
* **Null model.** One simulated path per ticker by default. The stationary bootstrap also removes
  genuine long-memory effects that are not bubbles.
* **Scan density.** Results are for the 10-day / 25-day-window scan. A denser scan (`step: 5`) can be
  validated with `lppl validate --label <label>` once it finishes.
