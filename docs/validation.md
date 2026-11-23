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
5. End-to-end pipeline: tuned vs baseline (2019–2026, 2026-11-21; configs retired)
6. Burst-bubble universe and pre-registered labels (2026-11-22; configs retired)
7. Full-period pipeline: the 12 instruments from 1987 (`make pipeline-full`, 2026-11-22)
8. Robust variant: several seeds and persistence (pre-registered 2026-11-23)
9. Limitations and next steps

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
| 10 | **Label suite** | tests 1, 3 and 4 repeated for each pre-registered label (`lppl/labels.py`, see §6); p-values Holm-adjusted together with test 11 | Holm p < 0.05 |
| 11 | **`tc` window** | for the first alarm of each episode: is a major top inside the predicted `[tc_q25, tc_q75]`? Placebo: the window offsets are permuted across episodes (2,000 times) | hit rate clearly above the permuted windows |

All randomness is seeded (`seed` in the config), so a report is reproducible for a given scan.

## 3. Running it

```bash
make validate-quick        # ~20 s, app image: tests 1–6 and 9 (no null-model rescans, no MLflow)
make validate              # ~18 min, tuning image: all 9 tests, logged to MLflow (lppl-validation)
make validate LABEL=quick  # validate another scan label
make pipeline-full         # 12 instruments over their whole history, every step (~5 h, §7)

# or directly
lppl validate --label default [--run-id ID] [--null-paths N] [--no-walk-forward] [--no-mlflow]
```

**Outputs** go to `reports/validation/<label>_<run>/`: one CSV per test (`skill.csv`,
`calibration.csv`, `bootstrap.csv`, `placebo.csv`, `grid.csv`, `timing.csv`, `walk_forward.csv`,
`null_model.csv`, `backtest.csv`, `backtest_curves.csv`, `events.csv`, `labels.csv`,
`tc_window.csv`, `tc_episodes.csv`), plus `summary.json` (key
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
([methodology.md](methodology.md) §9). On assets with real burst bubbles, the picture changes; see §6.

## 5. End-to-end pipeline: tuned vs baseline

> **Retired configs.** The configs of this run (`config/pipeline*.yaml`) were removed on 2026-11-23 to keep
> `config/` to the files in use. They can be recovered with `git show ce9d40e:config/<file>`.

The pipeline (then `make pipeline`; its successor is `make pipeline-full`, §7) ran every stage in one go:

1. ingest prices;
2. baseline scan (`pipeline-base`);
3. GA study (Ray Tune + ASHA) and detector study (Optuna);
4. `lppl tune adopt`;
5. tuned scan (`pipeline-tuned`);
6. full validation of both scans;
7. `lppl compare-reports`, written to `reports/pipeline_<id>.md`.

**Configuration.** The scan used `config/pipeline.yaml`: the defaults with an analysis date every
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

## 6. Burst-bubble universe and pre-registered labels

§4 and §5 tested 12 large US stocks that kept growing, over 2019–2026, with about 20 alarm
episodes and few bursts. This section repeats the test where LPPLS should work best: assets with
**documented bubbles that burst**, over long histories.

### Design: fixed before looking at the results

**Universe** (`config/bubbles.yaml`, retired: `git show ce9d40e:config/bubbles.yaml`): 16 assets with data from 1985 and analysis dates from 1987.

| group | assets (bubble episodes) |
|---|---|
| indices | S&P 500 (1987, 2000, 2007), Nasdaq (2000), Nikkei 225 (1989), Hang Seng (1997, 2007, 2018), Shanghai Composite (2007, 2015) |
| other assets | Bitcoin (2017, 2021), gold futures (2011) |
| dot-com stocks | Cisco, Qualcomm, Oracle |
| recent manias | ARK Innovation ETF, GameStop (2021), Zoom (2020), Alibaba, NVIDIA, Tesla |

Oil is left out because WTI futures closed at a negative price in April 2020.

**Model settings.** These are the pipeline's adopted config (`config/pipeline_tuned.yaml`): the
tuned GA and the default detector filters. **Nothing was tuned on these assets**, so the whole
sample is out of sample for the model. The scan density is the standard one: an analysis date every
10 trading days and 26 windows of 125–750 days. Early in a series, only the windows that fit the
history are used.

**Labels** (`src/lppl/labels.py`, `config/validation.yaml → labels`): four labels defined in advance.

| label | the event after analysis date t |
|---|---|
| `drawdown_63` | the original: within 63 days, a fall from the running peak ≥ 1.5 σ √63 |
| `drawdown_126` | the same within 126 days |
| `major_top_126` | a **major top** within 126 days: the highest close of the previous year and of the next 6 months, followed within 6 months by a fall ≥ 1.5 σ √126. This is what LPPLS claims, the *end* of a bubble |
| `rally_end_126` | the price is lower 126 days later |

To this I add the **`tc`-window test** (test 11). The five p-values (one time-shift placebo per
label, plus the `tc` test) are **Holm-adjusted**, so trying several labels cannot create a result by
chance.

**Run:** scan `1e4cd450`. 10,726 labelled dates; 1,728 of them are followed by a `drawdown_63`
event. The scan took 1 h 43 min on 4 cores and the full validation 23 min.

### Verdict

| question (5 % level) | 12 stocks (§4) | **burst bubbles** |
|---|---|---|
| better than random, all dates | ✘ | **✔** |
| better than random, 2024+ | ✘ | ✘ |
| better than momentum | ✘ | ✘ |
| timing matters (beats the time-shifted placebo) | ✘ (p = 0.40) | **✔ (p = 0.001)** |
| higher confidence = more events | ✘ | **✔** |
| fires more on real than on bubble-free prices | ✘ (p = 0.18) | **✔ (p = 0.005)** |
| beats placebo: `drawdown_63` / `drawdown_126` / `major_top_126` (Holm) | ✘ / ✘ / ✘ | **✔ / ✔ / ✔** (p = 0.005 each) |
| beats placebo: `rally_end_126` (Holm) | ✘ | ✘ (p = 0.46) |
| major tops inside the `tc` window (Holm) | ✘ (0 of 20) | ✘ (2 of 66, p = 0.79) |

For the 12-stock rows of the new tests, I re-ran the label suite on scan `0df8f184`.

### Skill per label

| label | base rate | AP LPPLS | AP momentum | lift | 95 % interval, LPPLS − base rate | 95 % interval, LPPLS − momentum | precision at alarm |
|---|---|---|---|---|---|---|---|
| `drawdown_63` | 0.161 | 0.183 | **0.198** | 1.13 | [+0.012, +0.035] | [−0.034, +0.001] | 30 % |
| `drawdown_126` | 0.173 | 0.197 | **0.205** | 1.14 | [+0.013, +0.039] | [−0.028, +0.010] | 38 % |
| `major_top_126` | 0.097 | 0.144 | **0.217** | **1.48** | [+0.032, +0.067] | [−0.097, −0.050] | 29 % |
| `rally_end_126` | 0.365 | 0.365 | 0.341 | 1.00 | [−0.006, +0.008] | [+0.010, +0.037] | 41 % |

* **An alarm doubles the event rate.** Alarms (confidence ≥ 0.3) are followed by a 3-month
  drawdown in **30 %** of cases, against 16 % on a random date. They are followed by a major top
  within 6 months in 29 % of cases, against 10 %. Alarms are rare, though: they fire on 2.1 % of dates
  and catch about 4–5 % of events.
* **Calibration rises steadily.** The event rate is 15.4 % at confidence 0, 16.8 % up to 0.1, 19.9 %
  below 0.3, 29.6 % below 0.6, and 37.5 % at 0.6 or more (16 dates).
* **Momentum still has the higher AP**, clearly so for major tops. Momentum ranks *every* date, and
  big tops by definition follow big rallies, so it covers most events. LPPLS is silent on 83 % of
  dates.
* **`rally_end_126`:** LPPLS has no information about whether the price is merely lower 6 months
  later.

### Does LPPLS add anything beyond momentum?

This analysis is **exploratory and was not pre-registered**. I ran it after seeing that momentum
has the higher AP. It is a logistic regression of each label on 10 momentum-decile dummies plus the
LPPLS confidence, with 95 % intervals from a month-block bootstrap (300 resamples).

| label | LPPLS coefficient (per SD) | 95 % interval | odds ratio, confidence 0.3 vs 0 | event rate in the top momentum quintile: alarm vs confidence 0 |
|---|---|---|---|---|
| `drawdown_63` | +0.135 | [+0.068, +0.190] | ×1.73 | 33.5 % (n = 176) vs 16.1 % |
| `drawdown_126` | +0.155 | [+0.084, +0.225] | ×1.90 | 41.8 % (n = 170) vs 17.6 % |
| `major_top_126` | +0.105 | [+0.042, +0.176] | ×1.57 | 30.7 % (n = 150) vs 15.5 % |

**Yes.** Among assets with the same momentum, those with an LPPLS alarm are about twice as likely to
fall. In other words, LPPLS tells apart, *among strong rallies*, the ones that are about to break.
Momentum alone cannot do that. The AP comparison misses it because AP rewards ranking all dates, not
picking out a few.

*Follow-up* ([combined_model.md](combined_model.md) §4.3): with all market features as controls
(12-month momentum, the volatility ratio and the drawdown from the high), the coefficient about
halves. In honest walk-forward prediction it no longer improves the forecast.

### Null model, `tc` window and timing

* **Null model.** Alarms fire on 2.6 % of real dates and 2.0 % of bubble-free simulated dates
  (p = 0.005). The gap is largest for:
  * Hang Seng: 3.8 % vs 0.4 %;
  * Bitcoin: 2.1 % vs 0.5 %;
  * GameStop: 1.5 % vs 0.3 %;
  * Nikkei: 1.1 % vs 0 %;
  * gold: 5.7 % vs 2.9 %.

  The S&P 500, Nasdaq and Oracle fire about equally on both, and Tesla fires more on the simulated
  paths.
* **The `tc` window is not informative.** Of 66 alarm episodes (first alarm), only **2** had a
  major top inside `[tc_q25, tc_q75]`. Windows reassigned to other episodes hit 4 % of the time on
  average. The windows are narrow (median 16.5 days), and the predicted date should not be read as a
  forecast.
* **Timing.** Across 64 complete episodes, the price kept rising a median **+18 %** for
  **85 trading days (about 4 months)** after the first alarm, then fell a median **−23 %**.

### Walk-forward (detector re-tuned on earlier years only)

| test year | base rate | momentum | LPPLS as scanned | LPPLS re-tuned on the past |
|---|---|---|---|---|
| 2021 | 0.158 | 0.194 | 0.209 | **0.215** |
| 2022 | 0.256 | 0.275 | 0.260 | **0.271** |
| 2023 | 0.020 | 0.040 | 0.024 | **0.044** |
| 2024 | 0.063 | 0.121 | 0.068 | 0.069 |
| 2025 | 0.188 | 0.272 | 0.186 | **0.198** |
| 2026 | 0.205 | 0.284 | **0.233** | 0.208 |
| **mean** | 0.148 | **0.198** | 0.163 | 0.168 |

With 30 years of training data, re-tuning improves the following year in 5 of 6 years. On the
12-stock universe it improved only 1 of 6. More history gives the tuning something real to learn.
Momentum still has the higher AP in 2022–2026, a period with few of the universe's big bubbles.

*Correction made in this round:* the walk-forward and the detector study rejected any setting that
used fewer than 5 windows on **some** date. On long histories, the earliest dates fit only a few
windows, so every trial was rejected and the "re-tuned" column silently equalled the stored
detector. The check now considers only dates where the full window grid fits (`min_windows_used`).
The pipeline studies of §5 were not affected, because every date there had all 26 windows.

### Backtest

| | annual return | Sharpe | max drawdown |
|---|---|---|---|
| buy & hold (1987–2026) | 23.2 % | 0.97 | −55.7 % |
| LPPLS de-risk | 23.0 % | 0.98 | −55.2 % |
| momentum de-risk (same alarm rate) | 22.4 % | 0.97 | −55.1 % |
| LPPLS placebo | 22.9 % | 0.97 | −55.2 % |

Alarms are too rare (2 % of dates), and halving exposure is too mild, to change portfolio results.
The informative alarms are a small part of the total exposure.

### Conclusion

On assets with real bubbles, **the LPPLS confidence carries a statistically significant signal**:

* it passes the time-shift placebo, the null model and calibration;
* three of the four pre-registered labels pass after Holm correction;
* it adds information beyond momentum, roughly doubling the chance of a fall among equally strong
  rallies.

It is **not** a timing tool. The predicted crash window rarely contains the top, the alarm comes
about 4 months early, and on its own its AP is below momentum. The useful reading is "this rally
has the shape of a bubble, so a large fall in the next 3–6 months is about twice as likely". It is
not "the crash comes on `tc`".

The 12-stock result of §4 is consistent with this. That universe had few bursts, so there was little
to detect.

## 7. Full-period pipeline: the 12 instruments from 1987

§4 and §5 cover 2019–2026 only. `make pipeline-full` (`scripts/full_pipeline.sh`) runs every step
on the **same 12 instruments over their entire history**: data from 1985, analysis dates from
1987, or from each listing date (Google 2004, Tesla 2010, Meta 2012, …).

**Steps:**

1. ingest;
2. baseline scan (`full-base`, default settings);
3. GA study (Ray Tune) and detector study (Optuna, 300 trials, validation period 2024+);
4. adoption, only if validated;
5. tuned scan (`full-tuned`);
6. full validation of both scans, with the walk-forward starting in **1995** (31 yearly folds;
   `config/full_validation.yaml`);
7. comparison (`reports/full_<id>.md`);
8. short-scale scan (`full-short`);
9. combined model from 1995.

Scans, validation and the combined model run on the host (4 cores); tuning runs in the
`lppl-tune` image.

**Run of 2026-11-22.** About 5 h of compute in total:

| step | time |
|---|---|
| baseline scan | 1 h 36 min |
| tuning | 8 min |
| tuned scan | 1 h 18 min |
| validation | 31 + 25 min |
| short scan | 22 min |
| combined model | 1.5 min |

There are 8,598 labelled dates, 1,215 of them followed by a `drawdown_63` event, and 78 alarm
episodes.

### Tuning and adoption

| study | baseline | best tuned | adopted |
|---|---|---|---|
| GA (`ga-full-20261122`, benchmark windows from 1987) | loss 1.754 | **0.897** | ✔ (same settings as in §5) |
| detector (`indicator-full-20261122`, train 1987–2023) | AP 2024+ 0.1197 | 0.1134 | ✘ (over-fits again) |

The baseline AP on 2024+ is identical to §5's (0.1197). That is expected and confirms
determinism: the 2024+ fits have the same seeds and dates in both scans.

### Verdict, baseline vs tuned

| question (5 % level) | `full-base` | `full-tuned` |
|---|---|---|
| better than random / than momentum (`drawdown_63`) | ✘ / ✘ | ✘ / ✘ |
| timing matters (time-shift placebo) | ✘ (p = 0.24) | ✘ (p = 0.23) |
| higher confidence = more events | ✘ | ✘ |
| fires more on real than on bubble-free prices | **✔ (4.0 % vs 3.0 %, p = 0.004)** | **✔ (4.1 % vs 3.1 %, p = 0.002)** |
| `drawdown_63` / `drawdown_126` (Holm) | ✘ / ✘ (p = 0.98 / 0.99) | ✘ / ✘ |
| **`major_top_126` (Holm)** | **✔ (p = 0.03)** | **✔ (p = 0.03)** |
| `rally_end_126` (Holm) | ✘ | ✘ |
| tops inside the `tc` window | ✘ (0 of 78) | ✘ (0 of 77) |

The tuned GA changes nothing material: every number moves in the third decimal. The details below
are for `full-base`.

### Skill per label

| label | base rate | AP LPPLS | AP momentum | 95 % interval, LPPLS − base | 95 % interval, LPPLS − momentum |
|---|---|---|---|---|---|
| `drawdown_63` | 0.141 | 0.144 | 0.145 | [−0.003, +0.013] | [−0.021, +0.014] |
| `drawdown_126` | 0.153 | 0.155 | 0.149 | [−0.004, +0.011] | [−0.011, +0.021] |
| `major_top_126` | 0.080 | **0.096** | **0.125** | **[+0.008, +0.031]** | [−0.056, −0.009] |
| `rally_end_126` | 0.294 | 0.285 | 0.263 | [−0.014, −0.001] | [+0.008, +0.034] |

* **Only major tops are anticipated better than chance** (lift 1.21). Momentum does even better
  (lift 1.58).
* **The drawdown labels show no skill over 40 years.** Precision at alarm is 12.9 %, against a
  14.1 % base rate. Calibration is not monotone: the event rate is 14.2 % at confidence 0, 11.2 %,
  17.4 %, 12.4 %, and 20 % at ≥ 0.6 (20 dates).
* **Null model.** The alarm rate on real prices is significantly higher, but the gap is uneven:
  * much higher on real prices: Microsoft 11.2 % vs 1.9 % (the 1990s bubble), Broadcom 7.8 % vs
    1.5 %, Meta 4.0 % vs 1.2 %;
  * reversed: Tesla 0.2 % vs 4.2 %, Amazon 1.9 % vs 4.3 %;
  * roughly equal for the indices.
* **Timing.** Across 78 episodes, the price kept rising a median +20 % for **105 trading days**
  after the first alarm, then fell a median −19 %. The `tc` window never contained the top.
* **Walk-forward, 1995–2026 (31 folds).** The mean AP is 0.170 as scanned and 0.166 re-tuned on
  the past, against momentum 0.224 and a base rate of 0.153. Re-tuning helped in 10 of 31 years.
* **Backtest, 1987–2026.**

  | strategy | annual return | Sharpe | max drawdown |
  |---|---|---|---|
  | buy & hold | 26.5 % | 1.01 | −62.3 % |
  | LPPLS de-risk | 26.0 % | 1.00 | −61.2 % |
  | momentum de-risk | 26.7 % | 1.03 | −60.1 % |
  | LPPLS placebo | 26.0 % | 1.01 | −61.4 % |

### Combined model (1995–2026)

Long scan `full-tuned`, short scan `full-short`; 7,200–7,500 out-of-sample dates per label
([combined_model.md](combined_model.md) §5).

| label | market vs momentum | **+ LPPLS vs market** | + refined vs + LPPLS | Holm p |
|---|---|---|---|---|
| `drawdown_63` | +1.3 % (p = 0.09) | −0.16 % [−0.30, −0.02] | −0.24 % | 1.0 / 1.0 |
| `drawdown_126` | **+2.1 %** (p = 0.03) | −0.22 % [−0.42, −0.02] | −0.47 % | 1.0 / 1.0 |
| `major_top_126` | −1.9 % | −0.15 % [−0.72, +0.40] | −0.59 % | 1.0 / 1.0 |

As on the burst-bubble universe, **LPPLS does not improve a forecast that already uses market
features**, and the refinements make it slightly worse. None of the 6 pre-registered tests passes.

### Conclusion

Over its whole history, the 12-stock universe confirms both earlier pictures:

* **LPPLS detects something real but narrow.** It fires more on real than on bubble-free prices
  (p = 0.004, driven by Microsoft in the 1990s, Broadcom and Meta), and it anticipates **major
  tops** better than chance after correction.
* **It does not anticipate drawdowns in general**, does not time the top (0 of 78 windows), is
  beaten by momentum on every label, and adds nothing to a market-feature forecast.
* **The difference from the burst-bubble universe (§6)** is the share of real bubbles. These are
  large companies that mostly kept growing, so most drawdowns are ordinary corrections, which
  LPPLS does not claim to predict.

## 8. Robust variant: several seeds and persistence

**Why.** The full-period signal is real but fragile: detector tuning over-fitted three times out of
three, the `tc` window contained the top in 0 of 78 episodes, and a window's vote depends on one GA
run. The variant asks whether a *steadier* signal is also a *better* one, without tuning anything.

### Design: fixed before the run (2026-11-23)

**Variant** (`config/full_robust.yaml`, label `full-robust`): `config/full_tuned.yaml` (same 12
instruments, analysis dates, windows, tuned GA, frozen default filters) plus a `robust:` section:

* **3 GA seeds per window.** Seed 0 is the ordinary scan's fit. A window counts only if at least
  **2 of the 3** seeds are qualified with the same sign and their `tc` agree within
  0.25 × (median `tc` − `t2`) + 5 days. The stored fit is the agreeing seed with the lowest SSE;
  `n_seeds_ok` records the size of the agreeing group.
* **Persistence.** The stored confidence is the minimum of the raw confidence over the last
  **3 analysis dates** (about 30 trading days): the signal must stay on. It is causal.
* **Filters frozen** at the defaults (the detector tuning was never adopted). Walk-forward
  re-tuning is therefore skipped for this run; the null model rescans the real prices with the
  same variant so both sides use the same procedure.

**Baseline:** `full-tuned` (run `0cc0bafd`), validated with the same `config/full_validation.yaml`.

**Pre-registered questions and decision rule:**

| | question | measure | passes if |
|---|---|---|---|
| P1 | is it steadier? | flicker = mean \|Δ conf\| between consecutive analysis dates; number of alarm episodes (conf ≥ 0.3 after ≥ 45 days off) | both lower than the baseline |
| P2 | is it better on what LPPLS detects? | paired month-block bootstrap (1,000 reps, same dates) of the AP difference on `major_top_126` | difference > 0 with p < 0.05 |
| P3 | does it still pass its own validation? | label suite (`major_top_126`, Holm) and null model | both p < 0.05 |

**Adopt** the variant as the default signal if P1 and P3 pass and the P2 point estimate is ≥ 0
(steadier at no loss of skill). Claim it is *better* only if P2 passes. Reported as secondary, with
no decision attached: the other three labels, precision at alarm, the `tc` window test, the
backtest and the combined model (market + lppls vs market).

### Results (run `27c39baf`, 2026-11-23)

The scan took 4 h 20 min on 4 cores (3.3× the ordinary scan), the validation 2 h (the null model
rescans real and simulated prices with three seeds). Report: `reports/robust_variant.md` and
`reports/robust_variant_analysis.md`.

| | question | baseline `full-tuned` | variant `full-robust` | result |
|---|---|---|---|---|
| P1 | flicker (mean \|Δ conf\|) | 0.0201 | **0.0083** (−59 %) | |
| P1 | alarm episodes | 80 | **40** | **passes** |
| P2 | AP `major_top_126` (paired, 8,774 dates) | 0.0951 | 0.0868 | diff −0.0084, 95 % CI [−0.0154, −0.0032]: **fails, significantly worse** |
| P3 | `major_top_126` beats placebo (Holm) | p = 0.03 | p = 0.21 (raw 0.043) | **fails** |
| P3 | fires more on real than bubble-free prices | 4.1 % vs 3.1 %, p = 0.004 | 1.15 % vs 0.89 %, p = 0.12 | **fails** |

**Decision (pre-registered rule): not adopted.** The variant is steadier, but it loses the one skill
the baseline had.

**Secondary results** (no decision attached):

* `rally_end_126` AP rises from 0.2824 to 0.2864 (paired CI [+0.0004, +0.0072], one-sided
  p = 0.017). That is not significant once the four labels are corrected for, and the label still
  does not beat its placebo.
* Precision at alarm 0.131 vs 0.127; `tc` window 1 of 39 vs 0 of 77; backtest 26.3 % vs 26.0 % a
  year (buy & hold 26.5 %).
* Combined model: market + LPPLS vs market −0.15 %, −0.20 %, −0.14 % log-loss on the three labels,
  as for the baseline: **no gain**.

### Which part did what (exploratory)

Both components can be rebuilt from the stored fits, so each was evaluated alone on the same dates:

| signal | flicker | episodes | AP `major_top_126` (diff vs baseline, 95 % CI) |
|---|---|---|---|
| baseline | 0.0201 | 80 | 0.0951 |
| seed vote only | 0.0200 | 77 | 0.0953 (+0.0001 [−0.0015, +0.0024]) |
| persistence only | 0.0083 | 38 | 0.0866 (−0.0086 [−0.0151, −0.0032]) |
| both | 0.0083 | 40 | 0.0868 (−0.0084 [−0.0154, −0.0031]) |

* **The seed vote changes almost nothing.** The vote kept 94 % of the baseline's qualified windows,
  and three seeds agreed on 6,348 of the 9,429 windows with any qualified seed. With the tuned GA, the
  optimiser is already stable; its randomness was not where the signal's fragility came from.
* **Persistence accounts for all of the steadiness and all of the lost skill.** Requiring the signal
  to stay on for three analysis dates removes short bursts, and those bursts carried the information
  about major tops. The alarms that remain come later in the rally, which suits "the price will be
  lower in six months" slightly better (`rally_end_126`) and major tops worse.

**Conclusion.** The fragility of the LPPLS signal is not optimiser noise. Smoothing it trades its
skill for steadiness. The signal's information is in short, early bursts of qualified fits, so any
robustness step should keep them. A persistence rule is the wrong tool; pooling across instruments
(market breadth) or modelling the residuals is the next candidate.

## 9. Limitations and next steps

* **Sample size.** In the 12-stock universe there are only about 20 alarm episodes, so "not
  significant" there also means "a small effect cannot be ruled out". The burst-bubble universe (§6)
  has 66.
* **Universe selection.** The burst-bubble assets were chosen *because* they had bubbles. That makes
  them a fair test of "does LPPLS see bubbles that happened" (the null model and placebos control for
  it), but it overstates how often alarms pay off on a random asset.
* **The event definition is a choice.** There are four pre-registered labels with a Holm correction
  (§6), plus a grid of 9 drawdown variants (test 5). The momentum-controlled regression in §6 is
  exploratory and should be confirmed on new data.
* **Momentum threshold.** To match alarm rates, the momentum threshold is a quantile over the whole
  sample, a mild look-ahead that *favours* the baseline in the backtest only. The AP comparisons need no
  threshold.
* **Null model.** One simulated path per ticker by default. The stationary bootstrap also removes
  genuine long-memory effects that are not bubbles.
* **Done since: the combined model and the refined signals.** These are in
  [combined_model.md](combined_model.md).
  * A short-scale indicator, scale agreement, `tc` clustering, a residual test and Lagrange
    regularisation of `t1` did not beat the plain confidence.
  * In a walk-forward logistic model, the LPPLS confidence does **not** improve on momentum,
    volatility and drawdown features (0 of 6 pre-registered tests).
  * The momentum-controlled regression of §6 holds in-sample, with the effect about halved once
    all market features are controlled for. It does not carry over out of sample.
* **Robustness (§8).** Three GA seeds per window and a persistence rule made the signal steadier
  but cost its skill on major tops; not adopted.
* **Scan density.** Results are for the 10-day / 25-day-window scan. A denser scan (`step: 5`) can be
  validated with `lppl validate --label <label>` once it finishes.
