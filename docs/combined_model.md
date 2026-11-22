# Refined signal and combined model

Validation on assets with real bubbles ([validation.md](validation.md) §6) showed that the LPPLS
confidence carries a significant signal. It passes the time-shift placebo, the null model and
calibration, and it roughly doubles the chance of a fall among equally strong rallies. Two
questions follow:

1. **Do the refinements proposed in the literature make the signal stronger?** (§1)
2. **Does LPPLS add predictive value to a model that already uses ordinary market information?**
   (§2)

`lppl combine` answers both, and notebook `06_combined_model` presents the result. Every setting
was fixed in `config/combined.yaml` before the model was run on any data.

Contents

1. The refined LPPLS signals
2. The combined walk-forward model
3. Running it
4. Results: burst-bubble universe (1995–2026)
5. Results: 12-stock universe (2023–2026, and the full period 1995–2026)
6. Conclusion and limitations

---

## 1. The refined LPPLS signals

All signals are computed per (asset, analysis date) by `src/lppl/signals.py`. They use the stored
fits of a scan, plus a second, short-scale scan. **No fit is recomputed.**

| signal | definition | source |
|---|---|---|
| `conf` | long-scale confidence: share of the 26 windows (125–750 days) with a qualified fit with B < 0 | Sornette et al. (2015) |
| `conf_short` | the same on 8 **short** windows (40–180 days), from a separate scan (`config/*_short.yaml`) | DS LPPLS short scale |
| `conf_agree` | √(`conf` × `conf_short`): high only when **both scales agree** | DS LPPLS |
| `conf_cluster` | share of windows whose qualified `tc` lies close to the median `tc`: \|tc − median\| ≤ 0.25 × (median − t2) + 5 days | `tc` agreement across windows |
| `conf_resid` | share of windows with a qualified fit whose residuals pass a Dickey–Fuller test (t < −2.86) | Lin, Ren & Sornette (2014) |
| `lagr_pos` | 1 if the window chosen by **Lagrange regularisation** holds a qualified fit. The rule is χ²(N) = SSE/(N − 7) with its linear trend in N removed; take the argmin | Demos & Sornette (2017) |
| `tc_proximity` | exp(−(median `tc` − t2) / 63 days) when `conf` > 0, else 0 | how close the predicted end is |

The **short-scale scan** uses the same analysis dates as the long one (dates are counted back from
the last day with the same step), so the two line up row by row. It is cheap:

* 25 minutes for the 16 burst-bubble assets on 4 cores, against 1 h 43 min for the long scan;
* 7 minutes for the 12 stocks.

## 2. The combined walk-forward model

`src/lppl/combined.py` fits four **nested** logistic regressions. Each adds a block of predictors
to the previous one:

| model | predictors |
|---|---|
| `momentum` | 126-day momentum (in σ units) plus 5 hinge terms at its training quantiles, which capture the U-shape (strong rallies *and* falling assets come before drawdowns) |
| `market` | + momentum over 21 and 252 days, the 63-day/750-day volatility ratio, and the current drawdown from the 252-day high |
| `market+lppls` | + `conf` and `conf` × positive momentum |
| `market+refined` | + all refined signals of §1 |

**Fitting details:**
* the slopes are standardised and given an L2 penalty (0.5 · ‖w‖²), with no penalty on the intercept;
* all transforms (knots, means, scales) are fitted on the training rows only.

**Walk-forward protocol:**
* every calendar year from 1995 is predicted by models fitted on **all earlier dates only**;
* an **embargo** equal to the label's look-ahead is applied: 63, 126 or 252 trading days for
  `major_top_126`, which needs 126 days to the top and 126 to confirm the fall;
* a fold is scored once there are at least 1,000 labelled training rows.

**Scores** (out of sample, pooled over all test years):
* **primary: log-loss**, a proper scoring rule that rewards honest probabilities;
* secondary: Brier score, average precision (AP), and the precision of the top 2 % of predictions
  (the "alarms"), whose threshold is set on the training fit.

**Tests:**
* improvements between models get 95 % intervals from a **month-block bootstrap** of the
  out-of-sample predictions (1,000 resamples);
* the **pre-registered family** is two comparisons per label × 3 labels, Holm-adjusted:
  * *market+lppls vs market*: does LPPLS add anything?
  * *market+refined vs market+lppls*: do the refinements add anything more?

**Labels:** the three that passed in validation.md §6 (`drawdown_63`, `drawdown_126`,
`major_top_126`).

## 3. Running it

```bash
make bubbles-short     # short-scale scan of the burst-bubble universe (~25 min on 4 cores)
make combine           # lppl combine --label bubbles --short-label bubbles-short (~2 min)
make combine-pipeline  # the same for the 12-stock universe (short scan ~7 min, then the model)

# or directly
lppl combine --label bubbles --short-label bubbles-short [--first-test-year 1995] [--out DIR]
```

**Output** goes to `reports/combined/<label>_<run>/`:

| file | contents |
|---|---|
| `features.csv` | every signal, market feature and label |
| `oos.csv` | out-of-sample predictions of every model |
| `metrics.csv` | scores per model |
| `increments.csv` | model comparisons with intervals and Holm p-values |
| `per_year.csv` | scores per test year |
| `standalone.csv` | each signal on its own, with the placebo test |
| `coefficients.csv` | full-sample coefficients (descriptive) |
| `summary.json` | key numbers and verdict |
| `settings.yaml` | the settings used |
| `per_year_gain.png`, `calibration.png` | figures |

## 4. Results: burst-bubble universe (1995–2026)

Long scan `1e4cd450`, short scan `9908f575`. Out of sample: 1995–2026, with 9,100–9,600 dates per
label.

### 4.1 Each refined signal on its own

The lift is AP / base rate. The p-value is from the circular time-shift placebo (1,000 shifts).

| signal | dates with a signal | lift, `drawdown_63` | lift, `drawdown_126` | lift, `major_top_126` | placebo p |
|---|---|---|---|---|---|
| `conf` | 17.3 % | 1.13 | 1.14 | 1.48 | ≤ 0.002 |
| `conf_cluster` | 16.4 % | 1.13 | 1.14 | 1.45 | 0.001 |
| `conf_resid` | 17.2 % | 1.13 | 1.14 | 1.49 | ≤ 0.002 |
| `conf_short` | 7.2 % | 1.05 | 1.02 | 1.17 | 0.001–0.09 |
| `conf_agree` | 4.5 % | 1.07 | 1.04 | 1.15 | ≤ 0.006 |
| `lagr_pos` | 2.8 % | 1.03 | 1.04 | 1.08 | ≤ 0.002 |
| `tc_proximity` | 17.3 % | 1.04 | 1.03 | 1.41 | 0.001–0.10 |
| momentum (reference) | 100 % | 1.23 | 1.19 | **2.24** | ≤ 0.005 |

Inside strong rallies (top momentum quintile), the 3-month drawdown rate is:

| signal fires | dates | rate when the signal fires | rate otherwise |
|---|---|---|---|
| `conf` ≥ 0.3 | 176 | 33.5 % | 19.3 % |
| `conf_agree` ≥ 0.3 (both scales) | 47 | **42.6 %** | 20.0 % |
| `conf_cluster` ≥ 0.2 | 170 | 32.9 % | 19.4 % |
| `lagr_pos` | 183 | 31.5 % | 19.5 % |
| `conf_short` ≥ 0.3 | 93 | 31.2 % | 20.0 % |

What this shows:

* **Every refined signal is significant against its own placebo**, but none is *better* than plain
  `conf`.
* **`conf_agree` is the sharpest signal.** Within strong rallies, 43 % of its alarms were followed
  by a 3-month drawdown. It fires on only 47 such dates, though.
* **The residual test does not discriminate.** 98.9 % of qualified fits pass it, so `conf_resid` is
  practically identical to `conf` (correlation 1.00). In-window LPPLS residuals are mean-reverting
  almost by construction; the stricter Ornstein–Uhlenbeck and Lomb tests of Lin et al. would be
  needed.
* **`conf_cluster` ≈ `conf`** (correlation 0.93). Qualified fits already agree on `tc`, because the
  filters bound it tightly.
* **Lagrange regularisation mostly picks the shortest or longest window** (11 % and 13 % of dates).
  This is a known sign of a nearly flat χ²(N). On its own it is weak (lift ≈ 1.03).

### 4.2 Does LPPLS add value to the market model? (pre-registered)

Out-of-sample log-loss (lower is better):

| label | base rate | momentum | market | market + LPPLS | market + refined |
|---|---|---|---|---|---|
| `drawdown_63` | 16.2 % | 0.4397 | **0.4302** | 0.4303 | 0.4321 |
| `drawdown_126` | 17.1 % | 0.4596 | **0.4466** | 0.4493 | 0.4545 |
| `major_top_126` | 9.5 % | 0.2926 | **0.2891** | 0.2906 | 0.2927 |

Change in log-loss with 95 % intervals (positive = better):

| label | market vs momentum | **+ LPPLS vs market** | **+ refined vs + LPPLS** | Holm p (LPPLS / refined) |
|---|---|---|---|---|
| `drawdown_63` | **+2.2 %** [+0.7, +3.6] | −0.03 % [−0.4, +0.4] | −0.4 % [−0.8, −0.0] | 1.0 / 1.0 |
| `drawdown_126` | **+2.8 %** [+1.2, +4.4] | −0.6 % [−1.1, −0.1] | −1.2 % [−1.8, −0.6] | 1.0 / 1.0 |
| `major_top_126` | +1.2 % [−0.5, +3.0] | −0.5 % [−0.8, −0.1] | −0.7 % [−1.2, −0.3] | 1.0 / 1.0 |

(The intervals are shown here as % of the reference model's log-loss.)

* **The market features beat momentum alone**, significantly for the two drawdown labels.
* **Adding the LPPLS confidence does not improve out-of-sample prediction.** For two labels it makes
  it slightly but significantly *worse*. **None of the six pre-registered tests passes.**
* **The refinements make it worse again.** They are 6 correlated columns that each add noise to a
  rare signal.
* **Year by year,** the LPPLS model beats the market model in 16 of 32 years for `drawdown_63`, 12
  of 32 for `drawdown_126` and 10 of 31 for `major_top_126`: a coin toss or worse.
* **Precision of the alarms** (top 2 % of predictions):

  | label | market | market + LPPLS |
  |---|---|---|
  | `drawdown_63` | 39.0 % | **44.2 %** |
  | `drawdown_126` | 34.0 % | **38.3 %** |
  | `major_top_126` | 37.2 % | 33.9 % |

  The bootstrap shows an improvement in 91 % of resamples for both drawdown labels, which is
  suggestive but not significant. It reverses for `major_top_126`.

![out-of-sample gain per year](figures/combined_per_year_gain.png)

**Sensitivity: the early years.** The largest losses are in 1996–2000, when the training history
was short and the dot-com bubble was out of sample. From 2001 on, the losses shrink but no gain
appears:

| label | + LPPLS vs market | + refined vs + LPPLS |
|---|---|---|
| `drawdown_63` | −0.10 % [−0.48, +0.24] | −0.27 % [−0.57, +0.02] |
| `drawdown_126` | −0.29 % [−0.57, −0.00] | −0.51 % [−0.91, −0.09] |
| `major_top_126` | −0.15 % [−0.42, +0.14] | −0.78 % [−1.21, −0.40] |

This check was run after the main result and is not part of the pre-registered family.

![calibration](figures/combined_calibration.png)

Both models are well calibrated up to a predicted probability of about 25 %. In the top decile they
over-predict: a predicted 38–43 % turns into an event rate of 27–29 % for the drawdown labels. The
refined LPPLS terms do not change the calibration; they push the top decile slightly further out.

### 4.3 Why the in-sample signal does not become out-of-sample value

The LPPLS coefficient (per SD of `conf`) in a full-sample logistic regression, with a 95 %
month-block bootstrap interval (200 resamples), is:

| label | controls: momentum spline only | controls: all market features |
|---|---|---|
| `drawdown_63` | +0.125 [+0.061, +0.180] | +0.072 [+0.005, +0.138] |
| `drawdown_126` | +0.151 [+0.087, +0.216] | +0.086 [+0.018, +0.144] |
| `major_top_126` | +0.096 [+0.025, +0.160] | +0.058 [−0.015, +0.114] |

* **About half of what LPPLS sees is already in ordinary market features.** These are 12-month
  momentum, rising volatility and distance from the high (`conf` correlates 0.39 with 6-month
  momentum and 0.21 with the drawdown measure).
* **What remains is real but small.** It is significant in-sample for the two drawdown labels. It
  is concentrated in the 2 % of dates with alarms, so it moves an average log-loss over all dates by
  less than the estimation noise of the extra coefficients.
* A model that predicts *every date* gets almost nothing from a signal that is silent on 83 % of
  them.

## 5. Results: 12-stock universe (2023–2026)

There is little training history (the scans start in 2019), and therefore only 2023–2026 out of
sample, with 560–1,090 dates.

* On its own, `lagr_pos` is the only signal that passes its placebo for both drawdown labels
  (p = 0.035 and 0.009); `tc_proximity` passes for major tops (p = 0.002). These are isolated
  results among 21 tests, so they are not counted as evidence.
* With so little data, **the market model is worse than momentum alone** (log-loss +3.6 %,
  +4.8 % and +13.6 %). Adding LPPLS or the refinements changes nothing significantly (all Holm
  p = 1.0).

As in [validation.md](validation.md) §4, this universe and period are too short to learn anything
from.

### 5.1 The same universe over its full history (`make pipeline-full`)

Long scan `full-tuned` (`0cc0bafd`) and short scan `full-short` (`ac49e3f7`), both from 1987. Out of
sample: 1995–2026, with 7,236–7,464 dates per label.

| label | momentum | market | market + LPPLS | market + refined |
|---|---|---|---|---|
| `drawdown_63` | 0.4110 | **0.4056** | 0.4062 | 0.4072 |
| `drawdown_126` | 0.4379 | **0.4288** | 0.4297 | 0.4317 |
| `major_top_126` | **0.2730** | 0.2783 | 0.2787 | 0.2803 |

*(out-of-sample log-loss, lower is better)*

* **LPPLS vs market:** −0.16 % [−0.30, −0.02], −0.22 % [−0.42, −0.02] and −0.15 % [−0.72, +0.40].
  Every Holm p = 1.0. LPPLS beats the market model in 17, 13 and 16 of 30–31 years: a coin toss.
* **Refined vs LPPLS:** worse on every label (−0.24 %, −0.47 % and −0.59 %).
* **Alarm precision** (top 2 %) does not improve either: 12.2 % vs 12.7 % for `drawdown_63`,
  21.2 % vs 22.3 % for `drawdown_126`, and 5.8 % vs 4.8 % for major tops.
* **On its own,** `conf` passes its placebo only for major tops (p = 0.005, lift 1.21), as do
  `conf_cluster`, `conf_resid` and `tc_proximity`. Nothing passes for the drawdown labels except
  `lagr_pos` (p = 0.044, lift 1.02), an isolated result.
* For major tops, momentum alone beats even the market model. The extra market features over-fit
  for this rarer label.

This confirms the burst-bubble result on a different universe and a longer period: **LPPLS adds no
out-of-sample value to ordinary market features.**

## 6. Conclusion and limitations

**Conclusion.**

* **The refinements of the literature do not make the signal stronger** on this data. The residual
  and `tc`-agreement filters are almost always satisfied by fits that already pass the standard
  filters. Lagrange window selection is weak. Requiring short and long scales to agree gives the
  sharpest but rarest alarms (43 % hit rate inside strong rallies, 47 dates).
* **LPPLS does not add predictive value to a market model** in honest walk-forward prediction. The
  pre-registered answer to "does it help?" is **no**, on every label. The bubble signal of
  validation.md §6 is genuine, but about half of it is shared with momentum, volatility and drawdown
  features. The rest is too small and too rare to improve probability forecasts across all dates.
* **Where it can still help** is at the alarm level: the top 2 % of predictions became 4–5 points
  more precise with LPPLS (not significant). This matches the practical reading of earlier sections:
  an LPPLS alarm is a reason to look closer at a rally, not a forecasting model.

**Limitations.**

* **The model is deliberately simple:** a linear logistic regression with one interaction. Tree
  models could find other interactions, at a higher risk of over-fitting.
* **The market features were chosen from common practice, not tuned.** Other choices (valuation,
  breadth, credit spreads) are not available from price data alone.
* **One parameter set per refinement,** fixed in advance. Different tolerances might help, but
  choosing them on this data would be over-fitting.
* **The residual test** is a plain Dickey–Fuller test. Lin et al. (2014) use stricter
  Ornstein–Uhlenbeck and Lomb-periodogram tests, which are not implemented.
* **The burst-bubble universe was selected for having bubbles** (validation.md §8).
