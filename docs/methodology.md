# Methodology: an extended review of LPPLS bubble detection with GA calibration

This document reviews the theory behind the Log-Periodic Power Law Singularity
(LPPLS) model, how the model is estimated, and why and how this project uses a genetic
algorithm (GA). It also covers the main criticisms of the approach and how the implementation
responds to them. Section numbers are referenced from the code docstrings.

Contents

1. Bubbles: economic background
2. The Johansen–Ledoit–Sornette (JLS) model
3. From the 7-parameter LPPL to the linearised LPPLS
4. Estimation: a hard nonlinear least-squares problem
5. The genetic algorithm used here
6. Qualification filters
7. Multi-window scanning and the confidence indicator
8. Validation strategy
9. Criticisms, limitations and how this project addresses them
10. Notation summary
11. References

---

## 1. Bubbles: economic background

A *speculative bubble* is a period in which prices rise faster than fundamentals
can justify, sustained by the expectation of further price increases. In the
*rational bubble* framework of Blanchard & Watson (1982), the price is
`p = p* + b`, where `p*` is the fundamental value and the bubble component `b`
grows at the required rate of return while it survives. Investors who hold the
asset are compensated for the risk that the bubble collapses.

Two empirical facts motivate the LPPLS approach:

- **Faster-than-exponential growth.** In many historical bubbles (1929, 1987,
  dot-com 2000, Chinese equities 2007 and 2015, oil 2008) the *growth rate itself*
  increases. In log price the curve is convex and super-exponential, not a straight
  line.
- **Accelerating oscillations.** The rise is decorated by oscillations whose
  period shrinks as the end approaches. This is interpreted as a signature of
  hierarchical herding, a property physicists call *discrete scale invariance*
  (Sornette 1998).

The LPPLS model describes both features and adds a *critical time* `tc`: the
most probable time at which the bubble ends, either in a crash or in a regime change
to a slower trend.

## 2. The Johansen–Ledoit–Sornette (JLS) model

JLS (Johansen, Ledoit & Sornette 2000; Johansen & Sornette 1999) model the price
before a crash as:

```
dp / p = μ(t) dt + σ(t) dW − κ dj
```

- `j` is a jump process that equals 0 before the crash and 1 afterwards. Its
  **hazard rate** `h(t)` is the probability per unit time that the crash happens
  in the next instant, given that it has not happened yet.
- `κ` is the size of the crash, as a fraction of the price.

**The no-arbitrage (martingale) condition.** Rational investors stay in the market
only if they are compensated for the crash risk. The expected drift has to offset the
expected loss:

```
μ(t) = κ h(t)
```

So *the price goes up because the crash risk goes up*. Conditional on no crash
having happened, integrating gives:

```
ln p(t) = ln p(t0) + κ ∫_{t0}^{t} h(t') dt'
```

**Herding and the hazard rate.** Crashes come from collective imitation among
traders. On hierarchical (Ising-like) imitation networks, the hazard rate has a
power-law singularity with log-periodic corrections:

```
h(t) ≈ B' (tc − t)^(m−1) [1 + C' cos(ω ln(tc − t) + φ')]
```

With `0 < m < 1`, the hazard rate diverges at `tc`, but its integral, and therefore the
price, stays finite. Integrating it gives the LPPL equation:

```
ln E[p(t)] = A + B (tc − t)^m + C (tc − t)^m cos(ω ln(tc − t) − φ)        (1)
```

Interpretation of the parameters:

| Parameter | Meaning |
|---|---|
| `A` | log price at `tc` (the finite peak) |
| `B` | strength of the power-law acceleration. `B < 0` means a positive bubble (price accelerating upward); `B > 0` means a negative bubble (accelerating fall) |
| `tc` | critical time: the end of the bubble regime, *not necessarily* a crash date |
| `m` | power-law exponent. Small `m` means a sharp acceleration near `tc` |
| `ω` | log-angular frequency. Successive oscillation peaks satisfy `(tc − t_{n+1}) / (tc − t_n) = exp(−2π/ω)` |
| `C, φ` | amplitude and phase of the log-periodic oscillations |

An important point is that `tc` is only the *most likely* end time. The model
assigns a non-zero probability to the bubble ending smoothly, which is needed for the
no-arbitrage argument to work at all.

## 3. From the 7-parameter LPPL to the linearised LPPLS

Equation (1) has 7 parameters, and fitting all of them directly by nonlinear least squares
is unstable: the cost surface has many quasi-degenerate local minima.
Filimonov & Sornette (2013) expand the cosine with `C1 = C cos φ`, `C2 = C sin φ`:

```
ln p(t) = A + B f(t) + C1 g(t) + C2 h(t)                                     (2)
f = (tc − t)^m,   g = f cos(ω ln(tc − t)),   h = f sin(ω ln(tc − t))
```

For fixed **nonlinear** parameters `θ = (tc, m, ω)`, equation (2) is *linear* in
`(A, B, C1, C2)`. These are then the ordinary-least-squares solution of the
4 × 4 normal equations:

```
[ N      Σf     Σg     Σh   ] [A ]   [ Σy  ]
[ Σf     Σf²    Σfg    Σfh  ] [B ] = [ Σfy ]
[ Σg     Σfg    Σg²    Σgh  ] [C1]   [ Σgy ]
[ Σh     Σfh    Σgh    Σh²  ] [C2]   [ Σhy ]
```

The calibration becomes a **3-dimensional** search over `θ` with the profile cost:

```
F(θ) = min_{A,B,C1,C2} Σ_i [ ln p(t_i) − LPPLS(t_i; θ, A, B, C1, C2) ]²        (3)
```

Filimonov & Sornette show that this "slaving" of the linear parameters removes the
quasi-periodic structure of the cost along the phase `φ` and produces a much smoother
landscape. This project uses it everywhere (`lppl.model.lppls.solve_linear`).

**Implementation notes.**

- The normal equations are built from row sums. This is several times faster than forming
  an `(P, N, 4)` design tensor, and is vectorised over a whole GA population `P` at once.
- They are solved after *symmetric diagonal scaling* (`D^{-1/2} XᵀX D^{-1/2}`) with
  a `1e-12` ridge term. This keeps the condition number manageable when `ω` is small
  and `f, g, h` become nearly collinear. A test checks the result against
  `numpy.linalg.lstsq` to relative precision `1e-6`.
- Candidates with `tc ≤ t2` (where `(tc − t)` would not be positive on the window) get
  `F = +∞`.

## 4. Estimation: a hard nonlinear least-squares problem

Even after slaving, `F(θ)` remains difficult:

- **Multi-modality in ω.** Different frequencies fit different subsets of wiggles
  about equally well. The landscape along `ω` has many narrow valleys.
- **Sloppiness in tc and m.** `tc` and `m` trade off against each other: a later
  `tc` with a smaller `m` produces a similar curve on the observed window. The cost is
  flat along a curved ridge (Brée, Challet & Peirano 2013).
- **Boundary optima.** On data without a bubble the best fit often sits at a bound
  (`m → 0` or `1`, `tc → t2`). Such fits must be *recognised*, not just fitted.

Approaches used in the literature:

| Method | Notes |
|---|---|
| Levenberg–Marquardt from a grid of starts | the original JLS practice. Fast locally, but depends heavily on the starts |
| Taboo search followed by Levenberg–Marquardt | Johansen, Ledoit & Sornette (2000) |
| Nelder–Mead / multistart | common in open-source code; cheap, but with no global guarantee |
| Genetic algorithms, differential evolution, CMA-ES | population-based global search; robust to multi-modality and need no gradients |
| Profile likelihood / Bayesian | Filimonov, Demos & Sornette (2017); Chang & Feigenbaum (2006). Give uncertainty, not just a point estimate |

**Why a GA here?**

1. The problem is low-dimensional (3) and bounded, multi-modal, and the objective is
   cheap and *batchable*: a whole population is evaluated in one vectorised call.
2. A GA needs no gradients and is not misled by the non-smooth `+∞` region at `tc ≤ t2`.
3. Crossover between good solutions in different `ω` valleys explores the oscillation
   structure efficiently, and elitism keeps the best basin found so far.
4. Using a local polish (Nelder–Mead) after the GA is a standard *memetic* or hybrid design:
   the GA finds the basin and the simplex search refines the optimum within it.

Notebook 04 compares the GA with SciPy's differential evolution and with multistart
Nelder–Mead on randomly drawn real windows. The comparison covers runtime, the excess SSE over the best
solution found, and the stability of `tc` across seeds.

## 5. The genetic algorithm used here

Source: `src/lppl/optim/ga.py`. It is a real-coded GA with these components:

| Component | Choice | Default |
|---|---|---|
| Encoding | genes in the unit cube `[0,1]^3`, decoded linearly to the bounds | — |
| Initialisation | uniform random | `population_size = 40` |
| Fitness | `F(θ)` from (3); `+∞` for invalid candidates | — |
| Selection | tournament: best of `k` random individuals | `k = 3` |
| Crossover | BLX-α (Eshelman & Schaffer 1993): each child gene uniform in `[min − αd, max + αd]` | `p_c = 0.9`, `α = 0.3` |
| Mutation | Gaussian, per gene with probability `p_m`; `σ_g = max(σ₀ · decay^g, σ_min)` | `p_m = 0.25`, `σ₀ = 0.15`, decay `0.95`, `σ_min = 0.005` |
| Boundary handling | reflection into `[0, 1]`, which keeps diversity better than clipping | — |
| Elitism | the best `E` individuals pass unchanged, so the best fitness never gets worse | `E = 3` |
| Stopping rule | `generations` reached, or `patience` generations with relative improvement `< tol` | `30`, `10`, `1e-6` |
| Polish | bounded Nelder–Mead from the best individual; accepted only if it improves | `maxiter = 200` |

**Search space.** For a window `[t1, t2]` with length `dt = t2 − t1` (in years):

| Parameter | Bounds | Reason |
|---|---|---|
| `tc` | `(t2 + 1 day, t2 + 0.5·dt]` | the model has to be defined on the whole window; looking further ahead than half the window has little support |
| `m` | `[0.01, 1.0]` | `m < 1` gives a singular hazard and a finite price |
| `ω` | `[1, 50]` | wider than the qualification range, so the filter rather than the search bound decides |

The search space is deliberately **wider** than the qualification ranges (section 6).
If they were equal, every fit would pass the `m`/`ω` checks by construction, and fits stuck at a bound
would look qualified.

**Determinism.** Every window gets its own generator from
`SeedSequence([seed, crc32(ticker), ordinal(t2), window_len])`. Results therefore do not
depend on the order of execution or on how many parallel workers there are. A test checks that
`n_jobs = 1` and `n_jobs = 2` give identical output.

**Tuning.** The defaults were chosen on synthetic LPPLS series with 1 % noise. Population 40,
30 generations and polish switched on matched the excess SSE of population 60 / 50 generations
(about 0 %) at roughly 65 % of the cost. Without the polish step, 35–55 % of runs ended above the best
SSE by more than 0.1 %. Notebook 04 repeats this sensitivity analysis on real data.

## 6. Qualification filters

A fit is **qualified** when all of these conditions hold (Sornette et al. 2015;
Financial Crisis Observatory practice). Source: `src/lppl/model/filters.py`.

| Condition | Default | Rationale |
|---|---|---|
| `m_min ≤ m ≤ m_max` | `0.1 ≤ m ≤ 0.9` | keeps the singularity meaningful and excludes fits pinned at the `m` search bounds (0.01 / 1.0) |
| `ω_min ≤ ω ≤ ω_max` | `2 ≤ ω ≤ 25` | rules out "oscillations" that are a trend (low ω) or noise (high ω) |
| `t2 − 0.05·dt ≤ tc ≤ t2 + 0.2·dt` | — | the critical time must be close to the present to count as a signal |
| `tc > t2 + 3 days` | `tc_margin_days = 3` | rejects fits pinned at the `tc` search bound (`t2 + 1 day`), see below |
| Damping `D = m·|B| / (ω·|C|) ≥ 0.5` | — | approximately ensures a non-negative hazard rate (Bothmer & Meister 2003 give the exact condition `−Bm − |C|√(m²+ω²) ≥ 0`) |
| Oscillations `O = (ω/π)·ln((tc − t1)/(tc − t2)) ≥ 2.5` | — | at least 2.5 half-periods in the window, so the log-periodicity is actually observed and not just fitted |
| finite SSE (optional RMSE cap) | — | numerical sanity |

**Boundary fits: a finding from this project.** With the looser filter that is common in
the literature (`0.01 ≤ m ≤ 0.99`, no `tc` margin), **24 % of qualified fits** in the
default scan had `tc` within 3 days of `t2`, and **18 %** had `m` at its lower search bound
0.01. These fits have converged to the edge of the search box, which describes "the
bubble is ending right now" and is not an interior optimum. They inflated the confidence
on several dates (for example MSFT in mid-2019 and AVGO in March 2024 reached 0.85). The
two tighter conditions above remove them. The scan stores every fit's parameters, so
`lppl requalify` re-applies new filters without refitting. A test checks that this gives
exactly the same result as a fresh scan.

The **sign** comes from `B`: `B < 0` means a positive bubble (super-exponential rise),
and `B > 0` means a negative bubble, also called an anti-bubble (super-exponential fall).

## 7. Multi-window scanning and the confidence indicator

A single fit depends strongly on the arbitrary choice of `t1`. Following the
Financial Crisis Observatory, for each analysis date `t2` we fit **every** window
length `L ∈ {750, 725, …, 125}` trading days (26 windows by default) and compute:

```
pos_conf(t2) = #{windows qualified with B < 0} / #{windows}
neg_conf(t2) = #{windows qualified with B > 0} / #{windows}
```

together with the quartiles of `tc` over the qualified fits. A high `pos_conf` means
that bubble signatures are robust across time scales. One window that happens to pass is
not considered a signal.

Analysis dates are taken every `step` trading days (default 10), counted backwards from
the last date available, so the most recent day is always included. Windows that would
start before the first available price are skipped. Young listings such as META
therefore get fewer windows in their first years.

**Reading the indicator.** Practitioners treat `pos_conf` values above roughly 0.2–0.3,
sustained over several weeks, as a warning. This is a *descriptive* statistic and not a
calibrated probability. Section 8 explains how it should be validated.

**Refined indicators** (`src/lppl/signals.py`, [combined_model.md](combined_model.md) §1). These
are computed from the stored fits plus a second scan of **short** windows (40–180 days, the short
scale of the DS LPPLS indicator):

* `conf_short`: the confidence on the short scale;
* `conf_agree = √(conf · conf_short)`: high only when both scales agree;
* `conf_cluster`: qualified fits whose `tc` agree with the median `tc`;
* `conf_resid`: qualified fits with stationary residuals (a Dickey–Fuller test; Lin, Ren &
  Sornette 2014);
* `lagr_pos`: whether the window selected by **Lagrange regularisation** is qualified. The cost
  χ²(N) = SSE/(N − 7) has its linear trend in the window size N removed before the minimum is taken
  (Demos & Sornette 2017).

On the burst-bubble universe, none of them beats plain `pos_conf`. The residual test passes 98.9 %
of qualified fits, so it adds nothing. The sharpest signal, but also the rarest, is agreement of the
two scales.

## 8. Validation strategy

| Level | What | Where |
|---|---|---|
| Algebra | the linear solve matches `lstsq`; exact recovery of `(A, B, C1, C2)` on noise-free data | `tests/test_lppls.py` |
| Optimiser | sphere and Rastrigin benchmarks, determinism, monotone elitism, bounds, early stopping | `tests/test_ga.py` |
| Calibration | recovers `tc` within ±0.05 yr in ≥ 95 % of seeded noisy trials, and agrees with differential evolution and Nelder–Mead on noise-free data | `tests/test_ga.py` |
| Indicator | a synthetic bubble scores higher than a random walk, and results are reproducible and independent of parallelism | `tests/test_indicator.py` |
| Real data | historical case studies (dot-com 2000, the 2007 peak, Tesla 2021, NVIDIA 2024) | notebook 03, `docs/model_presentation.md` |
| Benchmark | GA vs. differential evolution vs. multistart Nelder–Mead on random real windows | notebook 04 |

**Statistical validation** (`lppl validate`, [validation.md](validation.md)) tests the
indicator against references: the base rate, a momentum score, circular time-shift placebos,
and a bubble-free null model (a stationary block bootstrap of returns, rescanned). It does this for
four labels fixed in advance (3- and 6-month drawdowns, major tops, rally ends), with Holm-adjusted
p-values, a test of the `tc` window against permuted windows, walk-forward re-tuning and a
de-risking backtest.

**Incremental value** (`lppl combine`, [combined_model.md](combined_model.md)) is tested with
nested walk-forward logistic models (market features, then + LPPLS, then + refinements), scored by
out-of-sample log-loss.

Results:

| universe | result |
|---|---|
| 12 large US stocks, 2019–2026 | no significant skill |
| the same 12, 1987–2026 | anticipates major tops (Holm p = 0.03) and fires more on real than on bubble-free prices, but no skill for drawdowns and no gain in the combined model |
| 16 burst-bubble assets, 1987–2026 | a significant, calibrated signal that fires more on real than on bubble-free prices, but does not time the top and does not improve a market-feature forecast out of sample |

Still not implemented: GARCH null paths, Johansen & Sornette's epsilon-drawdowns, and
profile-likelihood intervals for `tc` (Filimonov, Demos & Sornette 2017).

## 9. Criticisms, limitations and how this project addresses them

1. **Overfitting and data snooping.** With 7 parameters and a free choice of window, almost any
   convex series can be fitted (Feigenbaum 2001). *Response:* fixed, pre-registered filter
   ranges in the config, plus rejection of fits pinned at search bounds (§6); confidence measured across many windows; every run's full
   configuration and git SHA stored in `fit_runs`.
2. **`tc` is poorly identified.** The cost surface is flat along the `tc`–`m` ridge
   (Brée, Challet & Peirano 2013). *Response:* we report the quartiles of `tc` across
   qualified windows instead of a single point estimate, and notebook 04 measures `tc`
   dispersion across seeds.
3. **Statistical tests of log-periodicity.** Residuals are autocorrelated, so classical
   F-tests are invalid. Lomb periodograms and unit-root tests on the residuals have been proposed (Lin, Ren &
   Sornette 2014; Geraskin & Fantazzini 2013). *Partly implemented:* a Dickey–Fuller test on the
   residuals (`conf_resid` in `signals.py`). Almost every qualified fit passes it, so it does not
   discriminate. The stricter Ornstein–Uhlenbeck and Lomb tests remain open.
4. **Low predictive power out of sample.** Brée & Joseph (2013) and others find weak
   predictive ability for crash timing. *Response:* the documentation presents the
   indicator as a regime diagnostic, not as a timing or trading signal. *Measured here:*
   * **timing:** the predicted `tc` window contained the top in only 2 of 66 burst-bubble episodes
     (validation.md §6);
   * **a genuine regime signal:** on burst bubbles, an alarm roughly doubles the chance of a fall,
     and it fires more on real than on bubble-free prices;
   * **no out-of-sample gain** over market features in a walk-forward forecast (combined_model.md §4). False positives are expected
   (strong secular trends such as NVDA 2023–24 can look like a bubble for long stretches).
5. **Non-stationarity of the universe.** Survivorship bias: today's Magnificent Seven are
   *ex post* winners. The burst-bubble universe (`config/bubbles.yaml`) has the opposite selection:
   it was chosen *for* having bubbles. The indicator on these names should not be read as an unbiased estimate
   of how often bubbles occur.
6. **Data issues.** Yahoo Finance revises adjusted prices when dividends and splits are applied,
   and has occasional gaps. *Response:* validation rules, hashed snapshots, and offline
   rebuilds from snapshots (`docs/reproducibility.md`).
7. **The choice of time unit matters little; calendar vs trading time matters more.** We use calendar
   decimal years (365.25 days). Weekends then compress no time, while `tc` is a calendar date.
   Trading-day time is a reasonable alternative and a simple change to `dates_to_years`.

## 10. Notation summary

| Symbol | Code | Meaning |
|---|---|---|
| `t1, t2` | `t1, t2` | window start and end (analysis date) |
| `dt` | `t2 - t1` | window length in years |
| `tc` | `tc`, `tc_date` | critical time (decimal year / date) |
| `m, ω` | `m`, `omega` | exponent, log-angular frequency |
| `A, B, C1, C2` | `a, b, c1, c2` | linear parameters |
| `C` | `fit.C` | `√(C1² + C2²)` |
| `D` | `damping` | `m|B| / (ω|C|)` |
| `O` | `n_osc` | `(ω/π) ln((tc − t1)/(tc − t2))` |
| `F` | `sse` | sum of squared log-price residuals |

## 11. References

BibTeX entries are in [`references.bib`](references.bib).

- Blanchard, O. J., & Watson, M. W. (1982). Bubbles, rational expectations and financial markets. *NBER Working Paper 945*.
- Bothmer, H.-C. G. v., & Meister, C. (2003). Predicting critical crashes? A new restriction for the free variables. *Physica A*, 320, 539–547.
- Brée, D. S., Challet, D., & Peirano, P. P. (2013). Prediction accuracy and sloppiness of log-periodic functions. *Quantitative Finance*, 13(2), 275–280.
- Brée, D. S., & Joseph, N. L. (2013). Testing for financial crashes using the log periodic power law model. *International Review of Financial Analysis*, 30, 287–297.
- Chang, G., & Feigenbaum, J. (2006). A Bayesian analysis of log-periodic precursors to financial crashes. *Quantitative Finance*, 6(1), 15–36.
- Demos, G., & Sornette, D. (2017). Birth or burst of financial bubbles: which one is easier to diagnose? *Quantitative Finance*, 17(5), 657–675.
- Eshelman, L. J., & Schaffer, J. D. (1993). Real-coded genetic algorithms and interval-schemata. *Foundations of Genetic Algorithms 2*, 187–202.
- Feigenbaum, J. A. (2001). A statistical analysis of log-periodic precursors to financial crashes. *Quantitative Finance*, 1(3), 346–360.
- Filimonov, V., & Sornette, D. (2013). A stable and robust calibration scheme of the log-periodic power law model. *Physica A*, 392(17), 3698–3707.
- Filimonov, V., Demos, G., & Sornette, D. (2017). Modified profile likelihood inference and interval forecast of the burst of financial bubbles. *Quantitative Finance*, 17(8), 1167–1186.
- Geraskin, P., & Fantazzini, D. (2013). Everything you always wanted to know about log-periodic power laws for bubble modeling but were afraid to ask. *European Journal of Finance*, 19(5), 366–391.
- Goldberg, D. E. (1989). *Genetic Algorithms in Search, Optimization and Machine Learning*. Addison-Wesley.
- Holland, J. H. (1975). *Adaptation in Natural and Artificial Systems*. University of Michigan Press.
- Johansen, A., & Sornette, D. (1999). Critical crashes. *Risk*, 12(1), 91–94.
- Johansen, A., Ledoit, O., & Sornette, D. (2000). Crashes as critical points. *International Journal of Theoretical and Applied Finance*, 3(2), 219–255.
- Holm, S. (1979). A simple sequentially rejective multiple test procedure. *Scandinavian Journal of Statistics*, 6(2), 65–70.
- Lin, L., Ren, R. E., & Sornette, D. (2014). The volatility-confined LPPL model: a consistent model of 'explosive' financial bubbles with mean-reverting residuals. *International Review of Financial Analysis*, 33, 210–225.
- Nelder, J. A., & Mead, R. (1965). A simplex method for function minimization. *The Computer Journal*, 7(4), 308–313.
- Sornette, D. (1998). Discrete scale invariance and complex dimensions. *Physics Reports*, 297(5), 239–270.
- Sornette, D. (2003). *Why Stock Markets Crash: Critical Events in Complex Financial Systems*. Princeton University Press.
- Sornette, D., Woodard, R., Yan, W., & Zhou, W.-X. (2013). Clarifications to questions and criticisms on the Johansen–Ledoit–Sornette financial bubble model. *Physica A*, 392(19), 4417–4428.
- Politis, D. N., & Romano, J. P. (1994). The stationary bootstrap. *Journal of the American Statistical Association*, 89(428), 1303–1313.
- Sornette, D., Demos, G., Zhang, Q., Cauwels, P., Filimonov, V., & Zhang, Q. (2015). Real-time prediction and post-mortem analysis of the Shanghai 2015 stock market bubble and crash. *Journal of Investment Strategies*, 4(4), 77–95.
- Storn, R., & Price, K. (1997). Differential evolution: a simple and efficient heuristic for global optimization over continuous spaces. *Journal of Global Optimization*, 11, 341–359.
