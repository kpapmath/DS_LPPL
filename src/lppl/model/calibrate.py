"""Calibrate one LPPLS window: search (tc, m, omega), slave (A, B, C1, C2)."""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from lppl.model.lppls import DAYS_PER_YEAR, LPPLSFit, make_fit, solve_linear
from lppl.optim import baseline
from lppl.optim.ga import GAConfig, GAResult
from lppl.optim.ga import minimize as ga_minimize


@dataclass(frozen=True)
class SearchBounds:
    """Search space for the nonlinear parameters.

    ``tc`` is searched in (t2 + tc_min_days, t2 + tc_after * dt]. The lower end stays
    strictly after the last observation so that (tc - t) > 0 on the whole window.
    """

    m: tuple[float, float] = (0.01, 1.0)
    omega: tuple[float, float] = (1.0, 50.0)
    tc_after: float = 0.5
    tc_min_days: float = 1.0

    def for_window(self, t1: float, t2: float) -> list[tuple[float, float]]:
        dt = t2 - t1
        return [
            (t2 + self.tc_min_days / DAYS_PER_YEAR, t2 + self.tc_after * dt),
            self.m,
            self.omega,
        ]


@dataclass
class CalibrationResult:
    fit: LPPLSFit
    n_evals: int
    generations: int
    elapsed_ms: float
    history: list[float]


def make_objective(t: np.ndarray, y: np.ndarray):
    """Batched objective: (P, 3) array of [tc, m, omega] -> (P,) SSE."""

    def objective(x: np.ndarray) -> np.ndarray:
        x = np.atleast_2d(x)
        _, sse = solve_linear(t, y, x[:, 0], x[:, 1], x[:, 2])
        return sse

    return objective


def fit_window(
    t: np.ndarray,
    log_price: np.ndarray,
    *,
    bounds: SearchBounds | None = None,
    method: str = "ga",
    ga_config: GAConfig | None = None,
    rng: np.random.Generator | None = None,
) -> CalibrationResult:
    """Fit LPPLS to one window of (decimal-year time, log-price).

    ``method``: ``"ga"`` (default), ``"de"`` (scipy differential evolution) or
    ``"nm"`` (multi-start Nelder-Mead). The two latter are benchmarks.
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(log_price, dtype=float)
    if t.size != y.size or t.size < 10:
        raise ValueError("need matching t / log_price arrays with at least 10 points")
    rng = rng if rng is not None else np.random.default_rng()
    b = (bounds or SearchBounds()).for_window(float(t[0]), float(t[-1]))
    objective = make_objective(t, y)

    start = time.perf_counter()
    if method == "ga":
        res: GAResult = ga_minimize(objective, b, ga_config, rng)
        x, n_evals, gens, hist = res.x, res.n_evals, res.generations, res.history
    elif method == "de":
        x, _, n_evals = baseline.differential_evolution(objective, b, rng)
        gens, hist = 0, []
    elif method == "nm":
        x, _, n_evals = baseline.multistart_nelder_mead(objective, b, rng)
        gens, hist = 0, []
    else:
        raise ValueError(f"unknown method {method!r}")
    elapsed = (time.perf_counter() - start) * 1000.0

    fit = make_fit(t, y, float(x[0]), float(x[1]), float(x[2]))
    return CalibrationResult(fit=fit, n_evals=int(n_evals), generations=int(gens),
                             elapsed_ms=elapsed, history=list(hist))
