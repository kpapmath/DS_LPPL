"""Benchmark optimisers used to evaluate the GA (notebook 04)."""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy import optimize


def differential_evolution(objective, bounds: Sequence[tuple[float, float]],
                           rng: np.random.Generator, maxiter: int = 100, popsize: int = 15):
    """scipy differential evolution with the batched objective (``vectorized=True``).

    The returned evaluation count is the number of *candidates* evaluated (scipy's
    ``nfev`` counts batched calls when ``vectorized=True``)."""
    count = [0]

    def batched(x):
        x = np.atleast_2d(np.asarray(x).T)  # scipy passes (D, S) - or (D,) when polishing
        count[0] += x.shape[0]
        return objective(x)

    res = optimize.differential_evolution(
        batched, bounds=list(bounds), maxiter=maxiter, popsize=popsize, tol=1e-10,
        seed=rng, vectorized=True, updating="deferred", polish=True,
    )
    return np.asarray(res.x), float(res.fun), count[0]


def multistart_nelder_mead(objective, bounds: Sequence[tuple[float, float]],
                           rng: np.random.Generator, n_starts: int = 20, maxiter: int = 300):
    """Nelder-Mead from ``n_starts`` uniform random points; keep the best."""
    lo = np.array([b[0] for b in bounds])
    hi = np.array([b[1] for b in bounds])
    best_x, best_f, n_evals = None, np.inf, 0
    for x0 in lo + rng.random((n_starts, lo.size)) * (hi - lo):
        res = optimize.minimize(lambda x: float(objective(np.asarray(x)[None])[0]), x0,
                                method="Nelder-Mead", bounds=list(bounds),
                                options={"maxiter": maxiter})
        n_evals += int(res.nfev)
        if res.fun < best_f:
            best_x, best_f = np.asarray(res.x), float(res.fun)
    return best_x, best_f, n_evals
