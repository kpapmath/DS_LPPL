"""A small, dependency-free real-coded genetic algorithm.

Design (see docs/methodology.md, section 4):

* genes live in the unit hypercube and are decoded linearly to the search bounds;
* tournament selection, BLX-alpha crossover, Gaussian mutation with a step size
  that decays geometrically over generations;
* elitism (the best ``elite`` individuals survive unchanged), so the best fitness
  never gets worse;
* early stopping after ``patience`` generations without relative improvement;
* optional Nelder-Mead "polish" of the best individual (hybrid GA).

The objective is *batched*: it receives an (P, D) array and returns (P,) values
to minimise. All randomness comes from the supplied ``numpy.random.Generator``,
so a given seed always gives the same result.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

import numpy as np
from scipy.optimize import minimize as _sp_minimize

BatchObjective = Callable[[np.ndarray], np.ndarray]


@dataclass(frozen=True)
class GAConfig:
    population_size: int = 40
    generations: int = 30
    elite: int = 3
    tournament_size: int = 3
    crossover_rate: float = 0.9
    blx_alpha: float = 0.3
    mutation_rate: float = 0.25  # per gene
    mutation_sigma: float = 0.15  # initial std-dev, in unit-cube coordinates
    mutation_decay: float = 0.95  # sigma_g = sigma * decay**g
    mutation_sigma_min: float = 0.005
    patience: int = 10
    tol: float = 1e-6  # relative improvement counted as "progress"
    polish: bool = True
    polish_maxiter: int = 200

    def __post_init__(self):
        if self.population_size < 4:
            raise ValueError("population_size must be >= 4")
        if not 0 <= self.elite < self.population_size:
            raise ValueError("elite must be in [0, population_size)")
        if self.tournament_size < 1:
            raise ValueError("tournament_size must be >= 1")


@dataclass
class GAResult:
    x: np.ndarray
    fun: float
    generations: int
    n_evals: int
    history: list[float] = field(default_factory=list)
    stopped_early: bool = False
    polished: bool = False


# --------------------------------------------------------------------------- operators
def tournament_select(fitness: np.ndarray, n: int, k: int, rng: np.random.Generator) -> np.ndarray:
    """Pick ``n`` indices; each is the best of ``k`` uniformly drawn contestants."""
    contestants = rng.integers(0, fitness.size, size=(n, k))
    winners = np.argmin(fitness[contestants], axis=1)
    return contestants[np.arange(n), winners]


def blx_crossover(
    a: np.ndarray, b: np.ndarray, alpha: float, rng: np.random.Generator
) -> np.ndarray:
    """BLX-alpha: child gene uniform in [min - alpha*d, max + alpha*d], d = |a - b|."""
    lo = np.minimum(a, b)
    hi = np.maximum(a, b)
    d = hi - lo
    return rng.uniform(lo - alpha * d, hi + alpha * d)


def gaussian_mutation(
    pop: np.ndarray, rate: float, sigma: float, rng: np.random.Generator
) -> np.ndarray:
    mask = rng.random(pop.shape) < rate
    return pop + mask * rng.normal(0.0, sigma, size=pop.shape)


def reflect_unit(x: np.ndarray) -> np.ndarray:
    """Map values back into [0, 1] by reflection at the borders (keeps diversity
    better than clipping, which piles individuals onto the bounds)."""
    x = np.mod(x, 2.0)
    return np.where(x > 1.0, 2.0 - x, x)


# --------------------------------------------------------------------------- driver
def minimize(
    objective: BatchObjective,
    bounds: Sequence[tuple[float, float]],
    config: GAConfig | None = None,
    rng: np.random.Generator | None = None,
) -> GAResult:
    cfg = config or GAConfig()
    rng = rng if rng is not None else np.random.default_rng()
    lo = np.array([b[0] for b in bounds], dtype=float)
    hi = np.array([b[1] for b in bounds], dtype=float)
    if np.any(hi < lo):
        raise ValueError("each bound must satisfy low <= high")
    span = hi - lo
    dim = lo.size

    def decode(u: np.ndarray) -> np.ndarray:
        return lo + u * span

    def evaluate(u: np.ndarray) -> np.ndarray:
        f = np.asarray(objective(decode(u)), dtype=float)
        return np.where(np.isfinite(f), f, np.inf)

    P = cfg.population_size
    pop = rng.random((P, dim))
    fit = evaluate(pop)
    n_evals = P
    history = [float(fit.min())]
    stall = 0
    gen = 0
    n_children = P - cfg.elite

    for gen in range(1, cfg.generations + 1):
        sigma = max(cfg.mutation_sigma * cfg.mutation_decay ** (gen - 1), cfg.mutation_sigma_min)
        order = np.argsort(fit, kind="stable")
        elite_pop, elite_fit = pop[order[: cfg.elite]], fit[order[: cfg.elite]]

        ia = tournament_select(fit, n_children, cfg.tournament_size, rng)
        ib = tournament_select(fit, n_children, cfg.tournament_size, rng)
        children = pop[ia].copy()
        do_cx = rng.random(n_children) < cfg.crossover_rate
        if do_cx.any():
            children[do_cx] = blx_crossover(pop[ia][do_cx], pop[ib][do_cx], cfg.blx_alpha, rng)
        children = reflect_unit(gaussian_mutation(children, cfg.mutation_rate, sigma, rng))

        child_fit = evaluate(children)
        n_evals += n_children
        pop = np.vstack([elite_pop, children])
        fit = np.concatenate([elite_fit, child_fit])

        best = float(fit.min())
        prev = history[-1]
        improved = (np.isinf(prev) and np.isfinite(best)) or (prev - best) > cfg.tol * max(abs(prev), 1e-300)
        history.append(best)
        stall = 0 if improved else stall + 1
        if stall >= cfg.patience:
            break

    i_best = int(np.argmin(fit))
    x_best = decode(pop[i_best])
    f_best = float(fit[i_best])
    result = GAResult(
        x=x_best, fun=f_best, generations=gen, n_evals=n_evals, history=history,
        stopped_early=gen < cfg.generations,
    )

    if cfg.polish and np.isfinite(f_best):
        res = _sp_minimize(
            lambda x: float(evaluate(((np.asarray(x) - lo) / np.where(span > 0, span, 1))[None])[0]),
            x_best,
            method="Nelder-Mead",
            bounds=list(zip(lo, hi, strict=True)),
            options={"maxiter": cfg.polish_maxiter, "xatol": 1e-8, "fatol": 1e-12},
        )
        result.n_evals += int(res.nfev)
        if np.isfinite(res.fun) and res.fun < f_best:
            result.x = np.clip(np.asarray(res.x, dtype=float), lo, hi)
            result.fun = float(res.fun)
            result.polished = True
    return result
