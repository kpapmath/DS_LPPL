import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from lppl.model.calibrate import SearchBounds, fit_window
from lppl.optim.ga import (
    GAConfig,
    blx_crossover,
    gaussian_mutation,
    minimize,
    reflect_unit,
    tournament_select,
)


def sphere(x):
    return np.sum((x - 0.3) ** 2, axis=1)


def rastrigin(x):
    return 10 * x.shape[1] + np.sum(x**2 - 10 * np.cos(2 * np.pi * x), axis=1)


# ------------------------------------------------------------------ operators
def test_tournament_prefers_better_individuals():
    fit = np.arange(100, dtype=float)
    idx = tournament_select(fit, 5000, 3, np.random.default_rng(0))
    assert idx.mean() < 30  # E[min of 3 uniform] ~ 25
    assert idx.min() >= 0 and idx.max() < 100


def test_tournament_size_one_is_uniform():
    idx = tournament_select(np.arange(10.0), 20000, 1, np.random.default_rng(0))
    counts = np.bincount(idx, minlength=10)
    assert counts.min() > 1700


def test_blx_children_within_expanded_interval():
    rng = np.random.default_rng(0)
    a = rng.random((1000, 3))
    b = rng.random((1000, 3))
    c = blx_crossover(a, b, 0.3, rng)
    d = np.abs(a - b)
    assert np.all(c >= np.minimum(a, b) - 0.3 * d - 1e-12)
    assert np.all(c <= np.maximum(a, b) + 0.3 * d + 1e-12)


def test_mutation_rate_zero_is_identity():
    pop = np.random.default_rng(0).random((10, 3))
    np.testing.assert_array_equal(gaussian_mutation(pop, 0.0, 0.1, np.random.default_rng(1)), pop)


@given(st.lists(st.floats(-10, 10, allow_nan=False), min_size=1, max_size=50))
def test_reflect_maps_into_unit_interval(xs):
    out = reflect_unit(np.array(xs))
    assert np.all((out >= 0) & (out <= 1))


def test_reflect_is_identity_inside():
    x = np.linspace(0, 1, 11)
    np.testing.assert_allclose(reflect_unit(x), x)


# ------------------------------------------------------------------ driver
def test_ga_minimises_sphere():
    res = minimize(sphere, [(-5, 5)] * 3, GAConfig(polish=False), np.random.default_rng(0))
    np.testing.assert_allclose(res.x, 0.3, atol=0.05)


def test_ga_with_polish_is_precise():
    res = minimize(sphere, [(-5, 5)] * 3, GAConfig(), np.random.default_rng(0))
    np.testing.assert_allclose(res.x, 0.3, atol=1e-4)


def test_ga_handles_multimodal_rastrigin():
    cfg = GAConfig(population_size=80, generations=80, patience=30)
    res = minimize(rastrigin, [(-5.12, 5.12)] * 2, cfg, np.random.default_rng(3))
    assert res.fun < 1.0  # global minimum 0; local minima have f >= 1


def test_ga_is_deterministic_given_seed():
    a = minimize(rastrigin, [(-5, 5)] * 2, GAConfig(), np.random.default_rng(123))
    b = minimize(rastrigin, [(-5, 5)] * 2, GAConfig(), np.random.default_rng(123))
    np.testing.assert_array_equal(a.x, b.x)
    assert a.history == b.history


def test_elitism_makes_best_fitness_monotone():
    res = minimize(rastrigin, [(-5, 5)] * 3, GAConfig(polish=False), np.random.default_rng(5))
    h = np.array(res.history)
    assert np.all(np.diff(h) <= 0)


@settings(max_examples=15, deadline=None)
@given(st.integers(0, 10_000))
def test_result_respects_bounds(seed):
    bounds = [(1.0, 2.0), (-3.0, -1.0)]
    res = minimize(lambda x: -np.sum(x, axis=1), bounds, GAConfig(generations=10),
                   np.random.default_rng(seed))
    assert 1.0 <= res.x[0] <= 2.0 and -3.0 <= res.x[1] <= -1.0
    # optimum is the upper corner
    np.testing.assert_allclose(res.x, [2.0, -1.0], atol=1e-2)


def test_infinite_objective_values_are_tolerated():
    def f(x):
        out = sphere(x)
        out[x[:, 0] < 0] = np.inf
        return out

    res = minimize(f, [(-1, 1)] * 2, GAConfig(), np.random.default_rng(0))
    assert np.isfinite(res.fun)
    assert res.x[0] >= 0


def test_early_stopping():
    res = minimize(lambda x: np.zeros(len(x)), [(0, 1)], GAConfig(generations=100, patience=5,
                                                                 polish=False),
                   np.random.default_rng(0))
    assert res.stopped_early and res.generations == 5


@pytest.mark.parametrize("bad", [dict(population_size=2), dict(elite=40), dict(tournament_size=0)])
def test_invalid_config(bad):
    with pytest.raises(ValueError):
        GAConfig(**bad)


# ------------------------------------------------------------------ LPPLS calibration
def test_fit_window_recovers_noise_free_parameters(synthetic):
    t, y, p = synthetic
    res = fit_window(t, y, rng=np.random.default_rng(0))
    assert res.fit.tc == pytest.approx(p["tc"], abs=1e-3)
    assert res.fit.m == pytest.approx(p["m"], abs=1e-2)
    assert res.fit.omega == pytest.approx(p["omega"], abs=1e-1)
    assert res.fit.B == pytest.approx(p["B"], rel=1e-2)


@pytest.mark.slow
def test_fit_window_recovers_tc_with_noise_across_seeds(synthetic):
    """Acceptance criterion from docs/PLAN.md: >= 95% of seeded trials within tolerance."""
    t, y0, p = synthetic
    hits = 0
    n = 20
    for s in range(n):
        rng = np.random.default_rng(s)
        y = y0 + 0.01 * rng.normal(size=y0.size)
        res = fit_window(t, y, rng=rng)
        hits += abs(res.fit.tc - p["tc"]) < 0.05  # ~18 days on a 2-year window
    assert hits / n >= 0.95


def test_fit_window_methods_agree(synthetic):
    t, y, p = synthetic
    for method in ("de", "nm"):
        res = fit_window(t, y, method=method, rng=np.random.default_rng(0))
        assert res.fit.tc == pytest.approx(p["tc"], abs=1e-2), method


def test_fit_window_respects_tc_bounds(random_walk_prices):
    from lppl.model.lppls import dates_to_years
    px = random_walk_prices.iloc[:300]
    t = dates_to_years(px.index)
    b = SearchBounds()
    res = fit_window(t, np.log(px.to_numpy()), bounds=b, rng=np.random.default_rng(0))
    lo, hi = b.for_window(t[0], t[-1])[0]
    assert lo <= res.fit.tc <= hi
    assert res.fit.tc > t[-1]


def test_fit_window_input_validation():
    with pytest.raises(ValueError):
        fit_window(np.arange(5.0), np.arange(5.0))
    with pytest.raises(ValueError):
        fit_window(np.arange(20.0), np.arange(20.0), method="bogus")
