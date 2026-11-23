"""Robust variant of the confidence: seed voting and persistence (lppl.model.indicator)."""

from types import SimpleNamespace

import numpy as np
import pandas as pd
import pytest

from lppl.model.indicator import (
    RobustConfig,
    ScanSettings,
    WindowConfig,
    apply_persistence,
    robust_vote,
    scan_series,
)
from lppl.model.lppls import dates_to_years, lppls
from lppl.optim.ga import GAConfig


def _fit(tc, sign=1, sse=1.0, t2=2020.0):
    return SimpleNamespace(tc=tc, sign=sign, sse=sse, t2=t2)


def test_vote_needs_a_majority_that_agrees_on_tc():
    rc = RobustConfig(seeds=3, min_agree=2)
    # two qualified seeds with the same tc -> robust; the lower SSE one is stored
    pick, ok, n = robust_vote([_fit(2020.2, sse=2), _fit(2020.21, sse=1), _fit(2021.5)], [True, True, False], rc)
    assert ok and n == 2 and pick == 1
    # two qualified seeds whose tc disagree -> not robust
    _, ok, n = robust_vote([_fit(2020.1), _fit(2020.9), _fit(2020.5)], [True, True, False], rc)
    assert not ok and n < 2
    # opposite signs do not agree with each other
    _, ok, _ = robust_vote([_fit(2020.2, 1), _fit(2020.2, -1), _fit(2020.2)], [True, True, False], rc)
    assert not ok
    # nothing qualified -> seed 0 stored, not robust
    assert robust_vote([_fit(2020.2)] * 3, [False] * 3, rc) == (0, False, 0)


def test_invalid_config_is_rejected():
    with pytest.raises(ValueError):
        RobustConfig(seeds=2, min_agree=3)
    assert not RobustConfig().enabled and RobustConfig(persistence=2).enabled


def test_persistence_is_a_causal_running_minimum():
    d = pd.bdate_range("2020-01-01", periods=6)
    ind = pd.DataFrame({"ticker": "X", "date": d, "pos_conf": [0, .5, .6, .7, 0, .4], "neg_conf": 0.0})
    out = apply_persistence(ind, 3)
    np.testing.assert_allclose(out["pos_conf"], [0, 0, 0, .5, 0, 0])
    # changing a later value never changes an earlier output
    ind2 = ind.copy()
    ind2.loc[5, "pos_conf"] = 0.9
    np.testing.assert_allclose(apply_persistence(ind2, 3)["pos_conf"].iloc[:5], out["pos_conf"].iloc[:5])


@pytest.fixture(scope="module")
def bubble_prices():
    idx = pd.bdate_range("2018-01-01", periods=420)
    t = dates_to_years(idx)
    rng = np.random.default_rng(5)
    logp = lppls(t, t[-1] + 0.12, 0.5, 8.0, 5.0, -0.6, 0.02, 0.01) + 0.003 * rng.normal(size=len(t))
    return pd.Series(np.exp(logp), index=idx)


def _settings(**robust):
    return ScanSettings(windows=WindowConfig(150, 400, 125), ga=GAConfig(population_size=30, generations=5),
                        robust=RobustConfig(**robust))


def test_seed_zero_is_the_ordinary_scan(bubble_prices):
    """Replica 0 uses the ordinary RNG, so seeds=1 reproduces the ordinary scan exactly and the
    robust scan's stored fit equals it whenever seed 0 is the best agreeing seed."""
    kw = dict(start=bubble_prices.index[-30], step=15, n_jobs=1)
    f1, i1 = scan_series(bubble_prices, "X", ScanSettings(windows=WindowConfig(150, 400, 125),
                                                         ga=GAConfig(population_size=30, generations=5)), **kw)
    f2, i2 = scan_series(bubble_prices, "X", _settings(seeds=1), **kw)
    pd.testing.assert_frame_equal(i1, i2)
    f3, i3 = scan_series(bubble_prices, "X", _settings(seeds=3, min_agree=2, persistence=2), **kw)
    assert len(f3) == len(f1) and f3["n_seeds_ok"].between(0, 3).all()
    assert (f3["n_evals"] > f1["n_evals"]).all()                 # all seeds are counted
    assert (i3["pos_conf"] <= i3["pos_conf"].cummax()).all() and len(i3) == len(i1)
