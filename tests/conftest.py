from __future__ import annotations

import os

import numpy as np
import pandas as pd
import pytest

from lppl.model.lppls import lppls

TRUE = dict(tc=2022.3, m=0.5, omega=8.0, A=5.0, B=-0.6, C1=0.03, C2=0.02)


@pytest.fixture
def t_grid() -> np.ndarray:
    return np.linspace(2020.0, 2022.0, 400)


@pytest.fixture
def synthetic(t_grid):
    """Noise-free LPPLS log-price with known parameters."""
    return t_grid, lppls(t_grid, **TRUE), TRUE


@pytest.fixture
def bubble_prices() -> pd.Series:
    """A business-day price series that follows an LPPLS bubble (+ small noise)
    towards tc shortly after the last observation."""
    dates = pd.bdate_range("2019-01-01", periods=800)
    from lppl.model.lppls import dates_to_years
    t = dates_to_years(dates)
    tc = t[-1] + 0.08
    rng = np.random.default_rng(7)
    y = lppls(t, tc, 0.45, 9.0, 6.0, -0.9, 0.04, -0.03) + 0.004 * rng.normal(size=t.size)
    return pd.Series(np.exp(y), index=dates, name="adj_close")


@pytest.fixture
def random_walk_prices() -> pd.Series:
    dates = pd.bdate_range("2019-01-01", periods=800)
    rng = np.random.default_rng(11)
    return pd.Series(100 * np.exp(np.cumsum(0.01 * rng.normal(size=dates.size))), index=dates)


@pytest.fixture(scope="session")
def pg_engine():
    """Engine for integration tests: LPPL_DATABASE_URL, or the compose DB on localhost."""
    from sqlalchemy import text

    from lppl.config import DBSettings
    from lppl.data import db

    url = os.environ.get("LPPL_TEST_DATABASE_URL") or DBSettings().url
    eng = db.get_engine(url)
    try:
        with eng.connect() as conn:
            conn.execute(text("select 1"))
    except Exception as exc:  # pragma: no cover
        pytest.skip(f"PostgreSQL not reachable at {url}: {exc}")
    # isolated schema so tests never touch real data
    with eng.begin() as conn:
        conn.execute(text("drop schema if exists lppl_test cascade"))
        conn.execute(text("create schema lppl_test"))
    test_eng = db.get_engine(url).execution_options(schema_translate_map={None: "lppl_test"})
    db.init_db(test_eng)
    yield test_eng
    with eng.begin() as conn:
        conn.execute(text("drop schema if exists lppl_test cascade"))
