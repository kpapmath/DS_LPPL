import numpy as np
import pandas as pd
import pytest

from lppl.model.filters import FilterConfig
from lppl.model.indicator import (
    ScanSettings,
    WindowConfig,
    aggregate,
    requalify,
    scan_date,
    scan_dates_positions,
    scan_series,
    window_rng,
)
from lppl.model.lppls import dates_to_years

FAST = ScanSettings(windows=WindowConfig(min_len=200, max_len=700, step=100))


def test_window_lengths():
    w = WindowConfig(min_len=125, max_len=750, step=25)
    lens = w.lengths()
    assert lens[0] == 750 and lens[-1] == 125 and len(lens) == 26
    assert w.lengths(available=300) == [300, 275, 250, 225, 200, 175, 150, 125]
    with pytest.raises(ValueError):
        WindowConfig(min_len=100, max_len=50).lengths()


def test_scan_positions_include_last_date_and_respect_history():
    dates = pd.bdate_range("2020-01-01", periods=300)
    pos = scan_dates_positions(dates, None, None, step=10, min_history=125)
    assert pos[-1] == 299
    assert pos[0] >= 124
    assert np.all(np.diff(pos) == 10)
    pos2 = scan_dates_positions(dates, "2020-12-01", "2020-12-31", step=1, min_history=1)
    assert all(pd.Timestamp("2020-12-01") <= dates[p] <= pd.Timestamp("2020-12-31") for p in pos2)
    assert scan_dates_positions(dates, "2030-01-01", None, 5, 1) == []


def test_window_rng_is_deterministic_and_distinct():
    t2 = pd.Timestamp("2024-01-02")
    a = window_rng(42, "AAPL", t2, 250).random(3)
    np.testing.assert_array_equal(a, window_rng(42, "AAPL", t2, 250).random(3))
    assert not np.array_equal(a, window_rng(42, "MSFT", t2, 250).random(3))
    assert not np.array_equal(a, window_rng(42, "AAPL", t2, 275).random(3))
    assert not np.array_equal(a, window_rng(43, "AAPL", t2, 250).random(3))


def test_scan_date_records(bubble_prices):
    dates = pd.DatetimeIndex(bubble_prices.index)
    recs = scan_date(dates, np.log(bubble_prices.to_numpy()), len(dates) - 1, FAST, 0, "SYN")
    assert len(recs) == len(FAST.windows.lengths())
    r = recs[0]
    assert r["t2"] == dates[-1].date()
    assert {"tc", "m", "omega", "qualified", "sign", "tc_date"} <= r.keys()
    assert all(rec["tc"] > dates_to_years(dates[-1:])[0] for rec in recs)


def test_aggregate_confidence_bounds_and_counts():
    fits = pd.DataFrame({
        "ticker": ["X"] * 4, "t2": [pd.Timestamp("2024-01-02").date()] * 4,
        "qualified": [True, True, False, True], "sign": [1, 1, 1, -1],
        "tc": [2024.1, 2024.2, 2030.0, 2024.3],
    })
    out = aggregate(fits).iloc[0]
    assert out["pos_conf"] == 0.5 and out["neg_conf"] == 0.25
    assert out["n_windows"] == 4 and out["n_qualified"] == 3
    assert out["tc_q25_date"] <= out["tc_median_date"] <= out["tc_q75_date"]


def test_aggregate_without_qualified_fits():
    fits = pd.DataFrame({"ticker": ["X"], "t2": [pd.Timestamp("2024-01-02").date()],
                         "qualified": [False], "sign": [1], "tc": [2024.5]})
    out = aggregate(fits).iloc[0]
    assert out["pos_conf"] == 0 and pd.isna(out["tc_median_date"])
    assert aggregate(pd.DataFrame()).empty


@pytest.mark.slow
def test_synthetic_bubble_scores_higher_than_random_walk(bubble_prices, random_walk_prices):
    _, ind_b = scan_series(bubble_prices, "BUB", FAST, step=40, n_jobs=1,
                           start=bubble_prices.index[-121])
    _, ind_r = scan_series(random_walk_prices, "RW", FAST, step=40, n_jobs=1,
                           start=random_walk_prices.index[-121])
    assert ind_b["pos_conf"].between(0, 1).all() and ind_r["pos_conf"].between(0, 1).all()
    assert ind_b["pos_conf"].iloc[-1] >= 0.5
    assert ind_b["pos_conf"].mean() > ind_r["pos_conf"].mean()


def test_scan_series_is_reproducible(bubble_prices):
    kw = dict(step=200, n_jobs=1, master_seed=1, start=bubble_prices.index[-1])
    f1, i1 = scan_series(bubble_prices, "BUB", FAST, **kw)
    f2, i2 = scan_series(bubble_prices, "BUB", FAST, **kw)
    cols = ["tc", "m", "omega", "sse"]
    pd.testing.assert_frame_equal(f1[cols], f2[cols])
    pd.testing.assert_frame_equal(i1, i2)


def test_scan_series_parallel_equals_serial(bubble_prices):
    kw = dict(step=100, master_seed=3, start=bubble_prices.index[-201])
    f1, _ = scan_series(bubble_prices, "BUB", FAST, n_jobs=1, **kw)
    f2, _ = scan_series(bubble_prices, "BUB", FAST, n_jobs=2, **kw)
    np.testing.assert_array_equal(f1["tc"].to_numpy(), f2["tc"].to_numpy())


def test_requalify_equals_fresh_scan_with_new_filters(bubble_prices):
    """Filters do not influence the GA, so re-qualifying stored fits == rescanning."""
    from dataclasses import replace
    loose = replace(FAST, filters=FilterConfig(m_min=0.01, m_max=0.99, tc_margin_days=0))
    kw = dict(step=100, n_jobs=1, master_seed=5, start=bubble_prices.index[-201])
    fits_loose, _ = scan_series(bubble_prices, "BUB", loose, **kw)
    fits_new, ind_new = scan_series(bubble_prices, "BUB", FAST, **kw)
    req = requalify(fits_loose, FAST.filters)
    assert req["qualified"].tolist() == fits_new["qualified"].tolist()
    pd.testing.assert_frame_equal(aggregate(req), ind_new)
