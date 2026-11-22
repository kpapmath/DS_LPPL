"""Validation suite: event table, metrics, significance tests, timing, backtest, persistence."""

import numpy as np
import pandas as pd
import pytest

from lppl import validation as V


# --------------------------------------------------------------------------- fixtures
@pytest.fixture
def prices():
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2017-01-02", periods=2200)
    return pd.DataFrame({t: 100 * np.exp(np.cumsum(0.012 * rng.normal(size=idx.size)))
                         for t in ("AAA", "BBB")}, index=idx)


@pytest.fixture
def ind(prices):
    """Indicator rows every 10 trading days from position 800 onwards."""
    rng = np.random.default_rng(1)
    rows = []
    for t in prices:
        for p in range(800, 2190, 10):
            d = prices.index[p]
            c = float(rng.choice([0, 0, 0, 0.05, 0.1, 0.35, 0.7]))
            rows.append({"ticker": t, "date": d, "pos_conf": c, "neg_conf": 0.0,
                         "tc_q25_date": d + pd.Timedelta(days=10) if c else pd.NaT,
                         "tc_median_date": d + pd.Timedelta(days=20) if c else pd.NaT,
                         "tc_q75_date": d + pd.Timedelta(days=30) if c else pd.NaT})
    return pd.DataFrame(rows)


@pytest.fixture
def settings():
    return V.ValidationSettings(split_date="2023-06-01", bootstrap=50, placebo=50,
                                events=V.EventSettings(horizon=40, vol_window=300),
                                grid=V.GridSettings(horizons=[20, 40], k_sigmas=[1.0, 1.5]),
                                backtest=V.BacktestSettings(placebo_shifts=5))


@pytest.fixture
def ev(ind, prices, settings):
    return V.build_events(ind, prices, settings.events, settings.split_date, settings.embargo_days)


# --------------------------------------------------------------------------- config
def test_default_validation_config_loads():
    s = V.load_validation_config()
    assert s.events.horizon == 63 and s.alarm_threshold == 0.3 and s.bootstrap > 0


# --------------------------------------------------------------------------- events & skill
def test_events_have_labels_splits_and_no_future_leak(ev, settings):
    assert set(ev["split"]) == {"train", "embargo", "valid"}
    assert ev.loc[ev.split == "train", "date"].max() < ev.loc[ev.split == "valid", "date"].min()
    assert ev["label"].dropna().isin([0, 1]).all()
    assert ev["label"].isna().any()  # the last dates have an incomplete horizon


def test_skill_table(ev):
    sk = V.skill(ev, 0.3).set_index("split")
    assert {"all", "train", "valid"} <= set(sk.index)
    for split in sk.index:
        assert 0 <= sk.loc[split, "base_rate"] <= 1
        assert 0 <= sk.loc[split, "ap_lppl"] <= 1


def test_calibration_bands_cover_all_labelled_rows(ev):
    cal = V.calibration(ev)
    assert list(cal["band"].astype(str)) == V.CAL_LABELS
    assert cal["n"].sum() == V._labeled(ev).shape[0]


# --------------------------------------------------------------------------- significance
def test_block_bootstrap_is_seeded_and_brackets_the_point(ev):
    a = V.block_bootstrap(ev, 80, seed=3)
    b = V.block_bootstrap(ev, 80, seed=3)
    pd.testing.assert_frame_equal(a, b)
    base = a[a.metric == "base_rate"]
    assert ((base.ci_low <= base.point + 1e-12) & (base.point <= base.ci_high + 1e-12)).all()


def test_placebo_detects_a_real_signal_and_not_noise(ev):
    d = ev.copy()
    d["conf"] = d["label"].fillna(0) * 0.8 + 0.01 * np.random.default_rng(0).random(len(d))
    strong = V.placebo_test(d, 60, min_shift_rows=10, seed=0)
    assert strong["p_value"] < 0.05 and strong["actual_ap"] > strong["placebo_p95"]
    d["conf"] = np.random.default_rng(1).random(len(d))
    noise = V.placebo_test(d, 60, min_shift_rows=10, seed=0)
    assert noise["p_value"] > 0.05


def test_event_grid_shape(ind, prices, settings):
    g = V.event_grid(ind, prices, settings)
    assert len(g) == 2 * 2 * 2  # horizons x k x splits
    assert g["base_rate"].between(0, 1).all()


# --------------------------------------------------------------------------- timing
def test_timing_finds_episode_and_tc_hit():
    idx = pd.bdate_range("2020-01-01", periods=600)
    px = pd.Series(np.r_[np.linspace(100, 200, 300), np.linspace(200, 120, 300)], index=idx)
    alarm_day = idx[250]
    top = idx[299]
    ev = pd.DataFrame({"ticker": "X", "date": [idx[100], alarm_day, idx[260]],
                       "conf": [0.0, 0.5, 0.4], "tc_q25": [pd.NaT, top - pd.Timedelta(days=5), pd.NaT],
                       "tc_median": [pd.NaT, top, pd.NaT], "tc_q75": [pd.NaT, top + pd.Timedelta(days=5), pd.NaT]})
    tm = V.timing(ev, pd.DataFrame({"X": px}), 0.3)
    assert len(tm) == 1
    r = tm.iloc[0]
    assert r.first_alarm == alarm_day and r.top_date == top and r.top_in_tc_window
    assert r.lead_trading_days == 49 and r.complete
    assert r.fall_after_top == pytest.approx(px.iloc[299:426].min() / px.iloc[299] - 1)


# --------------------------------------------------------------------------- null model helper
def test_stationary_bootstrap_length_and_support():
    r = np.random.default_rng(0).normal(size=500)
    sim = V.stationary_bootstrap(r, 1234, 20, np.random.default_rng(1))
    assert sim.shape == (1234,)
    assert np.isin(sim, r).all()
    again = V.stationary_bootstrap(r, 1234, 20, np.random.default_rng(1))
    np.testing.assert_array_equal(sim, again)


# --------------------------------------------------------------------------- backtest
def test_exposure_has_no_lookahead():
    days = pd.bdate_range("2024-01-01", periods=10)
    e = V._exposure(days[1:], [days[0], days[3], days[6]], np.array([False, True, False]), 0.5)
    # signal of days[3] (alarm) applies from days[4] up to and including days[6]
    expected = [1, 1, 1, 0.5, 0.5, 0.5, 1, 1, 1]
    assert e.tolist() == expected


def test_backtest_without_alarms_equals_buy_and_hold(ev, prices, settings):
    quiet = ev.assign(conf=0.0)
    tab, curves = V.backtest(quiet, prices, settings)
    t = tab.set_index(["period", "strategy"])
    assert t.loc[("all", "lppl"), "total_return"] == pytest.approx(t.loc[("all", "buy_and_hold"), "total_return"])
    assert t.loc[("all", "lppl"), "avg_exposure"] == 1.0
    assert {"buy_and_hold", "lppl", "momentum", "lppl_placebo"} <= set(curves.columns)


def test_backtest_costs_reduce_return(ev, prices, settings):
    cheap = settings.model_copy(update={"backtest": settings.backtest.model_copy(update={"cost_bps": 0})})
    dear = settings.model_copy(update={"backtest": settings.backtest.model_copy(update={"cost_bps": 100})})
    a = V.backtest(ev, prices, cheap)[0].set_index(["period", "strategy"]).loc[("all", "lppl"), "total_return"]
    b = V.backtest(ev, prices, dear)[0].set_index(["period", "strategy"]).loc[("all", "lppl"), "total_return"]
    assert b < a


# --------------------------------------------------------------------------- report
def test_summary_verdict_and_round_trip(ev, ind, prices, settings, tmp_path):
    rep = V.ValidationReport(run_id="r" * 36, label="t", settings=settings)
    rep.tables.update(events=ev, skill=V.skill(ev, 0.3), calibration=V.calibration(ev),
                      bootstrap=V.block_bootstrap(ev, 30, 0), grid=V.event_grid(ind, prices, settings),
                      timing=V.timing(ev, prices, 0.3))
    rep.placebo = {s: V.placebo_test(ev, 30, 5, 0, s) for s in ("conf", "momentum")}
    rep.tables["placebo"] = pd.DataFrame([{k: v for k, v in p.items() if k != "distribution"}
                                          for p in rep.placebo.values()])
    rep.tables["backtest"], rep.curves = V.backtest(ev, prices, settings)
    rep.summary = V.summarize(rep)
    assert set(rep.summary["verdict"]) >= {"beats_base_rate_validation", "beats_time_shifted_placebo"}
    out = V.save_report(rep, tmp_path / "rep")
    for f in ("summary.json", "skill.csv", "calibration.png", "placebo.png", "backtest.png"):
        assert (out / f).exists(), f
    back = V.load_report(out)
    assert back.summary["run_id"] == rep.run_id
    assert set(back.placebo) == {"conf", "momentum"}
    pd.testing.assert_frame_equal(back.tables["skill"], rep.tables["skill"], check_dtype=False)


def test_two_proportion_p():
    assert V.two_proportion_p(50, 1000, 10, 1000) < 0.001  # clearly higher
    assert V.two_proportion_p(10, 1000, 50, 1000) > 0.999  # clearly lower
    assert 0.4 < V.two_proportion_p(20, 1000, 20, 1000) < 0.6  # equal
