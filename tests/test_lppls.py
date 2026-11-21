import numpy as np
import pandas as pd
import pytest

from lppl.model.lppls import (
    LPPLSFit,
    dates_to_years,
    lppls,
    make_fit,
    solve_linear,
    years_to_date,
)


def test_lppls_matches_hand_computation():
    t, tc, m, w = 2020.0, 2021.0, 0.5, 6.0
    A, B, C1, C2 = 1.0, -0.5, 0.1, 0.2
    tau = tc - t
    expected = A + B * tau**m + tau**m * (C1 * np.cos(w * np.log(tau)) + C2 * np.sin(w * np.log(tau)))
    assert lppls(t, tc, m, w, A, B, C1, C2) == pytest.approx(expected)


def test_lppls_is_nan_at_and_after_tc():
    out = lppls(np.array([2020.0, 2021.0, 2022.0]), 2021.0, 0.5, 6.0, 1, -1, 0.1, 0.1)
    assert np.isfinite(out[0])
    assert np.isnan(out[1:]).all()


def test_linear_parameters_recovered_exactly(synthetic):
    t, y, p = synthetic
    coef, sse = solve_linear(t, y, p["tc"], p["m"], p["omega"])
    np.testing.assert_allclose(coef[0], [p["A"], p["B"], p["C1"], p["C2"]], rtol=1e-6, atol=1e-8)
    assert sse[0] < 1e-12


def test_solve_linear_vectorised_matches_loop(synthetic):
    t, y, _ = synthetic
    tc = np.array([2022.1, 2022.5, 2023.0])
    m = np.array([0.3, 0.6, 0.9])
    w = np.array([5.0, 9.0, 14.0])
    coef, sse = solve_linear(t, y, tc, m, w)
    for i in range(3):
        c1, s1 = solve_linear(t, y, tc[i], m[i], w[i])
        np.testing.assert_allclose(coef[i], c1[0], rtol=1e-10)
        assert sse[i] == pytest.approx(s1[0])


def test_solve_linear_matches_numpy_lstsq(synthetic):
    t, y, _ = synthetic
    rng = np.random.default_rng(0)
    y = y + 0.01 * rng.normal(size=y.size)
    tc, m, w = 2022.4, 0.7, 11.0
    tau = tc - t
    X = np.column_stack([np.ones_like(t), tau**m, tau**m * np.cos(w * np.log(tau)),
                         tau**m * np.sin(w * np.log(tau))])
    ref, *_ = np.linalg.lstsq(X, y, rcond=None)
    coef, sse = solve_linear(t, y, tc, m, w)
    np.testing.assert_allclose(coef[0], ref, rtol=1e-6, atol=1e-8)
    assert sse[0] == pytest.approx(np.sum((y - X @ ref) ** 2), rel=1e-8)


def test_sse_is_inf_when_tc_inside_window(synthetic):
    t, y, _ = synthetic
    _, sse = solve_linear(t, y, np.array([2021.0, 2022.5]), 0.5, 8.0)
    assert np.isinf(sse[0]) and np.isfinite(sse[1])


def test_sse_non_negative(synthetic):
    t, y, _ = synthetic
    rng = np.random.default_rng(1)
    _, sse = solve_linear(t, y, 2022.0 + rng.random(50), rng.uniform(0.01, 1, 50),
                          rng.uniform(1, 50, 50))
    assert (sse >= 0).all()


def test_fit_object_derived_quantities(synthetic):
    t, y, p = synthetic
    fit = make_fit(t, y, p["tc"], p["m"], p["omega"])
    assert isinstance(fit, LPPLSFit)
    assert fit.C == pytest.approx(np.hypot(p["C1"], p["C2"]), rel=1e-6)
    assert fit.damping == pytest.approx(p["m"] * abs(p["B"]) / (p["omega"] * fit.C), rel=1e-6)
    expected_osc = p["omega"] / np.pi * np.log((p["tc"] - t[0]) / (p["tc"] - t[-1]))
    assert fit.n_oscillations == pytest.approx(expected_osc)
    assert fit.sign == 1  # B < 0 -> positive bubble
    np.testing.assert_allclose(fit.predict(t), y, atol=1e-8)
    d = fit.to_dict()
    assert {"tc", "m", "omega", "damping", "n_oscillations", "rmse", "sign"} <= d.keys()


def test_time_conversion_round_trip():
    dates = pd.to_datetime(["1995-01-03", "2000-03-10", "2024-02-29", "2026-11-20"])
    years = dates_to_years(dates)
    assert years[0] == pytest.approx(1995.0, abs=0.01)
    assert np.all(np.diff(years) > 0)
    for d, y in zip(dates, years, strict=True):
        assert years_to_date(y) == d


def test_time_conversion_handles_timezones():
    naive = dates_to_years(pd.to_datetime(["2024-01-02"]))
    aware = dates_to_years(pd.to_datetime(["2024-01-02"]).tz_localize("America/New_York"))
    assert naive[0] == pytest.approx(aware[0])
