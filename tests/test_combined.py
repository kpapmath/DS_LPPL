"""Refined LPPLS signals (lppl.signals) and the walk-forward combined model (lppl.combined)."""

import numpy as np
import pandas as pd
import pytest

from lppl import combined as C
from lppl import signals as S
from lppl.model.lppls import dates_to_years, lppls


# --------------------------------------------------------------------------- residual test
def test_adf_separates_noise_from_random_walk():
    rng = np.random.default_rng(0)
    assert S.adf_tstat(rng.normal(size=400)) < -10
    walks = [S.adf_tstat(np.cumsum(rng.normal(size=400))) for _ in range(50)]
    assert np.mean(np.array(walks) < -2.86) < 0.15  # ~5 % false rejections expected


@pytest.fixture
def lppls_prices():
    idx = pd.bdate_range("2018-01-01", periods=600)
    t = dates_to_years(idx)
    tc, m, om, A, B, C1, C2 = t[-1] + 0.15, 0.5, 8.0, 5.0, -0.6, 0.02, 0.01
    rng = np.random.default_rng(1)
    logp = lppls(t, tc, m, om, A, B, C1, C2) + 0.002 * rng.normal(size=len(t))
    return pd.Series(np.exp(logp), index=idx), dict(tc=tc, m=m, omega=om, a=A, b=B, c1=C1, c2=C2)


def _fits(px, params, lengths, qualified, tc_shift=None, sse=None):
    t2 = px.index[-1]
    rows = []
    for i, L in enumerate(lengths):
        r = {"ticker": "X", "t2": t2, "t1": px.index[-L], "window_len": L, "qualified": qualified[i],
             "sign": 1, **params, "sse": (sse[i] if sse is not None else 1.0)}
        if tc_shift is not None:
            r["tc"] = params["tc"] + tc_shift[i]
        rows.append(r)
    return pd.DataFrame(rows)


def test_residuals_of_a_true_fit_are_stationary(lppls_prices):
    px, p = lppls_prices
    f = _fits(px, p, [200, 400], [True, True])
    assert (S.residual_tstats(f, px) < -2.86).all()


def test_signals_conf_cluster_and_lagrange(lppls_prices):
    px, p = lppls_prices
    # 4 windows: 3 qualified; one of them has a tc far from the others
    f = _fits(px, p, [150, 250, 350, 450], [True, True, True, False],
              tc_shift=[0, 0.002, 0.5, 0], sse=[1.0, 1.5, 2.6, 1.0])
    sig = S.lppls_signals(f, pd.DataFrame({"X": px}), S.SignalSettings()).iloc[0]
    assert sig["conf"] == pytest.approx(0.75)
    assert sig["conf_cluster"] == pytest.approx(0.5)      # the outlying tc is not counted
    assert 0 < sig["tc_proximity"] <= 1
    assert sig["lagr_window"] in (150, 250, 350, 450)
    assert np.isnan(sig["conf_short"])


def test_short_scale_merge_and_agreement(lppls_prices):
    px, p = lppls_prices
    f = _fits(px, p, [150, 250], [True, False])
    short = pd.DataFrame({"ticker": ["X"], "date": [px.index[-1]], "pos_conf": [0.5]})
    sig = S.lppls_signals(f, pd.DataFrame({"X": px}), S.SignalSettings(), short).iloc[0]
    assert sig["conf_short"] == 0.5 and sig["conf_agree"] == pytest.approx(np.sqrt(0.5 * 0.5))


def test_market_features_have_no_lookahead():
    rng = np.random.default_rng(2)
    idx = pd.bdate_range("2015-01-01", periods=1200)
    px = pd.Series(np.exp(np.cumsum(0.01 * rng.normal(size=1200))), index=idx)
    a = S.market_features(px, idx[[900, 1000]])
    px2 = px.copy()
    px2.iloc[1001:] *= 3.0                               # change only the future
    b = S.market_features(px2, idx[[900, 1000]])
    pd.testing.assert_frame_equal(a, b)
    assert set(S.MARKET_FEATURES) == set(a.columns)


# --------------------------------------------------------------------------- model
def test_logistic_recovers_coefficients():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(20000, 2))
    y = (rng.random(20000) < 1 / (1 + np.exp(-(-1 + 1.5 * X[:, 0] - 0.5 * X[:, 1])))).astype(float)
    b = C.fit_logistic(X, y, l2=0.0)
    np.testing.assert_allclose(b, [-1, 1.5, -0.5], atol=0.08)


def _synthetic_features(informative: bool, n_tickers=4, years=range(2000, 2012)):
    rng = np.random.default_rng(4)
    rows = []
    for k in range(n_tickers):
        for d in pd.bdate_range(f"{years[0]}-01-01", f"{years[-1]}-12-31")[::10]:
            mom = rng.normal()
            conf = float(rng.random() < 0.15) * rng.random()
            z = -2 + 0.4 * mom ** 2 + (3 * conf if informative else 0)
            rows.append({"ticker": f"T{k}", "date": d, "conf": conf, "mom_126": mom,
                         "mom_21": rng.normal(), "mom_252": rng.normal(), "vol_ratio": 1 + 0.1 * rng.normal(),
                         "drawdown_252": -abs(rng.normal()), "drawdown_63": float(rng.random() < 1 / (1 + np.exp(-z)))})
    f = pd.DataFrame(rows)
    for c in S.LPPLS_SIGNALS[1:]:
        f[c] = 0.0
    f["conf_short"] = np.nan
    return f


@pytest.mark.parametrize("informative", [True, False])
def test_walk_forward_detects_information_only_when_present(informative):
    s = C.CombinedSettings(first_test_year=2004, min_train=300, bootstrap=200, labels=["drawdown_63"])
    feat = _synthetic_features(informative)
    oos = C.walk_forward(feat, "drawdown_63", s)
    assert oos["year"].min() == 2004 and set(C.MODELS) <= set(oos.columns)
    met, inc = C.compare(oos, s, "drawdown_63")
    r = inc.set_index(["model", "versus"]).loc[("market+lppls", "market")]
    if informative:
        assert r["log_loss_gain"] > 0 and r["p_value"] < 0.05
    else:
        assert r["p_value"] > 0.05


def test_walk_forward_respects_the_embargo(monkeypatch):
    s = C.CombinedSettings(first_test_year=2005, min_train=300, bootstrap=10, labels=["drawdown_63"])
    feat = _synthetic_features(True)
    last_train_date = []
    orig = C.Design.fit

    def spy(df, features, q):
        last_train_date.append(df["date"].max())
        return orig(df, features, q)

    monkeypatch.setattr(C.Design, "fit", staticmethod(spy))
    C.walk_forward(feat, "drawdown_63", s)
    assert last_train_date[0] < pd.Timestamp("2005-01-01") - pd.offsets.BDay(63)
