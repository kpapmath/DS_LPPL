"""Pre-registered labels, major-top detection, Holm adjustment and the tc-window test."""

import numpy as np
import pandas as pd
import pytest

from lppl import labels as L
from lppl import validation as V


@pytest.fixture
def bubble():
    """1000 noisy days up, a top at position 1200, then a 40 % crash, then flat."""
    rng = np.random.default_rng(0)
    idx = pd.bdate_range("2010-01-01", periods=1600)
    trend = np.r_[np.linspace(0, 1.0, 1201), np.linspace(1.0, 0.5, 150)[1:], np.full(250, 0.5)]
    logp = 4 + trend + 0.003 * rng.normal(size=1600)
    logp[1200] = logp[:1350].max() + 0.01          # make the top unambiguous
    return pd.Series(np.exp(logp), index=idx)


def test_major_top_found_once(bubble):
    tops = L.major_tops(bubble)
    assert 1200 in tops
    assert len(tops) <= 2


def test_major_top_label_window(bubble):
    d = bubble.index
    y = L.compute_labels(bubble, d[[1000, 1100, 1199, 1200, 1300]], "major_top_126")
    # top at 1200 is within (t, t + 126] for t = 1100 and 1199 only
    assert y.tolist() == [0.0, 1.0, 1.0, 0.0, 0.0]


def test_labels_undecidable_at_the_end(bubble):
    d = bubble.index
    assert np.isnan(L.compute_labels(bubble, d[[1500]], "major_top_126"))[0]
    assert np.isnan(L.compute_labels(bubble, d[[1550]], "rally_end_126"))[0]
    assert L.compute_labels(bubble, d[[1150]], "rally_end_126")[0] == 1.0
    assert L.compute_labels(bubble, d[[600]], "rally_end_126")[0] == 0.0


def test_drawdown_labels_match_the_original(bubble):
    from lppl.tuning.detection import drawdown_labels

    dates = bubble.index[800:1400:10]
    a = L.compute_labels(bubble, dates, "drawdown_63")
    b = drawdown_labels(bubble, dates, 63, 1.5, 750)["label"].to_numpy()
    np.testing.assert_array_equal(a, b)


def test_no_tops_gives_zero_labels():
    idx = pd.bdate_range("2010-01-01", periods=900)
    steady = pd.Series(np.exp(np.linspace(0, 1, 900)), index=idx)   # never falls
    y = L.compute_labels(steady, idx[[100, 500, 800]], "major_top_126")
    assert y[0] == 0 and y[1] == 0 and np.isnan(y[2])


def test_unknown_label():
    with pytest.raises(ValueError):
        L.compute_labels(pd.Series([1.0]), [], "nope")


def test_holm():
    np.testing.assert_allclose(L.holm([0.01, 0.04, 0.03, 0.5]), [0.04, 0.09, 0.09, 0.5])
    assert (L.holm([0.2, 0.9]) <= 1).all()


def test_tc_window_test_rewards_correct_windows(bubble):
    """Episodes whose tc window brackets their own top beat permuted windows."""
    rows, prices = [], {}
    for k in range(8):
        t = f"T{k}"
        s = bubble.copy()
        s.index = s.index + pd.Timedelta(days=0)
        prices[t] = s
        alarm = s.index[1200 - 40 - 10 * k]   # alarms at different leads to the top
        top = s.index[1200]
        rows.append({"ticker": t, "date": alarm, "conf": 0.5, "tc_q25": top - pd.Timedelta(days=7),
                     "tc_median": top, "tc_q75": top + pd.Timedelta(days=7)})
    ev = pd.DataFrame(rows)
    res, ep = V.tc_window_test(ev, pd.DataFrame(prices), V.ValidationSettings())
    assert res["episodes"] == 8 and res["hit_rate"] == 1.0
    assert res["placebo_mean"] < 1.0 and res["p_value"] < 0.2
