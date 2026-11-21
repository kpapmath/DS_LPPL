import numpy as np
import pandas as pd
import pytest

from lppl.data import yahoo
from lppl.data.db import PRICE_COLUMNS


def raw_history(n=10, tz="America/New_York") -> pd.DataFrame:
    """Shape returned by yfinance Ticker.history(auto_adjust=False)."""
    idx = pd.date_range("2024-01-02", periods=n, freq="B", tz=tz, name="Date")
    close = np.linspace(100, 110, n)
    return pd.DataFrame({"Open": close - 1, "High": close + 1, "Low": close - 2, "Close": close,
                         "Adj Close": close * 0.98, "Volume": np.arange(n) * 1000}, index=idx)


def test_normalize_flat_frame():
    df = yahoo.normalize(raw_history())
    assert list(df.columns) == PRICE_COLUMNS
    assert df["date"].dt.tz is None
    assert (df["date"] == df["date"].dt.normalize()).all()
    assert df["adj_close"].iloc[0] == pytest.approx(98.0)


def test_normalize_multiindex_from_download():
    raw = raw_history()
    raw.columns = pd.MultiIndex.from_product([raw.columns, ["AAPL"]], names=["Price", "Ticker"])
    df = yahoo.normalize(raw)
    assert list(df.columns) == PRICE_COLUMNS and len(df) == 10


def test_normalize_without_adj_close_and_with_nans():
    raw = raw_history().drop(columns="Adj Close")
    raw.iloc[3, raw.columns.get_loc("Close")] = np.nan
    df = yahoo.normalize(raw)
    assert len(df) == 9
    assert (df["adj_close"] == df["close"]).all()


def test_validate_ok_and_warnings():
    df = yahoo.normalize(raw_history(30))
    rep = yahoo.validate(df)
    assert rep.n_rows == 30 and not rep.issues
    df2 = df.drop(index=range(5, 15)).reset_index(drop=True)
    assert any("gaps" in i for i in yahoo.validate(df2).issues)


@pytest.mark.parametrize("mutate, msg", [
    (lambda d: pd.concat([d, d.iloc[[0]]]).sort_values("date"), "duplicate"),
    (lambda d: d.assign(close=d["close"].where(d.index != 2, -1.0)), "non-positive"),
    (lambda d: d.iloc[::-1], "sorted"),
    (lambda d: d.iloc[0:0], "no rows"),
])
def test_validate_fatal(mutate, msg):
    df = mutate(yahoo.normalize(raw_history()))
    with pytest.raises(yahoo.DataValidationError, match=msg):
        yahoo.validate(df)


def test_snapshot_round_trip_and_hash_stable(tmp_path):
    df = yahoo.normalize(raw_history())
    path, h1 = yahoo.save_snapshot(df, tmp_path, "^GSPC")
    assert path.name == "IDX_GSPC.csv"
    back, h2 = yahoo.load_snapshot(tmp_path, "^GSPC")
    assert h1 == h2
    pd.testing.assert_frame_equal(back, df, check_dtype=False)
    _, h3 = yahoo.save_snapshot(back, tmp_path, "^GSPC")
    assert h3 == h1


def test_download_retries_then_succeeds(monkeypatch):
    calls = {"n": 0}

    class FakeTicker:
        def __init__(self, t):
            pass

        def history(self, **kw):
            calls["n"] += 1
            if calls["n"] < 3:
                raise ConnectionError("boom")
            assert kw["auto_adjust"] is False
            return raw_history()

    import yfinance
    monkeypatch.setattr(yfinance, "Ticker", FakeTicker)
    monkeypatch.setattr(yahoo.time, "sleep", lambda s: None)
    df = yahoo.download("AAPL", "2024-01-01", "2024-02-01")
    assert calls["n"] == 3 and len(df) == 10


def test_download_gives_up(monkeypatch):
    class Empty:
        def __init__(self, t):
            pass

        def history(self, **kw):
            return pd.DataFrame()

    import yfinance
    monkeypatch.setattr(yfinance, "Ticker", Empty)
    monkeypatch.setattr(yahoo.time, "sleep", lambda s: None)
    with pytest.raises(RuntimeError, match="failed to download"):
        yahoo.download("NOPE", "2024-01-01", retries=2)


@pytest.mark.network
def test_live_download_smoke():
    df = yahoo.download("^GSPC", "2024-01-01", "2024-01-31")
    assert 15 <= len(df) <= 23
    yahoo.validate(df)
