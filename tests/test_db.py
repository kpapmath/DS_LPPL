"""Integration tests: need PostgreSQL (``make up``). Run with ``pytest -m integration``."""

import numpy as np
import pandas as pd
import pytest

from lppl.config import Instrument
from lppl.data import db

pytestmark = pytest.mark.integration


def price_frame(n=5, start="2024-01-02", base=100.0):
    dates = pd.bdate_range(start, periods=n)
    c = base + np.arange(n, dtype=float)
    return pd.DataFrame({"date": dates, "open": c, "high": c + 1, "low": c - 1, "close": c,
                         "adj_close": c, "volume": np.arange(n, dtype="int64")})


@pytest.fixture
def engine(pg_engine):
    with pg_engine.begin() as conn:
        for t in reversed(db.metadata.sorted_tables):
            conn.execute(t.delete())
    db.upsert_instruments(pg_engine, [Instrument(ticker="TEST", name="Test", group="extra")])
    return pg_engine


def test_price_upsert_is_idempotent(engine):
    db.upsert_prices(engine, "TEST", price_frame())
    db.upsert_prices(engine, "TEST", price_frame())
    assert len(db.load_prices(engine, "TEST")) == 5


def test_price_upsert_updates_revised_values(engine):
    db.upsert_prices(engine, "TEST", price_frame())
    db.upsert_prices(engine, "TEST", price_frame(base=200.0))
    px = db.load_prices(engine, "TEST")
    assert px["adj_close"].iloc[0] == 200.0


def test_load_prices_date_filter(engine):
    db.upsert_prices(engine, "TEST", price_frame(10))
    px = db.load_prices(engine, "TEST", start="2024-01-04", end="2024-01-09")
    assert px.index.min() == pd.Timestamp("2024-01-04")
    assert px.index.max() == pd.Timestamp("2024-01-09")


def test_run_fits_and_indicator_round_trip(engine):
    run = db.create_run(engine, {"k": 1}, seed=1, label="t")
    fits = pd.DataFrame([{
        "ticker": "TEST", "t1": pd.Timestamp("2023-01-02").date(),
        "t2": pd.Timestamp("2024-01-02").date(), "window_len": 250, "tc": 2024.1,
        "tc_date": pd.Timestamp("2024-02-07").date(), "m": 0.5, "omega": 8.0, "a": 1.0,
        "b": -0.5, "c1": 0.01, "c2": 0.02, "sse": 0.1, "rmse": 0.02, "damping": 1.0,
        "n_osc": float("nan"), "qualified": True, "sign": 1, "generations": 30,
        "n_evals": 1200, "elapsed_ms": 50.0,
    }])
    assert db.write_fits(engine, run, fits) == 1
    db.write_fits(engine, run, fits)  # idempotent (unique run/ticker/t2/window)
    ind = pd.DataFrame([{"ticker": "TEST", "date": pd.Timestamp("2024-01-02").date(),
                         "pos_conf": 1.0, "neg_conf": 0.0, "n_windows": 1, "n_qualified": 1,
                         "tc_median_date": None, "tc_q25_date": None, "tc_q75_date": None}])
    db.write_indicator(engine, run, ind)
    db.finish_run(engine, run)
    assert db.latest_run_id(engine) == run
    got = db.load_fits(engine, run)
    assert len(got) == 1 and pd.isna(got["n_osc"].iloc[0])
    gi = db.load_indicator(engine)
    assert gi["pos_conf"].tolist() == [1.0]


def test_load_indicator_prefers_latest_run(engine):
    def write(conf):
        run = db.create_run(engine, {}, seed=0)
        db.write_indicator(engine, run, pd.DataFrame([{
            "ticker": "TEST", "date": pd.Timestamp("2024-01-02").date(), "pos_conf": conf,
            "neg_conf": 0.0, "n_windows": 1, "n_qualified": 0}]))
        db.finish_run(engine, run)
    write(0.1)
    write(0.9)
    assert db.load_indicator(engine)["pos_conf"].tolist() == [0.9]


def test_ingest_from_snapshot(engine, tmp_path, monkeypatch):
    from lppl import pipeline
    from lppl.config import AppConfig, DataConfig
    from lppl.data import yahoo

    yahoo.save_snapshot(price_frame(20), tmp_path, "TEST")
    cfg = AppConfig(data=DataConfig(snapshot_dir=tmp_path, instruments=[
        Instrument(ticker="TEST", name="Test", group="extra")]))
    summary = pipeline.ingest(cfg, engine, from_snapshot=True)
    assert summary["rows"].tolist() == [20]
    assert len(db.load_prices(engine, "TEST")) == 20
