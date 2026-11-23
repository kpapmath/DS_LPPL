"""PostgreSQL schema and data-access helpers (SQLAlchemy Core).

The schema is defined once, here, and created with ``lppl init-db``
(``metadata.create_all`` is idempotent). See docs/data_dictionary.md.
"""

from __future__ import annotations

import math
import uuid
from collections.abc import Iterable
from datetime import UTC, datetime

import pandas as pd
from sqlalchemy import (
    BigInteger, Boolean, Column, Date, DateTime, Float, ForeignKey, Index, Integer, MetaData,
    SmallInteger, String, Table, Text, UniqueConstraint, create_engine, func, select, text,
)
from sqlalchemy.dialects.postgresql import JSONB, insert
from sqlalchemy.engine import Engine

metadata = MetaData()

instruments = Table(
    "instruments", metadata,
    Column("ticker", String(16), primary_key=True),
    Column("name", Text, nullable=False),
    Column("grp", String(16), nullable=False),
    Column("asset_type", String(16), nullable=False),
)

prices = Table(
    "prices", metadata,
    Column("ticker", String(16), ForeignKey("instruments.ticker"), primary_key=True),
    Column("date", Date, primary_key=True),
    Column("open", Float), Column("high", Float), Column("low", Float),
    Column("close", Float), Column("adj_close", Float), Column("volume", BigInteger),
)

ingestion_log = Table(
    "ingestion_log", metadata,
    Column("id", Integer, primary_key=True, autoincrement=True),
    Column("ticker", String(16), nullable=False),
    Column("fetched_at", DateTime(timezone=True), nullable=False),
    Column("source", String(32), nullable=False),  # "yahoo" | "snapshot"
    Column("n_rows", Integer, nullable=False),
    Column("first_date", Date), Column("last_date", Date),
    Column("sha256", String(64)),
    Column("yfinance_version", String(32)),
    Column("issues", Text),
)

fit_runs = Table(
    "fit_runs", metadata,
    Column("run_id", String(36), primary_key=True),
    Column("created_at", DateTime(timezone=True), nullable=False),
    Column("finished_at", DateTime(timezone=True)),
    Column("status", String(16), nullable=False),
    Column("label", Text),
    Column("seed", Integer, nullable=False),
    Column("git_sha", String(40)),
    Column("package_version", String(32)),
    Column("config", JSONB, nullable=False),
)

lppl_fits = Table(
    "lppl_fits", metadata,
    Column("id", BigInteger, primary_key=True, autoincrement=True),
    Column("run_id", String(36), ForeignKey("fit_runs.run_id", ondelete="CASCADE"), nullable=False),
    Column("ticker", String(16), nullable=False),
    Column("t1", Date, nullable=False), Column("t2", Date, nullable=False),
    Column("window_len", Integer, nullable=False),
    Column("tc", Float), Column("tc_date", Date),
    Column("m", Float), Column("omega", Float),
    Column("a", Float), Column("b", Float), Column("c1", Float), Column("c2", Float),
    Column("sse", Float), Column("rmse", Float),
    Column("damping", Float), Column("n_osc", Float),
    Column("qualified", Boolean, nullable=False),
    Column("sign", SmallInteger, nullable=False),
    Column("generations", Integer), Column("n_evals", Integer), Column("elapsed_ms", Float),
    Column("n_seeds_ok", SmallInteger),  # robust variant: seeds in the agreeing majority
    UniqueConstraint("run_id", "ticker", "t2", "window_len", name="uq_fit_window"),
)
Index("ix_fits_run_ticker_t2", lppl_fits.c.run_id, lppl_fits.c.ticker, lppl_fits.c.t2)

bubble_indicator = Table(
    "bubble_indicator", metadata,
    Column("run_id", String(36), ForeignKey("fit_runs.run_id", ondelete="CASCADE"), primary_key=True),
    Column("ticker", String(16), primary_key=True),
    Column("date", Date, primary_key=True),
    Column("pos_conf", Float, nullable=False),
    Column("neg_conf", Float, nullable=False),
    Column("n_windows", Integer, nullable=False),
    Column("n_qualified", Integer, nullable=False),
    Column("tc_median_date", Date), Column("tc_q25_date", Date), Column("tc_q75_date", Date),
)

PRICE_COLUMNS = ["date", "open", "high", "low", "close", "adj_close", "volume"]


# --------------------------------------------------------------------------- engine
def get_engine(url: str | None = None) -> Engine:
    if url is None:
        from lppl.config import DBSettings
        url = DBSettings().url
    return create_engine(url, pool_pre_ping=True, future=True)


def init_db(engine: Engine) -> None:
    metadata.create_all(engine, checkfirst=True)
    with engine.begin() as conn:  # columns added after the first release
        conn.execute(text("ALTER TABLE lppl_fits ADD COLUMN IF NOT EXISTS n_seeds_ok SMALLINT"))


def drop_all(engine: Engine) -> None:
    metadata.drop_all(engine)


def ping(engine: Engine) -> bool:
    with engine.connect() as conn:
        return conn.execute(text("select 1")).scalar() == 1


# --------------------------------------------------------------------------- helpers
def _clean(v):
    if v is None:
        return None
    if isinstance(v, float) and (math.isnan(v) or math.isinf(v)):
        return None
    if v is pd.NaT:
        return None
    if hasattr(v, "item"):  # numpy scalar
        return _clean(v.item())
    if isinstance(v, pd.Timestamp):
        return v.date()
    return v


def records(df: pd.DataFrame) -> list[dict]:
    return [{k: _clean(v) for k, v in row.items()} for row in df.to_dict(orient="records")]


def upsert(engine: Engine, table: Table, rows: Iterable[dict], chunk: int = 2000) -> int:
    """INSERT ... ON CONFLICT (pk / unique) DO UPDATE. Returns number of rows sent."""
    rows = list(rows)
    if not rows:
        return 0
    pk = [c.name for c in table.primary_key.columns]
    uniques = [c for c in table.constraints if isinstance(c, UniqueConstraint)]
    conflict_cols = [c.name for c in uniques[0].columns] if uniques else pk
    with engine.begin() as conn:
        for i in range(0, len(rows), chunk):
            batch = rows[i : i + chunk]
            stmt = insert(table).values(batch)
            update_cols = {c: stmt.excluded[c] for c in batch[0]
                           if c not in conflict_cols and c not in pk}
            stmt = (stmt.on_conflict_do_update(index_elements=conflict_cols, set_=update_cols)
                    if update_cols else stmt.on_conflict_do_nothing(index_elements=conflict_cols))
            conn.execute(stmt)
    return len(rows)


# --------------------------------------------------------------------------- prices
def upsert_instruments(engine: Engine, items: Iterable) -> int:
    return upsert(engine, instruments, [
        {"ticker": i.ticker, "name": i.name, "grp": i.group, "asset_type": i.asset_type}
        for i in items
    ])


def upsert_prices(engine: Engine, ticker: str, df: pd.DataFrame) -> int:
    df = df[PRICE_COLUMNS].copy()
    df.insert(0, "ticker", ticker)
    return upsert(engine, prices, records(df))


def load_prices(engine: Engine, ticker: str, start=None, end=None) -> pd.DataFrame:
    q = select(prices).where(prices.c.ticker == ticker).order_by(prices.c.date)
    if start is not None:
        q = q.where(prices.c.date >= pd.Timestamp(start).date())
    if end is not None:
        q = q.where(prices.c.date <= pd.Timestamp(end).date())
    with engine.connect() as conn:
        df = pd.read_sql(q, conn, parse_dates=["date"])
    return df.set_index("date")


def load_all_prices(engine: Engine, field: str = "adj_close") -> pd.DataFrame:
    """Wide frame: index = date, columns = tickers."""
    with engine.connect() as conn:
        df = pd.read_sql(select(prices.c.ticker, prices.c.date, prices.c[field]), conn,
                         parse_dates=["date"])
    return df.pivot(index="date", columns="ticker", values=field).sort_index()


def load_instruments(engine: Engine) -> pd.DataFrame:
    with engine.connect() as conn:
        return pd.read_sql(select(instruments), conn)


def log_ingestion(engine: Engine, **row) -> None:
    row.setdefault("fetched_at", datetime.now(UTC))
    with engine.begin() as conn:
        conn.execute(ingestion_log.insert().values(**{k: _clean(v) for k, v in row.items()}))


# --------------------------------------------------------------------------- runs
def create_run(engine: Engine, config: dict, seed: int, label: str | None = None,
               git_sha: str | None = None, package_version: str | None = None) -> str:
    run_id = str(uuid.uuid4())
    with engine.begin() as conn:
        conn.execute(fit_runs.insert().values(
            run_id=run_id, created_at=datetime.now(UTC), status="running", label=label,
            seed=seed, git_sha=git_sha, package_version=package_version, config=config,
        ))
    return run_id


def finish_run(engine: Engine, run_id: str, status: str = "finished") -> None:
    with engine.begin() as conn:
        conn.execute(fit_runs.update().where(fit_runs.c.run_id == run_id)
                     .values(status=status, finished_at=datetime.now(UTC)))


def write_fits(engine: Engine, run_id: str, fits: pd.DataFrame) -> int:
    if fits.empty:
        return 0
    df = fits.copy()
    df.insert(0, "run_id", run_id)
    return upsert(engine, lppl_fits, records(df))


def write_indicator(engine: Engine, run_id: str, ind: pd.DataFrame) -> int:
    if ind.empty:
        return 0
    df = ind.copy()
    df.insert(0, "run_id", run_id)
    return upsert(engine, bubble_indicator, records(df))


def list_runs(engine: Engine) -> pd.DataFrame:
    q = select(fit_runs.c.run_id, fit_runs.c.created_at, fit_runs.c.finished_at,
               fit_runs.c.status, fit_runs.c.label, fit_runs.c.seed, fit_runs.c.git_sha
               ).order_by(fit_runs.c.created_at.desc())
    with engine.connect() as conn:
        return pd.read_sql(q, conn)


def latest_run_id(engine: Engine, status: str = "finished") -> str | None:
    q = (select(fit_runs.c.run_id).where(fit_runs.c.status == status)
         .order_by(fit_runs.c.created_at.desc()).limit(1))
    with engine.connect() as conn:
        return conn.execute(q).scalar()


def load_indicator(engine: Engine, run_id: str | None = None, tickers: list[str] | None = None,
                   label: str | None = None) -> pd.DataFrame:
    """Indicator rows with their run ``label``.

    * ``run_id`` given: that run only.
    * otherwise all *finished* runs (optionally only ``label``; a trailing ``*`` matches a
      prefix, e.g. ``"case:*"``). Duplicated (label, ticker, date) keep the most recent run.
    """
    q = select(bubble_indicator, fit_runs.c.label, fit_runs.c.created_at).join(
        fit_runs, fit_runs.c.run_id == bubble_indicator.c.run_id)
    if run_id is not None:
        q = q.where(bubble_indicator.c.run_id == run_id)
    else:
        q = q.where(fit_runs.c.status == "finished")
    if label is not None:
        q = (q.where(fit_runs.c.label.like(label[:-1] + "%")) if label.endswith("*")
             else q.where(fit_runs.c.label == label))
    if tickers:
        q = q.where(bubble_indicator.c.ticker.in_(tickers))
    with engine.connect() as conn:
        df = pd.read_sql(q, conn, parse_dates=["date", "tc_median_date", "tc_q25_date",
                                               "tc_q75_date"])
    df["label"] = df["label"].fillna("")
    df = (df.sort_values("created_at").drop_duplicates(["label", "ticker", "date"], keep="last")
          .drop(columns="created_at").sort_values(["label", "ticker", "date"])
          .reset_index(drop=True))
    return df


def latest_run_for_label(engine: Engine, label: str) -> str | None:
    q = (select(fit_runs.c.run_id).where(fit_runs.c.status == "finished",
                                         fit_runs.c.label == label)
         .order_by(fit_runs.c.created_at.desc()).limit(1))
    with engine.connect() as conn:
        return conn.execute(q).scalar()


def delete_run(engine: Engine, run_id: str) -> None:
    """Delete a run and (via ON DELETE CASCADE) its fits and indicator rows."""
    with engine.begin() as conn:
        conn.execute(fit_runs.delete().where(fit_runs.c.run_id == run_id))


def load_fits(engine: Engine, run_id: str | None = None, ticker: str | None = None,
              t2=None) -> pd.DataFrame:
    q = select(lppl_fits)
    if run_id is not None:
        q = q.where(lppl_fits.c.run_id == run_id)
    if ticker is not None:
        q = q.where(lppl_fits.c.ticker == ticker)
    if t2 is not None:
        q = q.where(lppl_fits.c.t2 == t2)
    with engine.connect() as conn:
        return pd.read_sql(q.order_by(lppl_fits.c.t2, lppl_fits.c.window_len), conn,
                           parse_dates=["t1", "t2", "tc_date"])


def table_counts(engine: Engine) -> dict[str, int]:
    out = {}
    with engine.connect() as conn:
        for t in metadata.sorted_tables:
            out[t.name] = conn.execute(select(func.count()).select_from(t)).scalar()
    return out
