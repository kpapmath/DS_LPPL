"""Yahoo Finance download, normalisation, validation and CSV snapshots."""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd

from lppl.data.db import PRICE_COLUMNS

log = logging.getLogger(__name__)

_RENAME = {
    "Date": "date", "Datetime": "date", "Open": "open", "High": "high", "Low": "low",
    "Close": "close", "Adj Close": "adj_close", "Volume": "volume",
}


class DataValidationError(ValueError):
    pass


@dataclass
class ValidationReport:
    n_rows: int
    first_date: pd.Timestamp | None
    last_date: pd.Timestamp | None
    issues: list[str] = field(default_factory=list)


def normalize(raw: pd.DataFrame) -> pd.DataFrame:
    """Turn a yfinance frame (flat or MultiIndex columns, tz-aware index) into
    columns ``PRICE_COLUMNS`` with a tz-naive daily ``date`` column."""
    df = raw.copy()
    if isinstance(df.columns, pd.MultiIndex):
        # yf.download(...) returns (field, ticker); keep the field level
        level = 0 if "Close" in df.columns.get_level_values(0) else -1
        df.columns = df.columns.get_level_values(level)
    if "date" not in df.columns and "Date" not in df.columns:
        df = df.reset_index()
    df = df.rename(columns=_RENAME)
    df["date"] = pd.to_datetime(df["date"])
    if df["date"].dt.tz is not None:
        df["date"] = df["date"].dt.tz_localize(None)
    df["date"] = df["date"].dt.normalize()
    if "adj_close" not in df.columns:
        df["adj_close"] = df["close"]
    for c in PRICE_COLUMNS:
        if c not in df.columns:
            df[c] = np.nan
    df = df[PRICE_COLUMNS].dropna(subset=["close", "adj_close"])
    df["volume"] = df["volume"].fillna(0).astype("int64")
    return df.sort_values("date").reset_index(drop=True)


def validate(df: pd.DataFrame, max_gap_days: int = 7) -> ValidationReport:
    """Raise on fatal problems, return warnings otherwise."""
    if df.empty:
        raise DataValidationError("no rows")
    if df["date"].duplicated().any():
        raise DataValidationError("duplicate dates")
    if (df[["close", "adj_close"]] <= 0).any().any():
        raise DataValidationError("non-positive prices")
    if not df["date"].is_monotonic_increasing:
        raise DataValidationError("dates not sorted")
    issues = []
    gaps = df["date"].diff().dt.days
    big = gaps[gaps > max_gap_days]
    if len(big):
        issues.append(f"{len(big)} gaps > {max_gap_days} calendar days (max {int(big.max())})")
    stale = (df["adj_close"].diff() == 0).astype(int)
    run = stale.groupby((stale == 0).cumsum()).cumsum().max()
    if run >= 5:
        issues.append(f"stale price run of {int(run)} days")
    hl = df[["high", "low"]].dropna()
    if (hl["high"] < hl["low"]).any():
        issues.append("high < low on some rows")
    return ValidationReport(len(df), df["date"].iloc[0], df["date"].iloc[-1], issues)


def download(ticker: str, start, end=None, retries: int = 4, backoff: float = 2.0) -> pd.DataFrame:
    """Download daily bars (unadjusted OHLC + Adj Close) with retry/backoff."""
    import yfinance as yf

    # yfinance treats `end` as exclusive; add one day so the configured end is included.
    end_excl = (pd.Timestamp(end) + pd.Timedelta(days=1)).date() if end is not None else None
    last_exc: Exception | None = None
    for attempt in range(retries):
        try:
            raw = yf.Ticker(ticker).history(start=str(start), end=end_excl and str(end_excl),
                                            interval="1d", auto_adjust=False, actions=False)
            if raw is None or raw.empty:
                raise RuntimeError(f"empty response for {ticker}")
            return normalize(raw)
        except Exception as exc:  # network / rate-limit / parsing
            last_exc = exc
            wait = backoff ** attempt
            log.warning("download %s failed (%s); retry in %.1fs", ticker, exc, wait)
            time.sleep(wait)
    raise RuntimeError(f"failed to download {ticker} after {retries} attempts") from last_exc


# --------------------------------------------------------------------------- snapshots
def snapshot_path(snapshot_dir: Path, ticker: str) -> Path:
    safe = ticker.replace("^", "IDX_").replace("/", "_")
    return Path(snapshot_dir) / f"{safe}.csv"


def to_csv_bytes(df: pd.DataFrame) -> bytes:
    out = df[PRICE_COLUMNS].copy()
    out["date"] = out["date"].dt.strftime("%Y-%m-%d")
    return out.to_csv(index=False, float_format="%.10g", lineterminator="\n").encode()


def save_snapshot(df: pd.DataFrame, snapshot_dir: Path, ticker: str) -> tuple[Path, str]:
    data = to_csv_bytes(df)
    path = snapshot_path(snapshot_dir, ticker)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(data)
    return path, hashlib.sha256(data).hexdigest()


def load_snapshot(snapshot_dir: Path, ticker: str) -> tuple[pd.DataFrame, str]:
    path = snapshot_path(snapshot_dir, ticker)
    data = path.read_bytes()
    df = pd.read_csv(path, parse_dates=["date"])
    df["volume"] = df["volume"].fillna(0).astype("int64")
    return df[PRICE_COLUMNS], hashlib.sha256(data).hexdigest()
