"""Multi-window scanning and the LPPLS confidence indicator.

For an analysis date t2 we fit LPPLS on every window [t2 - L, t2] for L in a
grid of lengths (in trading days). The *positive* confidence is the fraction of
windows whose fit is qualified with B < 0 (accelerating rise), the *negative*
confidence the fraction qualified with B > 0 (accelerating fall).
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from lppl.model.calibrate import SearchBounds, fit_window
from lppl.model.filters import FilterConfig, check
from lppl.model.lppls import dates_to_years, years_to_date
from lppl.optim.ga import GAConfig


@dataclass(frozen=True)
class WindowConfig:
    min_len: int = 125   # trading days (~6 months)
    max_len: int = 750   # trading days (~3 years)
    step: int = 25

    def lengths(self, available: int | None = None) -> list[int]:
        """Window lengths (number of observations), largest first, capped by ``available``."""
        if self.min_len < 10 or self.max_len < self.min_len or self.step < 1:
            raise ValueError("invalid window configuration")
        lens = list(range(self.max_len, self.min_len - 1, -self.step))
        if available is not None:
            lens = [n for n in lens if n <= available]
        return lens


@dataclass(frozen=True)
class ScanSettings:
    windows: WindowConfig = field(default_factory=WindowConfig)
    bounds: SearchBounds = field(default_factory=SearchBounds)
    filters: FilterConfig = field(default_factory=FilterConfig)
    ga: GAConfig = field(default_factory=GAConfig)
    method: str = "ga"


def window_rng(master_seed: int, ticker: str, t2: pd.Timestamp, length: int) -> np.random.Generator:
    """Deterministic, order-independent RNG for one (ticker, t2, window) task."""
    key = [int(master_seed), zlib.crc32(ticker.encode()), int(pd.Timestamp(t2).toordinal()), int(length)]
    return np.random.default_rng(np.random.SeedSequence(key))


def scan_date(
    dates: pd.DatetimeIndex,
    log_price: np.ndarray,
    end_pos: int,
    settings: ScanSettings,
    master_seed: int,
    ticker: str,
) -> list[dict]:
    """Fit all windows ending at position ``end_pos`` (inclusive). Returns one record per window."""
    t_all = dates_to_years(dates)
    t2_date = dates[end_pos]
    records = []
    for n in settings.windows.lengths(available=end_pos + 1):
        sl = slice(end_pos + 1 - n, end_pos + 1)
        t, y = t_all[sl], log_price[sl]
        res = fit_window(t, y, bounds=settings.bounds, method=settings.method,
                         ga_config=settings.ga, rng=window_rng(master_seed, ticker, t2_date, n))
        fit = res.fit
        conds = check(fit, settings.filters)
        records.append({
            "ticker": ticker,
            "t1": dates[sl.start].date(),
            "t2": t2_date.date(),
            "window_len": n,
            "tc": fit.tc,
            "tc_date": years_to_date(fit.tc).date(),
            "m": fit.m,
            "omega": fit.omega,
            "a": fit.A, "b": fit.B, "c1": fit.C1, "c2": fit.C2,
            "sse": fit.sse,
            "rmse": fit.rmse,
            "damping": fit.damping,
            "n_osc": fit.n_oscillations,
            "qualified": bool(conds["qualified"]),
            "sign": fit.sign,
            "generations": res.generations,
            "n_evals": res.n_evals,
            "elapsed_ms": res.elapsed_ms,
        })
    return records


def requalify(fits: pd.DataFrame, filters: FilterConfig) -> pd.DataFrame:
    """Recompute ``qualified`` for stored fits under (possibly new) filters.

    The GA result of a window does not depend on the filters, so this gives exactly what
    a fresh scan with ``filters`` would produce."""
    from lppl.model.filters import qualify_arrays
    from lppl.model.lppls import dates_to_years

    out = fits.copy()
    out["qualified"] = qualify_arrays(
        t1=dates_to_years(pd.to_datetime(out["t1"])), t2=dates_to_years(pd.to_datetime(out["t2"])),
        tc=out["tc"], m=out["m"], omega=out["omega"], B=out["b"], C1=out["c1"], C2=out["c2"],
        sse=out["sse"].astype(float), n_obs=out["window_len"], cfg=filters)
    return out


def aggregate(fits: pd.DataFrame) -> pd.DataFrame:
    """Turn per-window fits into one indicator row per (ticker, t2)."""
    cols = ["ticker", "date", "pos_conf", "neg_conf", "n_windows", "n_qualified",
            "tc_median_date", "tc_q25_date", "tc_q75_date"]
    if fits.empty:
        return pd.DataFrame(columns=cols)

    def one(g: pd.DataFrame) -> pd.Series:
        q = g[g["qualified"]]
        n = len(g)
        out = {
            "pos_conf": float(((g["qualified"]) & (g["sign"] > 0)).sum() / n),
            "neg_conf": float(((g["qualified"]) & (g["sign"] < 0)).sum() / n),
            "n_windows": n,
            "n_qualified": len(q),
        }
        if len(q):
            qs = np.quantile(q["tc"].to_numpy(), [0.5, 0.25, 0.75])
            out.update(tc_median_date=years_to_date(qs[0]).date(),
                       tc_q25_date=years_to_date(qs[1]).date(),
                       tc_q75_date=years_to_date(qs[2]).date())
        else:
            out.update(tc_median_date=None, tc_q25_date=None, tc_q75_date=None)
        return pd.Series(out)

    rows = [{"ticker": k[0], "date": k[1], **one(g).to_dict()}
            for k, g in fits.groupby(["ticker", "t2"], sort=True)]
    return pd.DataFrame(rows, columns=cols)


def scan_dates_positions(dates: pd.DatetimeIndex, start, end, step: int, min_history: int) -> list[int]:
    """Positions of analysis dates: every ``step`` trading days in [start, end], counted
    back from the last eligible date so the most recent day is always included."""
    dates = pd.DatetimeIndex(dates)
    mask = np.ones(len(dates), dtype=bool)
    if start is not None:
        mask &= dates >= pd.Timestamp(start)
    if end is not None:
        mask &= dates <= pd.Timestamp(end)
    mask &= np.arange(len(dates)) >= min_history - 1
    pos = np.flatnonzero(mask)
    if pos.size == 0:
        return []
    return sorted(pos[::-1][::step].tolist())


def scan_series(
    prices: pd.Series,
    ticker: str,
    settings: ScanSettings,
    *,
    start=None,
    end=None,
    step: int = 10,
    master_seed: int = 42,
    n_jobs: int = -1,
    progress: bool = False,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Scan a price series. Returns (per-window fits, aggregated indicator)."""
    prices = prices.dropna().sort_index()
    prices = prices[prices > 0]
    dates = pd.DatetimeIndex(prices.index)
    logp = np.log(prices.to_numpy(dtype=float))
    positions = scan_dates_positions(dates, start, end, step, settings.windows.min_len)
    if not positions:
        empty = pd.DataFrame()
        return empty, aggregate(empty)
    jobs = (delayed(scan_date)(dates, logp, p, settings, master_seed, ticker) for p in positions)
    results = Parallel(n_jobs=n_jobs, verbose=5 if progress else 0)(jobs)
    fits = pd.DataFrame([r for batch in results for r in batch])
    return fits, aggregate(fits)
