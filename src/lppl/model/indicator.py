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
class RobustConfig:
    """Robust variant of the confidence (docs/validation.md §8). The defaults switch it off.

    * ``seeds``: independent GA fits per window (seed 0 is the ordinary scan's fit).
    * ``min_agree``: a window counts only if at least this many seeds are qualified with the
      same sign, and their ``tc`` lie within ``tc_agree * (median tc - t2) + tc_agree_days``
      of their median.
    * ``persistence``: the stored confidence is the minimum of the raw confidence over the last
      ``persistence`` analysis dates (the signal must stay on; causal).
    """

    seeds: int = 1
    min_agree: int = 1
    tc_agree: float = 0.25
    tc_agree_days: float = 5.0
    persistence: int = 1

    def __post_init__(self):
        if self.seeds < 1 or not 1 <= self.min_agree <= self.seeds or self.persistence < 1:
            raise ValueError("invalid robust configuration")

    @property
    def enabled(self) -> bool:
        return self.seeds > 1 or self.persistence > 1


@dataclass(frozen=True)
class ScanSettings:
    windows: WindowConfig = field(default_factory=WindowConfig)
    bounds: SearchBounds = field(default_factory=SearchBounds)
    filters: FilterConfig = field(default_factory=FilterConfig)
    ga: GAConfig = field(default_factory=GAConfig)
    method: str = "ga"
    robust: RobustConfig = field(default_factory=RobustConfig)


def window_rng(master_seed: int, ticker: str, t2: pd.Timestamp, length: int,
               replica: int = 0) -> np.random.Generator:
    """Deterministic, order-independent RNG for one (ticker, t2, window) task. Replica 0 is
    the ordinary scan; replicas > 0 are the extra seeds of the robust variant."""
    key = [int(master_seed), zlib.crc32(ticker.encode()), int(pd.Timestamp(t2).toordinal()), int(length)]
    if replica:
        key.append(int(replica))
    return np.random.default_rng(np.random.SeedSequence(key))


def robust_vote(fits: list, oks: list[bool], rc: RobustConfig) -> tuple[int | None, bool, int]:
    """Combine the seed fits of one window. Returns (index of the fit to store, robust
    qualification, number of seeds in the agreeing majority).

    The agreeing set is the largest group of qualified seeds of one sign whose ``tc`` lie
    within the tolerance of their median. The stored fit is its lowest-SSE member (seed 0 if
    no seed qualifies)."""
    best: list[int] = []
    for sign in (1, -1):
        idx = [i for i, (f, ok) in enumerate(zip(fits, oks, strict=True)) if ok and f.sign == sign]
        if not idx:
            continue
        tcs = np.array([fits[i].tc for i in idx])
        med = float(np.median(tcs))
        tol = rc.tc_agree * max(med - fits[idx[0]].t2, 0.0) + rc.tc_agree_days / 365.25
        agree = [i for i, tc in zip(idx, tcs, strict=True) if abs(tc - med) <= tol]
        if len(agree) > len(best):
            best = agree
    if not best:
        return 0, False, 0
    pick = min(best, key=lambda i: fits[i].sse)
    return pick, len(best) >= rc.min_agree, len(best)


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
        rc = settings.robust
        runs = [fit_window(t, y, bounds=settings.bounds, method=settings.method, ga_config=settings.ga,
                           rng=window_rng(master_seed, ticker, t2_date, n, k)) for k in range(rc.seeds)]
        oks = [bool(check(r.fit, settings.filters)["qualified"]) for r in runs]
        if rc.seeds > 1:
            pick, qualified, n_ok = robust_vote([r.fit for r in runs], oks, rc)
        else:
            pick, qualified, n_ok = 0, oks[0], None
        res = runs[pick]
        fit = res.fit
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
            "qualified": bool(qualified),
            "sign": fit.sign,
            "generations": res.generations,
            "n_evals": int(sum(r.n_evals for r in runs)),
            "elapsed_ms": float(sum(r.elapsed_ms for r in runs)),
            **({"n_seeds_ok": n_ok} if n_ok is not None else {}),
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


def apply_persistence(ind: pd.DataFrame, periods: int) -> pd.DataFrame:
    """Replace pos_conf / neg_conf by their minimum over the last ``periods`` analysis dates of
    each ticker (the current one included). Causal: uses no later date."""
    if periods <= 1 or ind.empty:
        return ind
    out = ind.sort_values(["ticker", "date"]).copy()
    for c in ("pos_conf", "neg_conf"):
        out[c] = out.groupby("ticker")[c].transform(lambda s: s.rolling(periods, min_periods=1).min())
    return out.reset_index(drop=True)


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
    return fits, apply_persistence(aggregate(fits), settings.robust.persistence)
