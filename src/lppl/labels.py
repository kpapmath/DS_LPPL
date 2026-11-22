"""Pre-registered event labels for validating the bubble signal (docs/validation.md §6).

Every label answers "did the event happen after analysis date t?" with 1 / 0, or NaN when the
data after t is too short to tell. Each one only looks forward for the label itself; volatility
thresholds use the trailing sigma at the time it applies (no look-ahead in the threshold).

=============== =====================================================================
drawdown_63     within 63 trading days, a fall from the running peak >= k sigma sqrt(63)
                (the original definition of the validation suite)
drawdown_126    the same within 126 trading days (six months)
major_top_126   a *major top* occurs within the next 126 trading days. A major top at p:
                the highest close of the previous ``top_lookback`` days and of the next
                126 days, followed within 126 days by a fall >= k sigma_p sqrt(126).
                This is what LPPLS claims: the end of a bubble, not just any drawdown.
rally_end_126   the price is lower 126 trading days later (the rally stops)
=============== =====================================================================

Momentum (the baseline score) is the trailing 126-day log return in sigma units.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from lppl.tuning.detection import drawdown_labels

LABELS = ("drawdown_63", "drawdown_126", "major_top_126", "rally_end_126")
HORIZON = 126


def _sigma(logp: np.ndarray, vol_window: int, min_obs: int = 250) -> np.ndarray:
    ret = np.diff(logp, prepend=np.nan)
    return pd.Series(ret).rolling(vol_window, min_periods=min_obs).std().to_numpy()


def major_tops(px: pd.Series, k_sigma: float = 1.5, lookback: int = 252, horizon: int = HORIZON,
               vol_window: int = 750) -> np.ndarray:
    """Positions (in ``px.dropna()``) of major tops; see the module docstring."""
    p = px.dropna().sort_index()
    logp = np.log(p.to_numpy(dtype=float))
    n = len(logp)
    s = pd.Series(logp)
    back_max = s.rolling(lookback + 1, min_periods=1).max().to_numpy()
    fwd_max = s[::-1].rolling(horizon + 1, min_periods=1).max().to_numpy()[::-1]
    # minimum over (p, p + horizon]: forward-rolling min, shifted by one day
    fwd_min = s[::-1].rolling(horizon, min_periods=horizon).min().to_numpy()[::-1]
    fwd_min = np.r_[fwd_min[1:], np.nan]
    sigma = _sigma(logp, vol_window)
    pos = np.arange(n)
    is_top = ((logp >= back_max) & (logp >= fwd_max) & (pos + horizon < n) & np.isfinite(sigma)
              & (fwd_min - logp <= -k_sigma * sigma * np.sqrt(horizon)))
    return np.flatnonzero(is_top)


def compute_labels(px: pd.Series, dates, name: str, k_sigma: float = 1.5, vol_window: int = 750,
                   top_lookback: int = 252) -> np.ndarray:
    """Label values of ``name`` for ``dates`` (a subset of the price index)."""
    if name not in LABELS:
        raise ValueError(f"unknown label {name!r}; choose from {LABELS}")
    if name.startswith("drawdown_"):
        h = int(name.split("_")[1])
        return drawdown_labels(px, dates, h, k_sigma, vol_window)["label"].to_numpy(float)
    p = px.dropna().sort_index()
    pos = pd.DatetimeIndex(p.index).get_indexer(pd.DatetimeIndex(dates))
    if (pos < 0).any():
        raise ValueError("some dates are not in the price index")
    n = len(p)
    logp = np.log(p.to_numpy(dtype=float))
    out = np.full(len(pos), np.nan)
    if name == "rally_end_126":
        ok = pos + HORIZON < n
        out[ok] = (logp[pos[ok] + HORIZON] < logp[pos[ok]]).astype(float)
        return out
    # major_top_126: needs t + 126 (top) + 126 (fall) of data to be decidable
    tops = major_tops(p, k_sigma, top_lookback, HORIZON, vol_window)
    ok = pos + 2 * HORIZON < n
    if len(tops) == 0:
        out[ok] = 0.0
        return out
    nxt = np.searchsorted(tops, pos + 1)                 # first top strictly after t
    has = (nxt < len(tops)) & (tops[np.minimum(nxt, len(tops) - 1)] <= pos + HORIZON)
    out[ok] = has[ok].astype(float)
    return out


def top_dates(px: pd.Series, **kw) -> pd.DatetimeIndex:
    p = px.dropna().sort_index()
    return pd.DatetimeIndex(p.index[major_tops(p, **kw)])


def holm(pvalues) -> np.ndarray:
    """Holm-Bonferroni adjusted p-values (same order as the input)."""
    p = np.asarray(pvalues, dtype=float)
    order = np.argsort(p)
    m = len(p)
    adj = np.empty(m)
    running = 0.0
    for rank, i in enumerate(order):
        running = max(running, min(1.0, (m - rank) * p[i]))
        adj[i] = running
    return adj
