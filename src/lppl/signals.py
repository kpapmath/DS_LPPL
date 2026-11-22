"""Refined LPPLS signals and market features, one row per (ticker, analysis date).

Everything here is computed from fits already stored by a scan plus the price history; nothing
is refitted. See docs/combined_model.md for the definitions and the literature.

LPPLS signals (all use the qualification filters stored with the scan)

=================== ==========================================================================
conf                long-scale confidence: share of the windows with a qualified, positive fit
conf_short          the same on the short-scale scan (windows 40..180 days), when given
conf_agree          sqrt(conf * conf_short): high only when both scales agree (DS LPPLS)
conf_cluster        share of windows with a qualified positive fit whose tc lies close to the
                    median tc of those fits: |tc - median| <= rel * (median - t2) + abs_days
conf_resid          share of windows with a qualified positive fit whose residuals are
                    stationary (ADF t-statistic below ``adf_crit``; Lin et al. 2014)
lagr_pos            1 if the window chosen by Lagrange regularisation (Demos & Sornette 2017)
                    holds a qualified positive fit
tc_proximity        exp(-(median tc - t2) / 63 days) when conf > 0, else 0
=================== ==========================================================================

Market features (no LPPLS): momentum over 21, 126 and 252 trading days in sigma units, the
ratio of 63-day to 750-day volatility, and the current drawdown from the 252-day high in sigma
units.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
from pydantic import BaseModel

from lppl.model.lppls import dates_to_years, lppls

LPPLS_SIGNALS = ["conf", "conf_short", "conf_agree", "conf_cluster", "conf_resid", "lagr_pos",
                 "tc_proximity"]
MARKET_FEATURES = ["mom_21", "mom_126", "mom_252", "vol_ratio", "drawdown_252"]


class SignalSettings(BaseModel):
    cluster_rel: float = 0.25     # tc agreement: relative tolerance (share of the time to tc)
    cluster_abs_days: float = 5.0  # ... plus an absolute tolerance in calendar days
    adf_crit: float = -2.86       # ADF 5 % critical value with a constant (MacKinnon)
    n_params: int = 7             # LPPLS parameters, for the Lagrange normalisation
    tc_scale_days: float = 63.0   # tc_proximity decay
    vol_window: int = 750


# --------------------------------------------------------------------------- residual test
def adf_tstat(e: np.ndarray) -> float:
    """t-statistic of rho in  de_t = a + rho * e_{t-1} + u_t  (Dickey-Fuller with constant, no lags)."""
    e = np.asarray(e, dtype=float)
    if len(e) < 10:
        return np.nan
    x, dy = e[:-1], np.diff(e)
    x = x - x.mean()
    sxx = float(x @ x)
    if sxx <= 0:
        return np.nan
    rho = float(x @ (dy - dy.mean())) / sxx
    resid = dy - dy.mean() - rho * x
    s2 = float(resid @ resid) / (len(dy) - 2)
    return rho / np.sqrt(s2 / sxx) if s2 > 0 else -np.inf


def residual_tstats(fits: pd.DataFrame, prices: pd.Series) -> np.ndarray:
    """ADF t-statistic of the LPPLS residuals of each fit (NaN where it cannot be computed)."""
    px = prices.dropna().sort_index()
    px = px[px > 0]
    idx = pd.DatetimeIndex(px.index)
    t_all = dates_to_years(idx)
    y_all = np.log(px.to_numpy(dtype=float))
    end = idx.get_indexer(pd.DatetimeIndex(fits["t2"]))
    out = np.full(len(fits), np.nan)
    cols = [fits[c].to_numpy(float) for c in ("tc", "m", "omega", "a", "b", "c1", "c2")]
    L = fits["window_len"].to_numpy(int)
    for i in range(len(fits)):
        e = end[i]
        if e < 0 or e + 1 - L[i] < 0:
            continue
        sl = slice(e + 1 - L[i], e + 1)
        model = lppls(t_all[sl], *(c[i] for c in cols))
        out[i] = adf_tstat(y_all[sl] - model)
    return out


# --------------------------------------------------------------------------- LPPLS signals
def lppls_signals(fits: pd.DataFrame, prices: pd.DataFrame, s: SignalSettings,
                  short: pd.DataFrame | None = None) -> pd.DataFrame:
    """One row per (ticker, t2) of ``fits`` with the LPPLS signals of the module docstring.

    ``fits``: lppl_fits rows of one run (``qualified`` as stored). ``short``: indicator rows
    (ticker, date, pos_conf) of the short-scale scan, or None."""
    f = fits.copy()
    f["t2"] = pd.to_datetime(f["t2"])
    f["pos"] = f["qualified"].astype(bool) & (f["sign"] > 0)
    f["t2y"] = dates_to_years(f["t2"])
    # residual test only where it matters (qualified positive fits)
    f["adf"] = np.nan
    for ticker, g in f[f["pos"]].groupby("ticker"):
        f.loc[g.index, "adf"] = residual_tstats(g, prices[ticker])
    rows = []
    for (ticker, t2), g in f.groupby(["ticker", "t2"], sort=True):
        n = len(g)
        pos = g[g["pos"]]
        r = {"ticker": ticker, "date": t2, "n_windows": n, "conf": len(pos) / n}
        if len(pos):
            med = float(np.median(pos["tc"]))
            tol = s.cluster_rel * (med - pos["t2y"].iloc[0]) + s.cluster_abs_days / 365.25
            r["conf_cluster"] = float((np.abs(pos["tc"] - med) <= tol).sum() / n)
            r["conf_resid"] = float((pos["adf"] < s.adf_crit).sum() / n)
            days = (med - pos["t2y"].iloc[0]) * 365.25
            r["tc_proximity"] = float(np.exp(-max(days, 0.0) / s.tc_scale_days))
        else:
            r.update(conf_cluster=0.0, conf_resid=0.0, tc_proximity=0.0)
        # Lagrange regularisation: chi2(N) = SSE / (N - p); remove its linear trend in N
        N = g["window_len"].to_numpy(float)
        chi2 = g["sse"].to_numpy(float) / np.maximum(N - s.n_params, 1.0)
        if n >= 3 and np.ptp(N) > 0:
            lam = np.polyfit(N, chi2, 1)[0]
            best = int(np.argmin(chi2 - lam * N))
        else:
            best = int(np.argmin(chi2))
        r["lagr_pos"] = float(g["pos"].iloc[best])
        r["lagr_window"] = int(N[best])
        rows.append(r)
    out = pd.DataFrame(rows)
    if short is not None and len(short):
        sh = short[["ticker", "date", "pos_conf"]].rename(columns={"pos_conf": "conf_short"})
        sh = sh.assign(date=pd.to_datetime(sh["date"]))
        out = out.merge(sh, on=["ticker", "date"], how="left")
        out["conf_agree"] = np.sqrt(out["conf"] * out["conf_short"])
    else:
        out["conf_short"] = np.nan
        out["conf_agree"] = np.nan
    return out


# --------------------------------------------------------------------------- market features
def market_features(prices: pd.Series, dates, vol_window: int = 750) -> pd.DataFrame:
    """Momentum, volatility ratio and current drawdown at ``dates`` (no look-ahead)."""
    px = prices.dropna().sort_index()
    px = px[px > 0]
    logp = pd.Series(np.log(px.to_numpy(dtype=float)), index=px.index)
    ret = logp.diff()
    sig = ret.rolling(vol_window, min_periods=250).std()
    sig63 = ret.rolling(63, min_periods=40).std()
    out = pd.DataFrame(index=pd.DatetimeIndex(dates))
    at = lambda s: s.reindex(out.index).to_numpy()
    for h in (21, 126, 252):
        out[f"mom_{h}"] = at((logp - logp.shift(h)) / (sig * np.sqrt(h)))
    out["vol_ratio"] = at(sig63 / sig)
    out["drawdown_252"] = at((logp - logp.rolling(252, min_periods=60).max()) / (sig * np.sqrt(63)))
    return out


def build_features(fits: pd.DataFrame, prices: pd.DataFrame, s: SignalSettings,
                   short: pd.DataFrame | None = None, labels: list[str] | None = None,
                   k_sigma: float = 1.5, top_lookback: int = 252) -> pd.DataFrame:
    """LPPLS signals + market features + the requested labels (``lppl.labels``)."""
    from lppl.labels import compute_labels

    sig = lppls_signals(fits, prices, s, short)
    parts = []
    for ticker, g in sig.groupby("ticker", sort=True):
        g = g.reset_index(drop=True)
        mf = market_features(prices[ticker], g["date"], s.vol_window).reset_index(drop=True)
        g = pd.concat([g, mf], axis=1)
        for name in labels or []:
            g[name] = compute_labels(prices[ticker], g["date"], name, k_sigma, s.vol_window, top_lookback)
        parts.append(g)
    return pd.concat(parts, ignore_index=True)
