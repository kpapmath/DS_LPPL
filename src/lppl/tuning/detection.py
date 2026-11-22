"""Study 2: tune the detector (filters + window grid) against forward drawdowns.

Design choices (docs/tuning.md §3):

* **Events.** At analysis date t2 the label is 1 if, within the next ``horizon`` trading
  days, the price suffers a log-drawdown (from its running peak after t2) of at least
  ``k_sigma * sigma * sqrt(horizon)``. ``sigma`` is the trailing daily volatility up to t2,
  so the threshold adapts to each asset without using future information.
* **Score.** The positive confidence is used as a ranking score for upcoming events and
  evaluated with *average precision* (area under the precision-recall curve), which
  suits rare events. A naive momentum score is reported as a baseline.
* **Split.** Train on t2 < split_date - embargo and validate on t2 >= split_date. The
  embargo stops train labels from looking into the validation period (purging).
* **Speed.** Fits are filter-independent, so each trial only re-qualifies stored fits
  (vectorised) and aggregates with ``np.bincount``. A trial takes milliseconds.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd

from lppl.model.filters import FilterConfig, derived_arrays, qualify_arrays
from lppl.model.indicator import WindowConfig
from lppl.model.lppls import dates_to_years
from lppl.tuning.settings import IndicatorStudySettings, filter_config, window_config


# --------------------------------------------------------------------------- labels
def drawdown_labels(prices: pd.Series, dates, horizon: int, k_sigma: float,
                    vol_window: int = 750, min_vol_obs: int = 250) -> pd.DataFrame:
    """Forward-drawdown event labels for ``dates`` (subset of the price index).

    Returns columns ``fwd_dd`` (log drawdown, <= 0), ``threshold``, ``label`` (1/0, NaN
    when the horizon runs past the data or sigma is unavailable) and ``momentum``
    (trailing 126-day log return in sigma units: the naive baseline score).
    """
    px = prices.dropna().sort_index()
    logp = np.log(px.to_numpy(dtype=float))
    ret = np.diff(logp, prepend=np.nan)
    sigma = pd.Series(ret).rolling(vol_window, min_periods=min_vol_obs).std().to_numpy()
    idx = pd.DatetimeIndex(px.index)
    pos = idx.get_indexer(pd.DatetimeIndex(dates))
    if (pos < 0).any():
        raise ValueError("some dates are not in the price index")
    n = len(logp)
    rows = []
    for p in pos:
        if p + horizon >= n or not np.isfinite(sigma[p]):
            rows.append((np.nan, np.nan, np.nan, np.nan))
            continue
        seg = logp[p : p + horizon + 1]
        dd = float(np.min(seg - np.maximum.accumulate(seg)))
        thr = k_sigma * sigma[p] * np.sqrt(horizon)
        mom = (logp[p] - logp[p - 126]) / (sigma[p] * np.sqrt(126)) if p >= 126 else np.nan
        rows.append((dd, thr, float(dd <= -thr), mom))
    return pd.DataFrame(rows, columns=["fwd_dd", "threshold", "label", "momentum"],
                        index=pd.DatetimeIndex(dates))


# --------------------------------------------------------------------------- metrics
def average_precision(y, score) -> float:
    """Area under the precision-recall curve (step-wise, ties handled as one threshold);
    equals sklearn's ``average_precision_score``. NaN if there are no positives."""
    y = np.asarray(y, dtype=float)
    s = np.asarray(score, dtype=float)
    keep = np.isfinite(y) & np.isfinite(s)
    y, s = y[keep], s[keep]
    n_pos = y.sum()
    if n_pos == 0:
        return float("nan")
    order = np.argsort(-s, kind="mergesort")
    y, s = y[order], s[order]
    last_of_tie = np.r_[np.diff(s) != 0, True]
    tp = np.cumsum(y)[last_of_tie]
    k = np.flatnonzero(last_of_tie) + 1
    precision = tp / k
    recall = tp / n_pos
    return float(np.sum(np.diff(np.r_[0.0, recall]) * precision))


def alarm_stats(y, score, threshold: float) -> dict[str, float]:
    y = np.asarray(y, dtype=float)
    s = np.asarray(score, dtype=float)
    keep = np.isfinite(y) & np.isfinite(s)
    y, alarm = y[keep] == 1, s[keep] >= threshold
    tp, fp, fn = (y & alarm).sum(), (~y & alarm).sum(), (y & ~alarm).sum()
    prec = tp / (tp + fp) if tp + fp else float("nan")
    rec = tp / (tp + fn) if tp + fn else float("nan")
    return {"precision": float(prec), "recall": float(rec), "alarm_rate": float(alarm.mean())}


# --------------------------------------------------------------------------- dataset
@dataclass
class DetectionData:
    """Everything a trial needs, as flat numpy arrays (cheap to ship to Ray workers)."""

    # one entry per stored fit
    group: np.ndarray            # index into ``groups``
    window_len: np.ndarray
    t1: np.ndarray
    t2: np.ndarray
    tc: np.ndarray
    m: np.ndarray
    omega: np.ndarray
    damping: np.ndarray
    n_osc: np.ndarray
    rmse: np.ndarray
    sse: np.ndarray
    sign: np.ndarray
    # one row per (ticker, analysis date)
    groups: pd.DataFrame         # ticker, date, label, momentum, split
    max_len: int
    grid_step: int               # spacing of the stored window lengths
    meta: dict = field(default_factory=dict)

    @property
    def n_groups(self) -> int:
        return len(self.groups)


def build_detection_data(fits: pd.DataFrame, prices: pd.DataFrame,
                         settings: IndicatorStudySettings, source_run_id: str = "") -> DetectionData:
    """``fits``: rows of ``lppl_fits`` for one run; ``prices``: wide adj-close frame."""
    fits = fits.copy()
    fits["t2"] = pd.to_datetime(fits["t2"])
    fits["t1"] = pd.to_datetime(fits["t1"])
    groups = (fits[["ticker", "t2"]].drop_duplicates().sort_values(["ticker", "t2"])
              .rename(columns={"t2": "date"}).reset_index(drop=True))
    labels = []
    for ticker, g in groups.groupby("ticker", sort=False):
        lab = drawdown_labels(prices[ticker], g["date"], settings.horizon, settings.k_sigma,
                              settings.vol_window)
        labels.append(lab.reset_index(drop=True).set_index(g.index))
    groups = groups.join(pd.concat(labels))
    split = pd.Timestamp(settings.split_date)
    embargo_end = split - pd.offsets.BDay(settings.embargo_days)
    groups["split"] = np.where(groups["date"] >= split, "valid",
                               np.where(groups["date"] < embargo_end, "train", "embargo"))
    key = pd.MultiIndex.from_frame(groups[["ticker", "date"]])
    gidx = key.get_indexer(pd.MultiIndex.from_frame(fits[["ticker", "t2"]]))
    t1y, t2y = dates_to_years(fits["t1"]), dates_to_years(fits["t2"])
    d = derived_arrays(t1=t1y, t2=t2y, tc=fits["tc"], m=fits["m"], omega=fits["omega"],
                       B=fits["b"], C1=fits["c1"], C2=fits["c2"], sse=fits["sse"].astype(float),
                       n_obs=fits["window_len"])
    data = DetectionData(
        group=gidx, window_len=fits["window_len"].to_numpy(int), t1=t1y, t2=t2y,
        tc=fits["tc"].to_numpy(float), m=fits["m"].to_numpy(float),
        omega=fits["omega"].to_numpy(float), damping=d["damping"], n_osc=d["n_osc"],
        rmse=d["rmse"], sse=fits["sse"].to_numpy(float), sign=np.sign(-fits["b"].to_numpy(float)),
        groups=groups, max_len=int(fits["window_len"].max()), grid_step=_grid_step(fits["window_len"]))
    lab = groups["label"]
    data.meta = {
        "source_run_id": source_run_id, "n_fits": len(fits), "n_groups": len(groups),
        "n_train": int((groups.split == "train").sum()), "n_valid": int((groups.split == "valid").sum()),
        "base_rate_train": float(lab[groups.split == "train"].mean()),
        "base_rate_valid": float(lab[groups.split == "valid"].mean()),
        "baseline_ap_train": average_precision(lab[groups.split == "train"],
                                               groups.loc[groups.split == "train", "momentum"]),
        "baseline_ap_valid": average_precision(lab[groups.split == "valid"],
                                               groups.loc[groups.split == "valid", "momentum"]),
    }
    return data


def _grid_step(lengths) -> int:
    u = np.unique(np.asarray(lengths, dtype=int))
    return int(np.gcd.reduce(np.diff(u))) if u.size > 1 else 1


# --------------------------------------------------------------------------- scoring
def min_windows_used(data: DetectionData, n_win: np.ndarray) -> float:
    """Fewest windows a detector setting uses on a date where the full stored grid fits.

    Dates early in a series have fewer stored windows (only those that fit the history); they
    must not make every setting look invalid, so only full-grid dates are considered."""
    stored = np.bincount(data.group, minlength=data.n_groups)
    full = stored == stored.max()
    return float(n_win[full].min()) if full.any() else 0.0


def confidence(data: DetectionData, filters: FilterConfig, min_len: int, max_len: int,
               step: int) -> tuple[np.ndarray, np.ndarray]:
    """Positive confidence and number of windows per group for one detector setting."""
    L = data.window_len
    inwin = (L >= min_len) & (L <= max_len) & ((max_len - L) % step == 0)
    ok = qualify_arrays(t1=data.t1, t2=data.t2, tc=data.tc, m=data.m, omega=data.omega,
                        sse=data.sse, cfg=filters,
                        derived={"damping": data.damping, "n_osc": data.n_osc, "rmse": data.rmse})
    n_win = np.bincount(data.group, weights=inwin, minlength=data.n_groups)
    pos = np.bincount(data.group, weights=inwin & ok & (data.sign > 0), minlength=data.n_groups)
    conf = np.divide(pos, n_win, out=np.zeros_like(pos), where=n_win > 0)
    return conf, n_win


def evaluate_detector(params: dict, data: DetectionData, settings: IndicatorStudySettings,
                      base_filters: FilterConfig | None = None) -> dict[str, float]:
    filters = filter_config(params, base_filters)
    win = window_config(params, WindowConfig(min_len=int(data.window_len.min()), max_len=data.max_len,
                                             step=data.grid_step), data.grid_step)
    conf, n_win = confidence(data, filters, win.min_len, win.max_len, win.step)
    g = data.groups
    out: dict[str, float] = {"n_windows_min": min_windows_used(data, n_win)}
    for split in ("train", "valid"):
        mask = (g["split"] == split).to_numpy()
        y = g.loc[mask, "label"].to_numpy()
        s = conf[mask]
        ap = average_precision(y, s)
        base = float(np.nanmean(y))
        out[f"ap_{split}"] = ap
        out[f"lift_{split}"] = ap / base if base else float("nan")
        out[f"mean_conf_{split}"] = float(s.mean())
        for k, v in alarm_stats(y, s, 0.3).items():
            out[f"{k}_at_0.3_{split}"] = v
    valid = out["n_windows_min"] >= settings.min_windows and np.isfinite(out["ap_train"])
    out["objective"] = out["ap_train"] if valid else 0.0
    return out
