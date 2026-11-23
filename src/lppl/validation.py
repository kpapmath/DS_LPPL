"""Validation suite for a stored scan (``lppl validate``, notebook 05, docs/validation.md).

Every test compares the LPPLS positive confidence with fair references (base rate, a
naive momentum score, time-shifted placebos, bubble-free simulated prices) and respects
the time order of the data:

1. skill          AP / precision / recall for forward-drawdown events, all dates + train/valid
2. calibration    event rate by confidence band
3. bootstrap      month-block bootstrap CIs (labels of nearby dates overlap)
4. placebo        circular time shifts of each ticker's confidence -> p-value
5. grid           robustness to the event definition (horizon x threshold)
6. timing         alarm episodes: lead time to the top, tc window hit rate
7. walk_forward   yearly folds; filters re-tuned on past dates only (Optuna, optional)
8. null_model     re-scan bubble-free bootstrap paths: false-alarm rate
9. backtest       de-risk on alarms vs buy & hold, momentum (same alarm rate), placebo
"""

from __future__ import annotations

import datetime as dt
import json
import logging
import zlib
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from pydantic import BaseModel

from lppl.config import PROJECT_ROOT, AppConfig
from lppl.tuning.detection import alarm_stats, average_precision, drawdown_labels

log = logging.getLogger(__name__)
DEFAULT_VALIDATION_CONFIG = PROJECT_ROOT / "config" / "validation.yaml"
TRADING_DAYS = 252


# --------------------------------------------------------------------------- settings
class EventSettings(BaseModel):
    horizon: int = 63
    k_sigma: float = 1.5
    vol_window: int = 750


class GridSettings(BaseModel):
    horizons: list[int] = [21, 63, 126]
    k_sigmas: list[float] = [1.0, 1.5, 2.0]


class WalkForwardSettings(BaseModel):
    first_test_year: int = 2021
    trials: int = 100


class NullSettings(BaseModel):
    paths: int = 1
    tickers: list[str] | None = None
    block_days: int = 20
    scan_step: int = 21
    window_step: int = 50


class BacktestSettings(BaseModel):
    cut: float = 0.5
    cost_bps: float = 10.0
    placebo_shifts: int = 200


class LabelSuiteSettings(BaseModel):
    names: list[str] = ["drawdown_63", "drawdown_126", "major_top_126", "rally_end_126"]
    k_sigma: float = 1.5
    top_lookback: int = 252
    reps: int = 1000              # placebo shifts and bootstrap replications per label
    tc_test: bool = True          # are major tops inside the predicted tc window?
    tc_reps: int = 2000


class ValidationSettings(BaseModel):
    seed: int = 0
    alarm_threshold: float = 0.3
    events: EventSettings = EventSettings()
    grid: GridSettings = GridSettings()
    split_date: dt.date = dt.date(2024, 1, 1)
    embargo_days: int = 63
    bootstrap: int = 1000
    placebo: int = 1000
    placebo_min_shift: int = 126
    walk_forward: WalkForwardSettings = WalkForwardSettings()
    null_model: NullSettings = NullSettings()
    backtest: BacktestSettings = BacktestSettings()
    labels: LabelSuiteSettings = LabelSuiteSettings()


def load_validation_config(path: str | Path | None = None) -> ValidationSettings:
    with open(Path(path) if path else DEFAULT_VALIDATION_CONFIG) as fh:
        return ValidationSettings.model_validate(yaml.safe_load(fh) or {})


# --------------------------------------------------------------------------- events
def build_events(ind: pd.DataFrame, prices: pd.DataFrame, ev: EventSettings,
                 split_date, embargo_days: int) -> pd.DataFrame:
    """One row per (ticker, analysis date): confidence, forward-drawdown label, momentum
    baseline score and split. ``ind`` are ``bubble_indicator`` rows of one run."""
    rows = []
    for ticker, g in ind.sort_values(["ticker", "date"]).groupby("ticker", sort=True):
        lab = drawdown_labels(prices[ticker], g["date"], ev.horizon, ev.k_sigma, ev.vol_window)
        rows.append(pd.DataFrame({
            "ticker": ticker, "date": g["date"].to_numpy(), "conf": g["pos_conf"].to_numpy(float),
            "neg_conf": g["neg_conf"].to_numpy(float),
            "tc_q25": g["tc_q25_date"].to_numpy(), "tc_median": g["tc_median_date"].to_numpy(),
            "tc_q75": g["tc_q75_date"].to_numpy(),
            "label": lab["label"].to_numpy(), "fwd_dd": lab["fwd_dd"].to_numpy(),
            "threshold": lab["threshold"].to_numpy(), "momentum": lab["momentum"].to_numpy()}))
    out = pd.concat(rows, ignore_index=True)
    split = pd.Timestamp(split_date)
    embargo_end = split - pd.offsets.BDay(embargo_days)
    out["split"] = np.where(out["date"] >= split, "valid",
                            np.where(out["date"] < embargo_end, "train", "embargo"))
    return out


def _labeled(ev: pd.DataFrame, split: str = "all") -> pd.DataFrame:
    d = ev[ev["label"].notna() & ev["momentum"].notna()]
    return d if split == "all" else d[d["split"] == split]


# --------------------------------------------------------------------------- 1. skill
def skill(ev: pd.DataFrame, alarm: float) -> pd.DataFrame:
    rows = []
    for split in ("all", "train", "valid"):
        d = _labeled(ev, split)
        if d.empty:
            continue
        base = float(d["label"].mean())
        r = {"split": split, "n": len(d), "events": int(d["label"].sum()), "base_rate": base,
             "ap_lppl": average_precision(d["label"], d["conf"]),
             "ap_momentum": average_precision(d["label"], d["momentum"])}
        r["lift_lppl"] = r["ap_lppl"] / base if base else np.nan
        r["lift_momentum"] = r["ap_momentum"] / base if base else np.nan
        r.update({f"{k}_at_alarm": v for k, v in alarm_stats(d["label"], d["conf"], alarm).items()})
        rows.append(r)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- 2. calibration
CAL_LABELS = ["0", "(0, 0.1]", "(0.1, 0.3)", "[0.3, 0.6)", ">= 0.6"]


def calibration(ev: pd.DataFrame) -> pd.DataFrame:
    d = _labeled(ev).copy()
    # bins: exactly 0 | (0,0.1] | (0.1,0.3) | [0.3,0.6) | >= 0.6
    c = d["conf"].to_numpy()
    band = np.select([c <= 0, c <= 0.1, c < 0.3, c < 0.6], CAL_LABELS[:4], CAL_LABELS[4])
    d["band"] = pd.Categorical(band, CAL_LABELS, ordered=True)
    out = d.groupby("band", observed=False).agg(
        n=("label", "size"), event_rate=("label", "mean"), mean_fwd_dd=("fwd_dd", "mean"))
    out["mean_fwd_dd"] = np.expm1(out["mean_fwd_dd"])  # log -> simple return
    return out.reset_index()


# --------------------------------------------------------------------------- 3. bootstrap
def block_bootstrap(ev: pd.DataFrame, reps: int, seed: int) -> pd.DataFrame:
    """Resample calendar months (all tickers of a month together) with replacement.

    This keeps the overlap of nearby labels and the cross-sectional correlation inside a
    block, which an ordinary bootstrap would ignore."""
    rng = np.random.default_rng(seed)
    rows = []
    for split in ("all", "valid"):
        d = _labeled(ev, split).reset_index(drop=True)
        if d.empty:
            continue
        months = d["date"].dt.to_period("M").to_numpy()
        blocks = [np.flatnonzero(months == m) for m in pd.unique(months)]
        y, s, m = (d[c].to_numpy(float) for c in ("label", "conf", "momentum"))

        def stats(idx, y=y, s=s, m=m):
            yy = y[idx]
            base = yy.mean()
            a, b = average_precision(yy, s[idx]), average_precision(yy, m[idx])
            return a, b, base, a - b, a - base

        point = stats(np.arange(len(d)))
        draws = np.array([stats(np.concatenate([blocks[i] for i in
                                                rng.integers(0, len(blocks), len(blocks))]))
                          for _ in range(reps)])
        names = ["ap_lppl", "ap_momentum", "base_rate", "lppl_minus_momentum", "lppl_minus_base"]
        for j, name in enumerate(names):
            col = draws[:, j][np.isfinite(draws[:, j])]
            rows.append({"split": split, "metric": name, "point": point[j],
                         "ci_low": np.quantile(col, 0.025), "ci_high": np.quantile(col, 0.975),
                         "share_above_0": float((col > 0).mean()) if "minus" in name else np.nan,
                         "n_blocks": len(blocks)})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- 4. placebo
def placebo_test(ev: pd.DataFrame, reps: int, min_shift_rows: int, seed: int,
                 score: str = "conf") -> dict[str, Any]:
    """Shift each ticker's score series circularly in time (labels stay put) and recompute
    the pooled AP. A real signal should beat nearly all shifted versions."""
    rng = np.random.default_rng(seed)
    d = ev.sort_values(["ticker", "date"]).reset_index(drop=True)
    keep = d["label"].notna() & d["momentum"].notna()
    actual = average_precision(d.loc[keep, "label"], d.loc[keep, score])
    groups = [np.flatnonzero((d["ticker"] == t).to_numpy()) for t in d["ticker"].unique()]
    s = d[score].to_numpy(float)
    y = d["label"].to_numpy(float)
    k = keep.to_numpy()
    dist = np.empty(reps)
    for r in range(reps):
        shifted = s.copy()
        for idx in groups:
            n = len(idx)
            lo, hi = min_shift_rows, n - min_shift_rows
            if hi <= lo:
                continue
            shifted[idx] = np.roll(s[idx], int(rng.integers(lo, hi + 1)))
        dist[r] = average_precision(y[k], shifted[k])
    return {"score": score, "actual_ap": actual, "placebo_mean": float(np.nanmean(dist)),
            "placebo_p95": float(np.nanquantile(dist, 0.95)),
            "p_value": float((1 + np.sum(dist >= actual)) / (reps + 1)), "distribution": dist}


# --------------------------------------------------------------------------- 5. grid
def event_grid(ind: pd.DataFrame, prices: pd.DataFrame, s: ValidationSettings) -> pd.DataFrame:
    rows = []
    for h in s.grid.horizons:
        for k in s.grid.k_sigmas:
            ev = build_events(ind, prices, EventSettings(horizon=h, k_sigma=k,
                                                         vol_window=s.events.vol_window),
                              s.split_date, s.embargo_days)
            for split in ("all", "valid"):
                d = _labeled(ev, split)
                base = float(d["label"].mean())
                ap, am = average_precision(d["label"], d["conf"]), average_precision(d["label"], d["momentum"])
                rows.append({"horizon": h, "k_sigma": k, "split": split, "n": len(d), "base_rate": base,
                             "ap_lppl": ap, "ap_momentum": am, "lift_lppl": ap / base if base else np.nan,
                             "lift_momentum": am / base if base else np.nan})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- 6. timing
def timing(ev: pd.DataFrame, prices: pd.DataFrame, alarm: float, gap_days: int = 45,
           look: int = 126) -> pd.DataFrame:
    """Alarm episodes (consecutive alarm dates of a ticker, gaps <= ``gap_days``)."""
    rows = []
    hot = ev[ev["conf"] >= alarm].sort_values(["ticker", "date"])
    for ticker, g in hot.groupby("ticker"):
        p = prices[ticker].dropna()
        for _, e in g.groupby((g["date"].diff().dt.days > gap_days).cumsum()):
            first, peak = e["date"].min(), e.loc[e["conf"].idxmax()]
            i0 = p.index.get_loc(first)
            fwd = p.iloc[i0 : i0 + look + 1]
            complete = len(fwd) == look + 1
            top = fwd.idxmax()
            after = p.loc[top:].iloc[: look + 1]
            in_tc = (pd.notna(peak["tc_q25"]) and pd.Timestamp(peak["tc_q25"]) <= top
                     <= pd.Timestamp(peak["tc_q75"]))
            rows.append({
                "ticker": ticker, "first_alarm": first, "last_alarm": e["date"].max(),
                "n_alarm_dates": len(e), "peak_conf": peak["conf"], "peak_conf_date": peak["date"],
                "tc_q25": peak["tc_q25"], "tc_q75": peak["tc_q75"], "top_date": top,
                "lead_trading_days": int(p.index.get_loc(top) - i0),
                "gain_to_top": float(fwd.max() / p.iloc[i0] - 1),
                "fall_after_top": float(after.min() / after.iloc[0] - 1),
                "top_in_tc_window": bool(in_tc),
                "top_minus_tc_median_days": (top - pd.Timestamp(peak["tc_median"])).days
                if pd.notna(peak["tc_median"]) else np.nan,
                "complete": complete and len(after) == look + 1,
            })
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- 7. walk-forward
def walk_forward(fits: pd.DataFrame, prices: pd.DataFrame, run_cfg: AppConfig,
                 s: ValidationSettings, space: dict | None = None) -> pd.DataFrame:
    """Yearly folds. Train = all labelled dates before the test year minus the embargo;
    test = the year's labelled dates. Reports the stored detector (as scanned), a detector
    re-tuned on the train dates only (if optuna is available) and the baselines."""
    from lppl.model.indicator import WindowConfig
    from lppl.tuning.detection import build_detection_data, confidence, min_windows_used
    from lppl.tuning.settings import (
        IndicatorStudySettings,
        default_params,
        filter_config,
        load_tuning_config,
        suggest,
        window_config,
    )

    space = space or load_tuning_config().indicator.space
    st = IndicatorStudySettings(horizon=s.events.horizon, k_sigma=s.events.k_sigma,
                                vol_window=s.events.vol_window, space=space)
    data = build_detection_data(fits, prices, st)
    g = data.groups
    y = g["label"].to_numpy(float)
    mom = g["momentum"].to_numpy(float)
    years = g["date"].dt.year.to_numpy()
    defaults = default_params(run_cfg, "indicator", space, data.grid_step)

    def conf_for(params):
        filt = filter_config(params, run_cfg.filters)
        win = window_config(params, WindowConfig(min_len=int(data.window_len.min()),
                                                 max_len=data.max_len, step=data.grid_step),
                            data.grid_step)
        c, n = confidence(data, filt, win.min_len, win.max_len, win.step)
        return c, min_windows_used(data, n)

    conf_default, _ = conf_for(defaults)
    try:
        import optuna
        optuna.logging.set_verbosity(optuna.logging.WARNING)
    except ImportError:
        optuna = None
        log.warning("optuna not installed: walk-forward re-tuning skipped (stored detector only)")

    ok = np.isfinite(y) & np.isfinite(mom)
    rows = []
    for year in sorted(set(years[ok])):
        if year < s.walk_forward.first_test_year:
            continue
        start = pd.Timestamp(f"{year}-01-01")
        train = ok & (g["date"] < start - pd.offsets.BDay(s.embargo_days)).to_numpy()
        test = ok & (years == year)
        if test.sum() == 0 or y[test].sum() == 0 or y[train].sum() == 0:
            continue
        r = {"test_year": int(year), "n_train": int(train.sum()), "n_test": int(test.sum()),
             "base_rate_test": float(y[test].mean()),
             "ap_stored_test": average_precision(y[test], conf_default[test]),
             "ap_momentum_test": average_precision(y[test], mom[test]),
             "ap_stored_train": average_precision(y[train], conf_default[train])}
        if optuna is not None and s.walk_forward.trials > 0:
            def objective(trial, train=train):
                c, nmin = conf_for(suggest(trial, space))
                ap = average_precision(y[train], c[train])
                return ap if nmin >= 5 and np.isfinite(ap) else 0.0

            study = optuna.create_study(direction="maximize",
                                        sampler=optuna.samplers.TPESampler(seed=s.seed + int(year)))
            study.enqueue_trial(defaults)
            study.optimize(objective, n_trials=s.walk_forward.trials)
            c, _ = conf_for(study.best_params)
            r["ap_tuned_train"] = float(study.best_value)
            r["ap_tuned_test"] = average_precision(y[test], c[test])
        rows.append(r)
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- 8. null model
def stationary_bootstrap(returns: np.ndarray, n: int, mean_block: int,
                         rng: np.random.Generator) -> np.ndarray:
    """Politis-Romano stationary bootstrap: blocks of geometric length (mean ``mean_block``)
    drawn from ``returns`` with wrap-around. Keeps short-range dependence and volatility
    clustering, and destroys the long-range structure an LPPLS bubble needs."""
    r = np.asarray(returns, dtype=float)
    r = r[np.isfinite(r)]
    out = np.empty(n)
    i = 0
    p = 1.0 / mean_block
    while i < n:
        start = int(rng.integers(len(r)))
        length = int(rng.geometric(p))
        take = min(length, n - i)
        out[i : i + take] = r[(start + np.arange(take)) % len(r)]
        i += take
    return out


def null_model(fits: pd.DataFrame, prices: pd.DataFrame, run_cfg: AppConfig, s: ValidationSettings,
               n_jobs: int = -1) -> pd.DataFrame:
    """Alarm rate of the detector on bubble-free simulated paths vs. on the real prices,
    with the same (coarser) window grid."""
    from lppl.model.indicator import ScanSettings, WindowConfig, scan_series
    from lppl.tuning.detection import build_detection_data, confidence
    from lppl.tuning.settings import IndicatorStudySettings

    ns = s.null_model
    win = WindowConfig(min_len=run_cfg.windows.min_len, max_len=run_cfg.windows.max_len,
                       step=ns.window_step)
    settings = ScanSettings(windows=win, bounds=run_cfg.bounds, filters=run_cfg.filters,
                            ga=run_cfg.ga, method=run_cfg.scan.method, robust=run_cfg.robust)
    tickers = ns.tickers or sorted(fits["ticker"].unique())
    if run_cfg.robust.enabled:
        # The robust confidence cannot be rebuilt from the stored fits (it combines seeds and
        # dates), so the real prices are rescanned with exactly the null paths' settings.
        parts = []
        for ticker in tickers:
            _, ind = scan_series(prices[ticker].dropna(), ticker, settings, start=run_cfg.scan.start,
                                 end=run_cfg.scan.end, step=ns.scan_step, master_seed=run_cfg.seed,
                                 n_jobs=n_jobs)
            parts.append(ind.rename(columns={"pos_conf": "conf"}))
        real = pd.concat(parts, ignore_index=True)
    else:
        # real detector on the same window lengths (subset of the stored grid)
        data = build_detection_data(fits, prices, IndicatorStudySettings(space={}))
        lengths = win.lengths()
        stored = set(np.unique(data.window_len))
        if not set(lengths) <= stored:
            raise ValueError(f"null-model windows {lengths} are not all in the stored scan")
        real_conf, _ = confidence(data, run_cfg.filters, win.min_len, win.max_len, win.step)
        real = data.groups.assign(conf=real_conf)
    rows = []
    for ticker in tickers:
        px = prices[ticker].dropna()
        logp = np.log(px.to_numpy(float))
        r_real = real[real["ticker"] == ticker]
        for path in range(ns.paths):
            rng = np.random.default_rng([s.seed, path, zlib.crc32(ticker.encode())])
            sim = stationary_bootstrap(np.diff(logp), len(logp) - 1, ns.block_days, rng)
            sim_px = pd.Series(np.exp(logp[0] + np.r_[0.0, np.cumsum(sim)]), index=px.index)
            _, ind = scan_series(sim_px, f"{ticker}#null{path}", settings, start=run_cfg.scan.start,
                                 end=run_cfg.scan.end, step=ns.scan_step,
                                 master_seed=s.seed + 1000 + path, n_jobs=n_jobs)
            rows.append({"ticker": ticker, "path": path, "n_dates_null": len(ind),
                         "alarm_rate_null": float((ind["pos_conf"] >= s.alarm_threshold).mean()),
                         "mean_conf_null": float(ind["pos_conf"].mean()),
                         "alarm_rate_real": float((r_real["conf"] >= s.alarm_threshold).mean()),
                         "mean_conf_real": float(r_real["conf"].mean()),
                         "n_dates_real": len(r_real)})
            log.info("null model %s path %d: alarm rate %.3f (real %.3f)", ticker, path,
                     rows[-1]["alarm_rate_null"], rows[-1]["alarm_rate_real"])
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- 9. backtest
def _exposure(dates: pd.DatetimeIndex, sig_dates, signal: np.ndarray, cut: float) -> pd.Series:
    """Exposure per trading day. The signal at the close of an analysis date applies from
    the next trading day until the next analysis date (inclusive): no look-ahead."""
    s = pd.Series(np.where(signal, 1.0 - cut, 1.0), index=pd.DatetimeIndex(sig_dates))
    full = s.reindex(dates.union(s.index)).ffill().shift(1)
    return full.reindex(dates).fillna(1.0)


def _metrics(r: pd.Series, exposure: pd.Series | None = None) -> dict[str, float]:
    r = r.dropna()
    if r.empty:
        return {}
    eq = (1 + r).cumprod()
    years = len(r) / TRADING_DAYS
    ann = eq.iloc[-1] ** (1 / years) - 1 if years > 0 else np.nan
    vol = r.std() * np.sqrt(TRADING_DAYS)
    out = {"ann_return": float(ann), "ann_vol": float(vol),
           "sharpe": float(r.mean() * TRADING_DAYS / vol) if vol else np.nan,
           "max_drawdown": float((eq / eq.cummax() - 1).min()), "total_return": float(eq.iloc[-1] - 1)}
    if exposure is not None:
        e = exposure.reindex(r.index)
        out["avg_exposure"] = float(e.mean())
    return out


def backtest(ev: pd.DataFrame, prices: pd.DataFrame, s: ValidationSettings
             ) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Equal-weight portfolio of the scanned tickers, daily rebalanced. Strategies:
    buy & hold; LPPLS (cut exposure on alarms); momentum with the *same alarm rate*;
    LPPLS placebo (alarm series circularly shifted in time, averaged)."""
    bt = s.backtest
    rng = np.random.default_rng(s.seed)
    alarm_rate = float((ev["conf"] >= s.alarm_threshold).mean())
    mom_thr = float(np.nanquantile(ev["momentum"], 1 - alarm_rate)) if alarm_rate > 0 else np.inf
    cost = bt.cost_bps / 1e4
    start = ev["date"].min()
    rets, exps = {}, {}
    placebo_rets = []
    for ticker, g in ev.sort_values("date").groupby("ticker"):
        p = prices[ticker].dropna().loc[start:]
        r = p.pct_change().iloc[1:]
        days = r.index
        signals = {"lppl": (g["conf"] >= s.alarm_threshold).to_numpy(),
                   "momentum": (g["momentum"].fillna(-np.inf) >= mom_thr).to_numpy()}
        rets.setdefault("buy_and_hold", {})[ticker] = r
        for name, sig in signals.items():
            e = _exposure(days, g["date"], sig, bt.cut)
            rets.setdefault(name, {})[ticker] = e * r - cost * e.diff().abs().fillna(0)
            exps.setdefault(name, {})[ticker] = e
        n = len(signals["lppl"])
        shifts = rng.integers(1, n, size=bt.placebo_shifts) if n > 1 else np.zeros(bt.placebo_shifts, int)
        placebo_rets.append([(lambda e: e * r - cost * e.diff().abs().fillna(0))(
            _exposure(days, g["date"], np.roll(signals["lppl"], k), bt.cut)) for k in shifts])
    port = {name: pd.DataFrame(d).mean(axis=1) for name, d in rets.items()}
    port["lppl_placebo"] = pd.concat(
        [pd.DataFrame({i: pr[j] for i, pr in enumerate(placebo_rets)}).mean(axis=1)
         for j in range(bt.placebo_shifts)], axis=1).mean(axis=1)
    exp_port = {name: pd.DataFrame(d).mean(axis=1) for name, d in exps.items()}
    rows = []
    split = pd.Timestamp(s.split_date)
    for period, mask_fn in (("all", lambda x: x), ("valid", lambda x: x[x.index >= split])):
        for name, r in port.items():
            m = _metrics(mask_fn(r), mask_fn(exp_port[name]) if name in exp_port else None)
            rows.append({"period": period, "strategy": name, **m})
    curves = pd.DataFrame({k: (1 + v.fillna(0)).cumprod() for k, v in port.items()})
    return pd.DataFrame(rows), curves


# --------------------------------------------------------------------------- orchestration
@dataclass
class ValidationReport:
    run_id: str
    label: str
    settings: ValidationSettings
    tables: dict[str, pd.DataFrame] = field(default_factory=dict)
    placebo: dict[str, dict] = field(default_factory=dict)
    curves: pd.DataFrame | None = None
    summary: dict[str, Any] = field(default_factory=dict)


def run_validation(engine, run_id: str, label: str = "", settings: ValidationSettings | None = None,
                   run_walk_forward: bool = True, run_null: bool = True, n_jobs: int = -1,
                   space: dict | None = None) -> ValidationReport:
    from lppl.data import db
    from lppl.tuning.runners import source_run_config

    s = settings or load_validation_config()
    ind = db.load_indicator(engine, run_id)
    if ind.empty:
        raise RuntimeError(f"run {run_id} has no indicator rows")
    prices = db.load_all_prices(engine, "adj_close")
    run_cfg = source_run_config(engine, run_id)
    fits = db.load_fits(engine, run_id) if (run_walk_forward or run_null) else None
    rep = ValidationReport(run_id=run_id, label=label, settings=s)

    ev = build_events(ind, prices, s.events, s.split_date, s.embargo_days)
    rep.tables["events"] = ev
    rep.tables["skill"] = skill(ev, s.alarm_threshold)
    rep.tables["calibration"] = calibration(ev)
    rep.tables["bootstrap"] = block_bootstrap(ev, s.bootstrap, s.seed)
    step_rows = _rows_for_trading_days(ev, prices, s.placebo_min_shift)
    for score in ("conf", "momentum"):
        rep.placebo[score] = placebo_test(ev, s.placebo, step_rows, s.seed, score)
    rep.tables["placebo"] = pd.DataFrame([{k: v for k, v in p.items() if k != "distribution"}
                                          for p in rep.placebo.values()])
    rep.tables["grid"] = event_grid(ind, prices, s)
    rep.tables["timing"] = timing(ev, prices, s.alarm_threshold)
    if run_walk_forward and run_cfg.robust.enabled:
        log.warning("walk-forward re-tuning skipped: it requalifies stored fits, which the robust "
                    "variant does not allow (its filters are frozen by design)")
    elif run_walk_forward:
        rep.tables["walk_forward"] = walk_forward(fits, prices, run_cfg, s, space)
    if run_null and s.null_model.paths > 0:
        rep.tables["null_model"] = null_model(fits, prices, run_cfg, s, n_jobs=n_jobs)
    rep.tables["backtest"], rep.curves = backtest(ev, prices, s)
    if s.labels.names:
        rep.tables["labels"] = label_suite(ev, prices, s, step_rows)
    if s.labels.tc_test:
        res, ep = tc_window_test(ev, prices, s)
        rep.tables["tc_window"] = pd.DataFrame([res])
        rep.tables["tc_episodes"] = ep
    add_holm(rep)
    rep.summary = summarize(rep)
    return rep


def _rows_for_trading_days(ev: pd.DataFrame, prices: pd.DataFrame, days: int) -> int:
    """Convert a minimum shift in trading days into analysis-date rows."""
    t = ev["ticker"].iloc[0]
    pos = prices[t].dropna().index.get_indexer(pd.DatetimeIndex(ev.loc[ev["ticker"] == t, "date"]))
    spacing = float(np.median(np.diff(pos))) if len(pos) > 1 else 1.0
    return max(1, int(np.ceil(days / max(spacing, 1.0))))


def summarize(rep: ValidationReport) -> dict[str, Any]:
    t = rep.tables
    sk = t["skill"].set_index("split")
    bs = t["bootstrap"].set_index(["split", "metric"])
    out: dict[str, Any] = {"run_id": rep.run_id, "label": rep.label,
                           "event": rep.settings.events.model_dump(), "alarm": rep.settings.alarm_threshold}
    for split in sk.index:
        for c in ("n", "events", "base_rate", "ap_lppl", "ap_momentum", "lift_lppl",
                  "precision_at_alarm", "recall_at_alarm", "alarm_rate_at_alarm"):
            out[f"{split}.{c}"] = _num(sk.loc[split, c])
    for (split, metric), r in bs.iterrows():
        out[f"{split}.ci.{metric}"] = [_num(r["ci_low"]), _num(r["ci_high"])]
    for score, p in rep.placebo.items():
        out[f"placebo.{score}.p_value"] = p["p_value"]
    cal = t["calibration"]
    rates = cal.loc[cal["n"] >= 10, "event_rate"].to_numpy()
    out["calibration.monotone"] = bool(np.all(np.diff(rates) >= -0.02)) if len(rates) > 1 else None
    tm = t["timing"]
    done = tm[tm["complete"]] if not tm.empty else tm
    if not done.empty:
        out["timing.episodes_complete"] = len(done)
        out["timing.median_lead_trading_days"] = _num(done["lead_trading_days"].median())
        out["timing.top_in_tc_window"] = int(done["top_in_tc_window"].sum())
        out["timing.median_gain_to_top"] = _num(done["gain_to_top"].median())
        out["timing.median_fall_after_top"] = _num(done["fall_after_top"].median())
    if "walk_forward" in t and not t["walk_forward"].empty:
        wf = t["walk_forward"]
        for c in ("base_rate_test", "ap_stored_test", "ap_momentum_test", "ap_tuned_test"):
            if c in wf:
                out[f"walk_forward.mean_{c}"] = _num(wf[c].mean())
    if "null_model" in t and not t["null_model"].empty:
        nm = t["null_model"]
        real = nm.drop_duplicates("ticker")
        k_real = float((real["alarm_rate_real"] * real["n_dates_real"]).sum())
        n_real = float(real["n_dates_real"].sum())
        k_null = float((nm["alarm_rate_null"] * nm["n_dates_null"]).sum())
        n_null = float(nm["n_dates_null"].sum())
        out["null.alarm_rate_real"] = _num(k_real / n_real)
        out["null.alarm_rate_null"] = _num(k_null / n_null)
        out["null.p_value_real_gt_null"] = _num(two_proportion_p(k_real, n_real, k_null, n_null))
    b = t["backtest"].set_index(["period", "strategy"])
    for (period, strat), r in b.iterrows():
        for c in ("ann_return", "sharpe", "max_drawdown"):
            out[f"backtest.{period}.{strat}.{c}"] = _num(r[c])
    if "labels" in t and not t["labels"].empty:
        for _, r in t["labels"].iterrows():
            for c in ("n", "events", "base_rate", "ap_lppl", "ap_momentum", "lift_lppl",
                      "p_placebo", "p_holm", "precision_at_alarm"):
                if c in r:
                    out[f"labels.{r['label']}.{c}"] = _num(r[c])
            out[f"labels.{r['label']}.ci.lppl_minus_base"] = [_num(r["lppl_minus_base_ci_low"]),
                                                              _num(r["lppl_minus_base_ci_high"])]
            out[f"labels.{r['label']}.ci.lppl_minus_momentum"] = [
                _num(r["lppl_minus_momentum_ci_low"]), _num(r["lppl_minus_momentum_ci_high"])]
    if "tc_window" in t and not t["tc_window"].empty:
        for c, v in t["tc_window"].iloc[0].items():
            out[f"tc_window.{c}"] = _num(v)
    out["verdict"] = verdict(out)
    return out


def verdict(s: dict) -> dict[str, bool | None]:
    """Plain yes/no answers to the questions the suite is built for (5 % level)."""
    def ci_above(key):
        v = s.get(key)
        return None if v is None else bool(v[0] > 0)

    return {
        "beats_base_rate_all_dates": ci_above("all.ci.lppl_minus_base"),
        "beats_base_rate_validation": ci_above("valid.ci.lppl_minus_base"),
        "beats_momentum_all_dates": ci_above("all.ci.lppl_minus_momentum"),
        "beats_momentum_validation": ci_above("valid.ci.lppl_minus_momentum"),
        "beats_time_shifted_placebo": (s.get("placebo.conf.p_value", 1) < 0.05),
        "calibrated_monotone": s.get("calibration.monotone"),
        "fires_more_on_real_than_bubble_free_prices": (
            None if s.get("null.p_value_real_gt_null") is None
            else bool(s["null.p_value_real_gt_null"] < 0.05)),
        **{f"beats_placebo_{k.split('.')[1]}_holm": bool(v < 0.05)
           for k, v in s.items() if k.startswith("labels.") and k.endswith(".p_holm") and v is not None},
        "tops_inside_tc_window_holm": (None if s.get("tc_window.p_holm") is None
                                       else bool(s["tc_window.p_holm"] < 0.05)),
    }


def two_proportion_p(k1: float, n1: float, k2: float, n2: float) -> float:
    """One-sided p-value of H1: rate1 > rate2 (pooled two-proportion z-test)."""
    from scipy.stats import norm

    if min(n1, n2) == 0:
        return float("nan")
    p = (k1 + k2) / (n1 + n2)
    se = np.sqrt(p * (1 - p) * (1 / n1 + 1 / n2))
    return float(norm.sf((k1 / n1 - k2 / n2) / se)) if se > 0 else float("nan")


def _num(v):
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return None if not np.isfinite(f) else f


# --------------------------------------------------------------------------- 10. label suite
def label_suite(ev: pd.DataFrame, prices: pd.DataFrame, s: ValidationSettings,
                min_shift_rows: int) -> pd.DataFrame:
    """The skill tests (AP vs base rate and momentum, month-block bootstrap, time-shift
    placebo) for each pre-registered label of ``lppl.labels``, with Holm-adjusted p-values."""
    from lppl.labels import compute_labels

    ls = s.labels
    rows = []
    for name in ls.names:
        d = ev.copy()
        for ticker, idx in d.groupby("ticker").groups.items():
            d.loc[idx, "label"] = compute_labels(prices[ticker], d.loc[idx, "date"], name,
                                                 ls.k_sigma, s.events.vol_window, ls.top_lookback)
        lab = _labeled(d)
        bs = block_bootstrap(d, ls.reps, s.seed).set_index(["split", "metric"])
        pl = placebo_test(d, ls.reps, min_shift_rows, s.seed)
        r = {"label": name, "n": len(lab), "events": int(lab["label"].sum()),
             "base_rate": float(lab["label"].mean()),
             "ap_lppl": average_precision(lab["label"], lab["conf"]),
             "ap_momentum": average_precision(lab["label"], lab["momentum"]),
             "placebo_mean": pl["placebo_mean"], "p_placebo": pl["p_value"]}
        r["lift_lppl"] = r["ap_lppl"] / r["base_rate"] if r["base_rate"] else np.nan
        for metric in ("lppl_minus_base", "lppl_minus_momentum"):
            b = bs.loc[("all", metric)]
            r[f"{metric}_ci_low"], r[f"{metric}_ci_high"] = b["ci_low"], b["ci_high"]
        r.update({f"{k}_at_alarm": v for k, v in
                  alarm_stats(lab["label"], lab["conf"], s.alarm_threshold).items()})
        rows.append(r)
    return pd.DataFrame(rows)


def tc_window_test(ev: pd.DataFrame, prices: pd.DataFrame, s: ValidationSettings,
                   gap_days: int = 45) -> tuple[dict[str, Any], pd.DataFrame]:
    """Does a major top fall inside the predicted [tc_q25, tc_q75] window?

    One row per alarm episode (its first alarm date). Placebo: the windows' offsets from the
    alarm date are permuted across episodes, which keeps the alarm dates (inside rallies) and the
    window widths and only breaks the link between an episode and *its* predicted crash time."""
    from lppl.labels import top_dates

    hot = ev[(ev["conf"] >= s.alarm_threshold) & ev["tc_q25"].notna()].sort_values(["ticker", "date"])
    rows = []
    for ticker, g in hot.groupby("ticker"):
        px = prices[ticker].dropna()
        tops = top_dates(px, k_sigma=s.labels.k_sigma, lookback=s.labels.top_lookback,
                         vol_window=s.events.vol_window)
        last_decidable = px.index[-1] - pd.Timedelta(days=183)  # tops need 126 days to confirm
        for _, e in g.groupby((g["date"].diff().dt.days > gap_days).cumsum()):
            f = e.iloc[0]
            q25, q75 = pd.Timestamp(f["tc_q25"]), pd.Timestamp(f["tc_q75"])
            if q75 > last_decidable:
                continue
            rows.append({"ticker": ticker, "alarm_date": f["date"], "conf": f["conf"],
                         "tc_q25": q25, "tc_q75": q75,
                         "lo_days": (q25 - f["date"]).days, "hi_days": (q75 - f["date"]).days,
                         "hit": bool(((tops >= q25) & (tops <= q75)).any()),
                         "_tops": tops.to_numpy("datetime64[D]").astype("int64")})
    ep = pd.DataFrame(rows)
    if ep.empty:
        return {"episodes": 0}, ep
    t2 = pd.DatetimeIndex(ep["alarm_date"]).to_numpy("datetime64[D]").astype("int64")  # in days
    lo, hi = ep["lo_days"].to_numpy(), ep["hi_days"].to_numpy()
    tops = ep["_tops"].tolist()

    def hit_rate(perm):
        a, b = t2 + lo[perm], t2 + hi[perm]
        return np.mean([((tp >= a[i]) & (tp <= b[i])).any() for i, tp in enumerate(tops)])

    rng = np.random.default_rng(s.seed)
    actual = hit_rate(np.arange(len(ep)))
    dist = np.array([hit_rate(rng.permutation(len(ep))) for _ in range(s.labels.tc_reps)])
    res = {"episodes": len(ep), "hits": int(ep["hit"].sum()), "hit_rate": float(actual),
           "placebo_mean": float(dist.mean()), "placebo_p95": float(np.quantile(dist, 0.95)),
           "p_value": float((1 + np.sum(dist >= actual)) / (1 + len(dist))),
           "median_width_days": float((ep["hi_days"] - ep["lo_days"]).median())}
    return res, ep.drop(columns="_tops")


def add_holm(rep: "ValidationReport") -> None:
    """Holm adjustment over the pre-registered family: one placebo p-value per label plus the
    tc-window test."""
    from lppl.labels import holm

    lab = rep.tables.get("labels")
    if lab is None or lab.empty:
        return
    p = list(lab["p_placebo"])
    tc = rep.tables.get("tc_window")
    has_tc = tc is not None and not tc.empty and "p_value" in tc
    if has_tc:
        p.append(float(tc["p_value"].iloc[0]))
    adj = holm(p)
    lab["p_holm"] = adj[: len(lab)]
    if has_tc:
        tc["p_holm"] = adj[-1]


# --------------------------------------------------------------------------- persistence
def save_report(rep: ValidationReport, out_dir: Path) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, df in rep.tables.items():
        df.to_csv(out_dir / f"{name}.csv", index=False)
    for score, p in rep.placebo.items():
        np.save(out_dir / f"placebo_{score}.npy", p["distribution"])
    if rep.curves is not None:
        rep.curves.to_csv(out_dir / "backtest_curves.csv")
    (out_dir / "summary.json").write_text(json.dumps(rep.summary, indent=2, default=str))
    (out_dir / "settings.yaml").write_text(yaml.safe_dump(rep.settings.model_dump(mode="json"),
                                                          sort_keys=False))
    for fname, fig in figures(rep):
        fig.savefig(out_dir / fname, bbox_inches="tight")
    return out_dir


def load_report(out_dir: Path) -> ValidationReport:
    out_dir = Path(out_dir)
    summary = json.loads((out_dir / "summary.json").read_text())
    settings = ValidationSettings.model_validate(yaml.safe_load((out_dir / "settings.yaml").read_text()))
    rep = ValidationReport(run_id=summary["run_id"], label=summary.get("label", ""),
                           settings=settings, summary=summary)
    dates = {"events": ["date", "tc_q25", "tc_median", "tc_q75"],
             "timing": ["first_alarm", "last_alarm", "peak_conf_date", "tc_q25", "tc_q75", "top_date"],
             "tc_episodes": ["alarm_date", "tc_q25", "tc_q75"]}
    for csv in out_dir.glob("*.csv"):
        if csv.stem == "backtest_curves":
            rep.curves = pd.read_csv(csv, index_col=0, parse_dates=[0])
        else:
            rep.tables[csv.stem] = pd.read_csv(csv, parse_dates=dates.get(csv.stem, False))
    for npy in out_dir.glob("placebo_*.npy"):
        score = npy.stem.split("_", 1)[1]
        row = rep.tables["placebo"].set_index("score").loc[score].to_dict()
        rep.placebo[score] = {**row, "score": score, "distribution": np.load(npy)}
    return rep


def latest_report_dir(base: Path = PROJECT_ROOT / "reports" / "validation") -> Path | None:
    dirs = sorted((d for d in base.glob("*") if (d / "summary.json").exists()),
                  key=lambda d: d.stat().st_mtime)
    return dirs[-1] if dirs else None


# --------------------------------------------------------------------------- figures
def figures(rep: ValidationReport):
    import matplotlib

    if matplotlib.get_backend().lower() not in ("module://matplotlib_inline.backend_inline",):
        matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from lppl import viz

    viz.apply_style()
    t = rep.tables
    out = []

    # calibration
    cal = t["calibration"]
    fig, ax = plt.subplots(figsize=(7.5, 3.6))
    x = np.arange(len(cal))
    ax.bar(x, cal["event_rate"], color=viz.SERIES[0], edgecolor=viz.SURFACE, width=0.7)
    base = t["skill"].set_index("split").loc["all", "base_rate"]
    ax.axhline(base, color=viz.INK, ls="--", lw=1, label=f"base rate {base:.1%}")
    for i, (n, r) in enumerate(zip(cal["n"], cal["event_rate"], strict=True)):
        if np.isfinite(r):
            ax.text(i, 0.004, f"n = {n}", ha="center", va="bottom", fontsize=7, color="white")
    ax.set_xticks(x, cal["band"].astype(str))
    ax.set_ylim(0, max(0.05, float(np.nanmax(cal["event_rate"])) * 1.15))
    ax.set_xlabel("bubble confidence band")
    ax.set_ylabel("share of dates followed by an event")
    ax.set_title("Calibration: does a higher confidence mean more events?")
    ax.legend(loc="upper left")
    out.append(("calibration.png", fig))

    # bootstrap CIs
    bs = t["bootstrap"]
    d = bs[bs["metric"].isin(["lppl_minus_base", "lppl_minus_momentum"])]
    fig, ax = plt.subplots(figsize=(7.5, 2.8))
    labels = [f"{r.split}: LPPLS − {'base rate' if 'base' in r.metric else 'momentum'}" for r in d.itertuples()]
    y = np.arange(len(d))
    ax.hlines(y, d["ci_low"], d["ci_high"], color=viz.SERIES[0], lw=4, alpha=0.5)
    ax.scatter(d["point"], y, color=viz.SERIES[0], s=40, zorder=3, edgecolor=viz.SURFACE, linewidth=1.5)
    ax.axvline(0, color=viz.INK, lw=1)
    ax.set_yticks(y, labels)
    ax.invert_yaxis()
    ax.set_xlabel("difference in average precision (95 % block-bootstrap interval)")
    ax.set_title("Is the advantage real? Intervals crossing 0 are not significant")
    out.append(("bootstrap.png", fig))

    # placebo
    fig, axes = plt.subplots(1, len(rep.placebo), figsize=(10, 3), sharey=True)
    axes = np.atleast_1d(axes)
    for ax, (score, p) in zip(axes, rep.placebo.items(), strict=True):
        ax.hist(p["distribution"], bins=40, color=viz.MUTED, edgecolor=viz.SURFACE)
        ax.axvline(p["actual_ap"], color=viz.POS, lw=2)
        ax.set_title(f"{'LPPLS confidence' if score == 'conf' else 'momentum'}: p = {p['p_value']:.3f}", fontsize=10)
        ax.set_xlabel("AP of time-shifted series (grey) vs actual (red)")
    fig.suptitle("Placebo test: would a randomly time-shifted signal do as well?", x=0.01, ha="left",
                 fontweight="bold", fontsize=11)
    fig.tight_layout()
    out.append(("placebo.png", fig))

    # robustness grid (lift); each cell shows the number of events behind it
    g = t["grid"].assign(events=lambda d: (d["n"] * d["base_rate"]).round().astype(int))
    fig, axes = plt.subplots(1, 2, figsize=(10.5, 3.6))
    for ax, split in zip(axes, ("all", "valid"), strict=True):
        sub = g[g["split"] == split]
        piv = sub.pivot(index="horizon", columns="k_sigma", values="lift_lppl")
        nev = sub.pivot(index="horizon", columns="k_sigma", values="events")
        im = ax.imshow(piv.to_numpy(), cmap=viz.DIVERGING, vmin=0.5, vmax=1.5, aspect="auto")
        ax.set_xticks(range(piv.shape[1]), [f"{k}σ" for k in piv.columns])
        ax.set_yticks(range(piv.shape[0]), [f"{h} d" for h in piv.index])
        for i in range(piv.shape[0]):
            for j in range(piv.shape[1]):
                ax.text(j, i, f"{piv.iat[i, j]:.2f}\n{int(nev.iat[i, j])} events", ha="center",
                        va="center", fontsize=8, color=viz.INK)
        ax.set_title("all dates" if split == "all" else "validation (2024-)", fontsize=10)
        ax.set_xlabel("drawdown threshold")
        ax.set_ylabel("horizon (trading days)")
        ax.grid(False)
    fig.colorbar(im, ax=axes, fraction=0.03, label="lift = AP / base rate (1 = random)")
    fig.suptitle("Robustness: LPPLS lift for other event definitions (few events = noisy)",
                 x=0.01, y=1.04, ha="left", fontweight="bold", fontsize=11)
    out.append(("grid.png", fig))

    # walk-forward
    if "walk_forward" in t and not t["walk_forward"].empty:
        wf = t["walk_forward"]
        fig, ax = plt.subplots(figsize=(8.5, 3.4))
        x = np.arange(len(wf))
        cols = [("base_rate_test", "base rate", viz.MUTED),
                ("ap_momentum_test", "momentum", viz.SERIES[1]),
                ("ap_stored_test", "LPPLS as scanned", viz.SERIES[0]),
                ("ap_tuned_test", "LPPLS re-tuned on past", viz.SERIES[2])]
        cols = [c for c in cols if c[0] in wf]
        w = 0.8 / len(cols)
        for i, (c, lab, color) in enumerate(cols):
            ax.bar(x + (i - (len(cols) - 1) / 2) * w, wf[c], width=w, color=color, label=lab, edgecolor=viz.SURFACE)
        ax.set_xticks(x, wf["test_year"].astype(int))
        ax.set_ylabel("average precision (test year)")
        ax.set_title("Walk-forward: each year scored with settings chosen on earlier years only")
        ax.legend(ncol=len(cols), loc="upper left", fontsize=7)
        out.append(("walk_forward.png", fig))

    # null model
    if "null_model" in t and not t["null_model"].empty:
        nm = t["null_model"].groupby("ticker")[["alarm_rate_real", "alarm_rate_null"]].mean()
        fig, ax = plt.subplots(figsize=(8.5, 3.4))
        x = np.arange(len(nm))
        ax.bar(x - 0.2, nm["alarm_rate_real"], 0.4, color=viz.POS, label="real prices", edgecolor=viz.SURFACE)
        ax.bar(x + 0.2, nm["alarm_rate_null"], 0.4, color=viz.MUTED, label="bubble-free simulated prices",
               edgecolor=viz.SURFACE)
        ax.set_xticks(x, nm.index, rotation=0, fontsize=8)
        ax.set_ylabel("share of dates with an alarm")
        ax.set_title("False alarms: how often does the detector fire without any bubble?")
        ax.legend()
        out.append(("null_model.png", fig))

    # backtest: each strategy relative to buy & hold (the curves themselves are nearly equal)
    if rep.curves is not None and "buy_and_hold" in rep.curves:
        fig, ax = plt.subplots(figsize=(9, 3.6))
        names = {"lppl": ("LPPLS de-risk", viz.POS),
                 "momentum": ("momentum de-risk (same alarm rate)", viz.SERIES[1]),
                 "lppl_placebo": ("LPPLS placebo (time-shifted alarms)", viz.MUTED)}
        bh = rep.curves["buy_and_hold"]
        for c, (lab, color) in names.items():
            if c in rep.curves:
                ax.plot(rep.curves.index, 100 * (rep.curves[c] / bh - 1), label=lab, color=color, lw=1.4)
        ax.axhline(0, color=viz.INK, lw=1)
        ax.axvline(pd.Timestamp(rep.settings.split_date), color=viz.MUTED, ls=":", lw=1)
        ax.set_ylabel("% above / below buy & hold")
        ax.set_title(f"Backtest: cut exposure by {rep.settings.backtest.cut:.0%} on alarms "
                     f"({rep.settings.backtest.cost_bps:g} bp costs), relative to buy & hold")
        ax.legend(loc="lower left", fontsize=8)
        out.append(("backtest.png", fig))
    return out


# --------------------------------------------------------------------------- MLflow (optional)
def log_to_mlflow(rep: ValidationReport, out_dir: Path, experiment: str = "lppl-validation") -> str | None:
    try:
        from lppl.tuning import tracking
        cl = tracking.client()
        cl.search_experiments(max_results=1)  # reachability check
    except Exception as exc:  # mlflow missing or server down
        log.warning("MLflow logging skipped: %s", exc)
        return None
    tags = tracking.base_tags("validation", f"validate-{rep.label}-{rep.run_id[:8]}", "local",
                              {"source_run_id": rep.run_id, "source_label": rep.label})
    flat = {k: v for k, v in pd.json_normalize(rep.settings.model_dump(mode="json"), sep=".").iloc[0].items()}
    parent = tracking.start_parent(cl, experiment, tags["study"], tags, flat)
    metrics = {k.replace("[", "_").replace("]", ""): v for k, v in rep.summary.items()
               if isinstance(v, (int, float)) and not isinstance(v, bool)}
    metrics |= {f"verdict.{k}": float(v) for k, v in rep.summary["verdict"].items() if v is not None}
    tracking.log_metrics(cl, parent, metrics)
    for f in sorted(Path(out_dir).iterdir()):
        if f.suffix in (".csv", ".json", ".png", ".yaml"):
            tracking.log_file(cl, parent, f)
    tracking.finish(cl, parent)
    return parent
