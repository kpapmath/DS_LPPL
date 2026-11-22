"""Combined walk-forward model: does LPPLS add information beyond market features?

Nested logistic regressions (L2-regularised), refitted every year on all earlier data only
(with an embargo equal to the label's look-ahead) and scored on the following year:

================== ================================================================
momentum           126-day momentum (with hinge terms, so the U-shape is captured)
market             all market features (momentum 21/126/252, volatility ratio, drawdown)
market+lppls       market + long-scale LPPLS confidence (+ its interaction with momentum)
market+refined     market+lppls + the refined LPPLS signals (lppl.signals)
================== ================================================================

The primary score is the out-of-sample **log-loss** (a proper scoring rule); average precision,
Brier score and the precision of the top-``alarm_share`` predictions are secondary. Differences
between models get month-block bootstrap intervals; the pre-registered family (LPPLS vs market and
refined vs LPPLS, per label) is Holm-adjusted. See docs/combined_model.md.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd
import yaml
from pydantic import BaseModel
from scipy.optimize import minimize

from lppl.config import PROJECT_ROOT
from lppl.labels import holm
from lppl.signals import LPPLS_SIGNALS, MARKET_FEATURES, SignalSettings
from lppl.tuning.detection import average_precision

log = logging.getLogger(__name__)
DEFAULT_COMBINED_CONFIG = PROJECT_ROOT / "config" / "combined.yaml"
REFINED = [s for s in LPPLS_SIGNALS if s != "conf"]
PRIMARY_PAIRS = [("market+lppls", "market"), ("market+refined", "market+lppls")]
ALL_PAIRS = [("market", "momentum"), *PRIMARY_PAIRS, ("market+refined", "market")]


class CombinedSettings(BaseModel):
    seed: int = 0
    labels: list[str] = ["drawdown_63", "drawdown_126", "major_top_126"]
    embargo_days: dict[str, int] = {"drawdown_63": 63, "drawdown_126": 126, "major_top_126": 252,
                                    "rally_end_126": 126}
    first_test_year: int = 1995
    min_train: int = 1000            # labelled rows needed before a fold is scored
    l2: float = 1.0
    momentum_knots: list[float] = [0.1, 0.25, 0.5, 0.75, 0.9]
    alarm_share: float = 0.02
    bootstrap: int = 1000
    placebo: int = 1000              # standalone signals: time-shift placebo replications
    k_sigma: float = 1.5
    top_lookback: int = 252
    signals: SignalSettings = SignalSettings()


def load_combined_config(path: str | Path | None = None) -> CombinedSettings:
    with open(Path(path) if path else DEFAULT_COMBINED_CONFIG) as fh:
        return CombinedSettings.model_validate(yaml.safe_load(fh) or {})


# --------------------------------------------------------------------------- design
def model_features(name: str, has_short: bool) -> list[str]:
    refined = [s for s in REFINED if has_short or s not in ("conf_short", "conf_agree")]
    return {"momentum": ["mom_126"], "market": MARKET_FEATURES,
            "market+lppls": MARKET_FEATURES + ["conf"],
            "market+refined": MARKET_FEATURES + ["conf"] + refined}[name]


MODELS = ["momentum", "market", "market+lppls", "market+refined"]


@dataclass
class Design:
    """Column transforms fitted on the training rows only (knots, means, scales)."""
    features: list[str]
    knots: np.ndarray
    mean: np.ndarray | None = None
    sd: np.ndarray | None = None

    @classmethod
    def fit(cls, df: pd.DataFrame, features: list[str], knot_q) -> "Design":
        d = cls(features, np.quantile(df["mom_126"], knot_q))
        X = d._raw(df)
        d.mean, d.sd = X.mean(0), X.std(0)
        d.sd[d.sd == 0] = 1.0
        return d

    def _raw(self, df: pd.DataFrame) -> np.ndarray:
        cols = [df[f].to_numpy(float) for f in self.features]
        mom = df["mom_126"].to_numpy(float)
        cols += [np.maximum(mom - k, 0) for k in self.knots]          # momentum hinges
        if "conf" in self.features:                                    # conf x positive momentum
            cols.append(df["conf"].to_numpy(float) * np.maximum(mom, 0))
        return np.column_stack(cols)

    def transform(self, df: pd.DataFrame) -> np.ndarray:
        return (self._raw(df) - self.mean) / self.sd


def fit_logistic(X: np.ndarray, y: np.ndarray, l2: float) -> np.ndarray:
    """Intercept + L2-penalised slopes (penalty 0.5 * l2 * ||w||^2 on the summed log-loss)."""
    Xi = np.column_stack([np.ones(len(X)), X])
    pen = np.r_[0.0, np.full(X.shape[1], l2)]

    def f(b):
        z = Xi @ b
        return float(np.sum(np.logaddexp(0, z) - y * z) + 0.5 * np.sum(pen * b * b))

    def g(b):
        return Xi.T @ (1 / (1 + np.exp(-(Xi @ b))) - y) + pen * b

    b0 = np.zeros(Xi.shape[1])
    b0[0] = np.log(max(y.mean(), 1e-6) / max(1 - y.mean(), 1e-6))
    return minimize(f, b0, jac=g, method="L-BFGS-B").x


def predict(b: np.ndarray, X: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-(b[0] + X @ b[1:])))


def log_loss(y, p) -> np.ndarray:
    p = np.clip(p, 1e-9, 1 - 1e-9)
    return -(y * np.log(p) + (1 - y) * np.log(1 - p))


# --------------------------------------------------------------------------- walk-forward
def walk_forward(feat: pd.DataFrame, label: str, s: CombinedSettings) -> pd.DataFrame:
    """Out-of-sample predictions of every model, one row per test-year date."""
    has_short = feat["conf_short"].notna().any()
    d = feat.dropna(subset=MARKET_FEATURES + [label]).copy()
    for c in LPPLS_SIGNALS:
        d[c] = d[c].fillna(0.0)
    d["year"] = d["date"].dt.year
    out = []
    for year in sorted(d["year"].unique()):
        if year < s.first_test_year:
            continue
        cut = pd.Timestamp(f"{year}-01-01") - pd.offsets.BDay(s.embargo_days.get(label, 252))
        tr, te = d[d["date"] < cut], d[d["year"] == year]
        if len(tr) < s.min_train or tr[label].sum() < 10 or te.empty:
            continue
        rows = te[["ticker", "date", "year", label]].rename(columns={label: "y"}).copy()
        for m in MODELS:
            des = Design.fit(tr, model_features(m, has_short), s.momentum_knots)
            b = fit_logistic(des.transform(tr), tr[label].to_numpy(float), s.l2)
            rows[m] = predict(b, des.transform(te))
            thr = np.quantile(predict(b, des.transform(tr)), 1 - s.alarm_share)
            rows[f"{m}__alarm"] = rows[m] >= thr
        out.append(rows)
    return pd.concat(out, ignore_index=True) if out else pd.DataFrame()


def scores(y: np.ndarray, p: np.ndarray, alarm: np.ndarray) -> dict[str, float]:
    return {"log_loss": float(log_loss(y, p).mean()), "brier": float(np.mean((p - y) ** 2)),
            "ap": average_precision(y, p), "alarm_rate": float(alarm.mean()),
            "precision_at_alarm": float(y[alarm].mean()) if alarm.any() else np.nan}


def compare(oos: pd.DataFrame, s: CombinedSettings, label: str) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Per-model scores and pairwise improvements with month-block bootstrap intervals."""
    y = oos["y"].to_numpy(float)
    met = pd.DataFrame([{"label": label, "model": m, "n": len(y), "base_rate": y.mean(),
                         **scores(y, oos[m].to_numpy(), oos[f"{m}__alarm"].to_numpy())} for m in MODELS])
    months = oos["date"].dt.to_period("M").to_numpy()
    blocks = [np.flatnonzero(months == mo) for mo in pd.unique(months)]
    ll = {m: log_loss(y, oos[m].to_numpy()) for m in MODELS}
    rng = np.random.default_rng(s.seed)
    draws = [np.concatenate([blocks[i] for i in rng.integers(0, len(blocks), len(blocks))])
             for _ in range(s.bootstrap)]
    rows = []
    for a, b in ALL_PAIRS:
        gain = ll[b] - ll[a]                          # > 0: model a has the lower log-loss
        pa, pb = oos[a].to_numpy(), oos[b].to_numpy()
        bs_ll = np.array([gain[i].mean() for i in draws])
        bs_ap = np.array([average_precision(y[i], pa[i]) - average_precision(y[i], pb[i]) for i in draws])
        d_ap = average_precision(y, pa) - average_precision(y, pb)
        rows.append({"label": label, "model": a, "versus": b, "primary": (a, b) in PRIMARY_PAIRS,
                     "log_loss_gain": float(gain.mean()),
                     "log_loss_gain_pct": float(100 * gain.mean() / ll[b].mean()),
                     "ll_ci_low": float(np.quantile(bs_ll, 0.025)), "ll_ci_high": float(np.quantile(bs_ll, 0.975)),
                     "p_value": float((1 + np.sum(bs_ll <= 0)) / (1 + len(bs_ll))),
                     "ap_gain": float(d_ap), "ap_ci_low": float(np.nanquantile(bs_ap, 0.025)),
                     "ap_ci_high": float(np.nanquantile(bs_ap, 0.975))})
    return met, pd.DataFrame(rows)


def per_year(oos: pd.DataFrame, label: str) -> pd.DataFrame:
    rows = []
    for year, g in oos.groupby("year"):
        y = g["y"].to_numpy(float)
        r = {"label": label, "year": int(year), "n": len(g), "events": int(y.sum())}
        for m in MODELS:
            r[f"ll_{m}"] = float(log_loss(y, g[m].to_numpy()).mean())
            r[f"ap_{m}"] = average_precision(y, g[m].to_numpy())
        rows.append(r)
    return pd.DataFrame(rows)


def coefficients(feat: pd.DataFrame, label: str, s: CombinedSettings) -> pd.DataFrame:
    """Full-sample fit of the largest model on standardised columns (descriptive only)."""
    has_short = feat["conf_short"].notna().any()
    d = feat.dropna(subset=MARKET_FEATURES + [label]).copy()
    for c in LPPLS_SIGNALS:
        d[c] = d[c].fillna(0.0)
    feats = model_features("market+refined", has_short)
    des = Design.fit(d, feats, s.momentum_knots)
    b = fit_logistic(des.transform(d), d[label].to_numpy(float), s.l2)
    names = ["intercept", *feats, *[f"mom_126_hinge_q{q:g}" for q in s.momentum_knots], "conf_x_mom_pos"]
    return pd.DataFrame({"label": label, "term": names, "coef_per_sd": b})


def standalone(feat: pd.DataFrame, label: str, s: CombinedSettings) -> pd.DataFrame:
    """Each LPPLS signal on its own: AP, lift and the time-shift placebo of lppl.validation."""
    from lppl.validation import placebo_test

    d = feat.dropna(subset=["mom_126", label]).copy()
    d = d.rename(columns={label: "label", "mom_126": "momentum"})
    base = float(d["label"].mean())
    rows = []
    for sig in LPPLS_SIGNALS + ["momentum"]:
        if d[sig].isna().all():
            continue
        dd = d.assign(score=d[sig].fillna(0.0))
        pl = placebo_test(dd, s.placebo, 13, s.seed, "score")
        rows.append({"label": label, "signal": sig, "base_rate": base, "ap": pl["actual_ap"],
                     "lift": pl["actual_ap"] / base, "placebo_mean": pl["placebo_mean"],
                     "p_placebo": pl["p_value"], "share_nonzero": float((dd["score"] != 0).mean())})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- report
@dataclass
class CombinedReport:
    run_id: str
    label: str
    settings: CombinedSettings
    tables: dict[str, pd.DataFrame] = field(default_factory=dict)
    summary: dict[str, Any] = field(default_factory=dict)


def run_combined(engine, run_id: str, label: str = "", short_run_id: str | None = None,
                 settings: CombinedSettings | None = None) -> CombinedReport:
    from lppl.data import db
    from lppl.signals import build_features

    s = settings or load_combined_config()
    prices = db.load_all_prices(engine, "adj_close")
    fits = db.load_fits(engine, run_id)
    short = db.load_indicator(engine, short_run_id) if short_run_id else None
    log.info("features from %d fits (short scale: %s)", len(fits), short_run_id or "none")
    feat = build_features(fits, prices, s.signals, short, s.labels, s.k_sigma, s.top_lookback)
    rep = CombinedReport(run_id, label, s)
    rep.tables["features"] = feat
    oos_all, met, inc, yr, coef, solo = [], [], [], [], [], []
    for lab in s.labels:
        oos = walk_forward(feat, lab, s)
        if oos.empty:
            log.warning("no walk-forward folds for %s", lab)
            continue
        m, i = compare(oos, s, lab)
        oos_all.append(oos.assign(label=lab)); met.append(m); inc.append(i)
        yr.append(per_year(oos, lab)); coef.append(coefficients(feat, lab, s))
        solo.append(standalone(feat, lab, s))
    rep.tables.update(oos=pd.concat(oos_all), metrics=pd.concat(met), increments=pd.concat(inc),
                      per_year=pd.concat(yr), coefficients=pd.concat(coef), standalone=pd.concat(solo))
    inc = rep.tables["increments"]
    prim = inc["primary"].to_numpy()
    inc["p_holm"] = np.nan
    inc.loc[prim, "p_holm"] = holm(inc.loc[prim, "p_value"].to_numpy())
    rep.summary = summarize(rep, short_run_id)
    return rep


def summarize(rep: CombinedReport, short_run_id: str | None) -> dict[str, Any]:
    t = rep.tables
    out: dict[str, Any] = {"run_id": rep.run_id, "label": rep.label, "short_run_id": short_run_id,
                           "first_test_year": rep.settings.first_test_year,
                           "oos_years": sorted(int(y) for y in t["oos"]["year"].unique())}
    for _, r in t["metrics"].iterrows():
        for c in ("log_loss", "ap", "brier", "precision_at_alarm", "base_rate"):
            out[f"{r['label']}.{r['model']}.{c}"] = float(r[c])
    verdict = {}
    for _, r in t["increments"].iterrows():
        key = f"{r['label']}.{r['model']}_vs_{r['versus']}"
        out[f"{key}.log_loss_gain_pct"] = float(r["log_loss_gain_pct"])
        out[f"{key}.p_value"] = float(r["p_value"])
        if r["primary"]:
            out[f"{key}.p_holm"] = float(r["p_holm"])
            verdict[f"{r['label']}: {r['model']} beats {r['versus']}"] = bool(r["p_holm"] < 0.05)
    out["verdict"] = verdict
    return out


def save_report(rep: CombinedReport, out_dir: Path) -> Path:
    out_dir = Path(out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    for name, df in rep.tables.items():
        df.to_csv(out_dir / f"{name}.csv", index=False)
    (out_dir / "summary.json").write_text(json.dumps(rep.summary, indent=2, default=str))
    (out_dir / "settings.yaml").write_text(yaml.safe_dump(rep.settings.model_dump(mode="json"), sort_keys=False))
    for fname, fig in figures(rep):
        fig.savefig(out_dir / fname, bbox_inches="tight")
    return out_dir


def figures(rep: CombinedReport):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    out = []
    yr = rep.tables["per_year"]
    labels = list(yr["label"].unique())
    fig, axes = plt.subplots(len(labels), 1, figsize=(10, 2.6 * len(labels)), sharex=True, squeeze=False)
    for ax, lab in zip(axes[:, 0], labels):
        g = yr[yr["label"] == lab]
        gain_l = 100 * (g["ll_market"] - g["ll_market+lppls"]) / g["ll_market"]
        gain_r = 100 * (g["ll_market"] - g["ll_market+refined"]) / g["ll_market"]
        x = g["year"].to_numpy()
        ax.bar(x - 0.2, gain_l, 0.4, label="+ LPPLS confidence", color="#4C72B0")
        ax.bar(x + 0.2, gain_r, 0.4, label="+ refined LPPLS", color="#DD8452")
        ax.axhline(0, color="0.4", lw=0.8)
        ax.set_ylabel("log-loss gain %")
        ax.set_title(f"{lab}: out-of-sample improvement over the market-only model", fontsize=10)
    axes[0, 0].legend(fontsize=8, frameon=False)
    fig.tight_layout()
    out.append(("per_year_gain.png", fig))

    oos = rep.tables["oos"]
    fig, axes = plt.subplots(1, len(labels), figsize=(4 * len(labels), 3.6), squeeze=False)
    for ax, lab in zip(axes[0], labels):
        g = oos[oos["label"] == lab]
        for m, c in (("market", "0.5"), ("market+refined", "#DD8452")):
            q = pd.qcut(g[m], 10, duplicates="drop")
            cal = g.groupby(q, observed=True).agg(p=(m, "mean"), y=("y", "mean"))
            ax.plot(cal["p"], cal["y"], "o-", ms=4, color=c, label=m)
        lim = max(oos[["market", "market+refined"]].quantile(0.99).max(), g["y"].mean() * 2)
        ax.plot([0, lim], [0, lim], ":", color="0.6")
        ax.set_title(lab, fontsize=10); ax.set_xlabel("predicted probability"); ax.set_ylabel("event rate")
    axes[0, 0].legend(fontsize=8, frameon=False)
    fig.tight_layout()
    out.append(("calibration.png", fig))
    return out


def load_report(out_dir: Path) -> CombinedReport:
    out_dir = Path(out_dir)
    summary = json.loads((out_dir / "summary.json").read_text())
    s = CombinedSettings.model_validate(yaml.safe_load((out_dir / "settings.yaml").read_text()))
    rep = CombinedReport(summary["run_id"], summary.get("label", ""), s, summary=summary)
    for csv in out_dir.glob("*.csv"):
        rep.tables[csv.stem] = pd.read_csv(csv, parse_dates=["date"] if csv.stem in ("features", "oos") else False)
    return rep


def latest_report_dir(base: Path = PROJECT_ROOT / "reports" / "combined", label: str | None = None) -> Path | None:
    dirs = [d for d in base.glob(f"{label}_*" if label else "*") if (d / "summary.json").exists()]
    return max(dirs, key=lambda d: d.stat().st_mtime) if dirs else None
