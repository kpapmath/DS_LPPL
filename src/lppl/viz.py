"""Matplotlib helpers shared by the notebooks.

Colour roles (validated reference palette, see docs/model_presentation.md):
* prices: neutral primary ink; fitted LPPLS curve: categorical slot 1 (blue)
* positive-bubble confidence: red pole; negative (anti-bubble) confidence: blue pole
* signed confidence heatmap: diverging blue <-> gray <-> red
"""

from __future__ import annotations

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap

INK = "#0b0b0b"
INK_2 = "#52514e"
MUTED = "#898781"
GRID = "#e1e0d9"
AXIS = "#c3c2b7"
SURFACE = "#fcfcfb"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#008300", "#6250d6", "#e34948"]
POS = "#e34948"   # bubble (B < 0)
NEG = "#2a78d6"   # anti-bubble (B > 0)
FIT = "#2a78d6"
GROUP_COLORS = {"index": SERIES[0], "mag7": SERIES[1], "extra": SERIES[2]}

DIVERGING = LinearSegmentedColormap.from_list(
    "lppl_div", ["#104281", "#3987e5", "#9ec5f4", "#f0efec", "#f4a3a2", "#e34948", "#a32b2b"])
SEQUENTIAL = LinearSegmentedColormap.from_list(
    "lppl_seq", ["#cde2fb", "#86b6ef", "#3987e5", "#1c5cab", "#0d366b"])


_SANS_CANDIDATES = ["Helvetica Neue", "Helvetica", "Arial", "Liberation Sans", "DejaVu Sans"]


def _installed_sans() -> list[str]:
    """Preferred UI sans fonts that exist on this machine (DejaVu Sans ships with matplotlib)."""
    from matplotlib import font_manager

    installed = {f.name for f in font_manager.fontManager.ttflist}
    return [n for n in _SANS_CANDIDATES if n in installed] or ["DejaVu Sans"]


def plain_log_axis(ax) -> None:
    """Show log-scale ticks as plain numbers (300, not 3×10²)."""
    from matplotlib.ticker import FuncFormatter

    fmt = FuncFormatter(lambda v, _: f"{v:,.0f}" if v >= 10 else f"{v:g}")
    ax.yaxis.set_major_formatter(fmt)
    ax.yaxis.set_minor_formatter(fmt)


def apply_style() -> None:
    mpl.rcParams.update({
        "figure.facecolor": SURFACE, "axes.facecolor": SURFACE, "savefig.facecolor": SURFACE,
        "axes.edgecolor": AXIS, "axes.labelcolor": INK_2, "axes.titlecolor": INK,
        "axes.titlesize": 11, "axes.titleweight": "bold", "axes.titlelocation": "left",
        "axes.labelsize": 9, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.6,
        "axes.prop_cycle": mpl.cycler(color=SERIES),
        "xtick.color": MUTED, "ytick.color": MUTED, "xtick.labelsize": 8, "ytick.labelsize": 8,
        "legend.frameon": False, "legend.fontsize": 8, "lines.linewidth": 1.4,
        # Only fonts that are actually installed: matplotlib logs a "findfont" warning for
        # every missing family on every text element (CSS names like system-ui don't exist).
        "font.family": "sans-serif",
        "font.sans-serif": _installed_sans(),
        "figure.dpi": 110,
    })


def price_and_confidence(px: pd.Series, ind: pd.DataFrame, title: str, axes=None, log=True):
    """Two stacked panels: price (top) and positive / negative confidence (bottom)."""
    if axes is None:
        _, axes = plt.subplots(2, 1, figsize=(10, 4.6), sharex=True,
                               gridspec_kw={"height_ratios": [2.2, 1], "hspace": 0.08})
    ax_p, ax_c = axes
    ax_p.plot(px.index, px.to_numpy(), color=INK, lw=1.1)
    if log:
        ax_p.set_yscale("log")
        plain_log_axis(ax_p)
    ax_p.set_title(title)
    ax_p.set_ylabel("adj. close")
    if not ind.empty:
        d = ind.sort_values("date")
        ax_c.fill_between(d["date"], 0, d["pos_conf"], color=POS, alpha=0.35, lw=0, step="post")
        ax_c.step(d["date"], d["pos_conf"], where="post", color=POS, lw=1.2, label="bubble (B<0)")
        ax_c.fill_between(d["date"], 0, -d["neg_conf"], color=NEG, alpha=0.35, lw=0, step="post")
        ax_c.step(d["date"], -d["neg_conf"], where="post", color=NEG, lw=1.2,
                  label="anti-bubble (B>0)")
        # shade the price panel where the positive confidence is high
        hot = d[d["pos_conf"] >= 0.3]
        for day in hot["date"]:
            ax_p.axvspan(day, day + pd.Timedelta(days=14), color=POS, alpha=0.08, lw=0)
    ax_c.axhline(0, color=AXIS, lw=0.8)
    ax_c.set_ylim(-1, 1)
    ax_c.set_yticks([-1, -0.5, 0, 0.5, 1])
    ax_c.set_yticklabels(["1", ".5", "0", ".5", "1"])
    ax_c.set_ylabel("confidence")
    ax_c.legend(loc="upper left", ncol=2)
    return axes


def signed_heatmap(ind: pd.DataFrame, ax=None, freq: str = "MS"):
    """Ticker x month heatmap of max(pos_conf) - max(neg_conf) per month."""
    d = ind.copy()
    d["month"] = d["date"].dt.to_period("M").dt.to_timestamp()
    g = d.groupby(["ticker", "month"])[["pos_conf", "neg_conf"]].max()
    s = (g["pos_conf"] - g["neg_conf"]).unstack("month")
    if ax is None:
        _, ax = plt.subplots(figsize=(11, 0.32 * len(s) + 1.2))
    im = ax.imshow(s.to_numpy(), aspect="auto", cmap=DIVERGING, vmin=-1, vmax=1,
                   interpolation="nearest")
    ax.set_yticks(range(len(s)), s.index)
    cols = s.columns
    step = max(1, len(cols) // 12)
    ax.set_xticks(range(0, len(cols), step), [c.strftime("%Y-%m") for c in cols[::step]],
                  rotation=45, ha="right")
    ax.grid(False)
    cb = plt.colorbar(im, ax=ax, fraction=0.025, pad=0.01)
    cb.set_label("anti-bubble  ←  confidence  →  bubble", fontsize=8, color=INK_2)
    cb.outline.set_visible(False)
    return ax


def fit_overlay(dates, log_price, fit, ax=None, title=None, extend_days: int = 120):
    """Observed log-price, fitted LPPLS curve (extended towards tc) and tc marker."""
    from lppl.model.lppls import dates_to_years, years_to_date
    if ax is None:
        _, ax = plt.subplots(figsize=(10, 3.6))
    dates = pd.DatetimeIndex(dates)
    ax.plot(dates, log_price, color=INK, lw=1.0, label="ln price")
    tc_date = years_to_date(fit.tc)
    end = min(tc_date - pd.Timedelta(days=1), dates[-1] + pd.Timedelta(days=extend_days))
    grid = pd.date_range(dates[0], max(end, dates[-1]), freq="D")
    ax.plot(grid, fit.predict(dates_to_years(grid)), color=FIT, lw=2, label="LPPLS fit")
    top = ax.get_xaxis_transform()  # x in data, y in axes fraction
    ax.axvline(dates[-1], color=MUTED, lw=0.8, ls=":")
    ax.text(dates[-1], 0.97, "t2 ", transform=top, color=INK_2, fontsize=8, ha="right", va="top")
    ax.axvline(tc_date, color=POS if fit.B < 0 else NEG, lw=1.2, ls="--")
    ax.text(tc_date, 0.97, f" tc ≈ {tc_date.date()}", transform=top, color=INK_2, fontsize=8,
            ha="left", va="top")
    ax.set_title(title or "LPPLS fit")
    ax.legend(loc="upper left")
    return ax


def direct_labels(ax, ends: dict[str, float], min_gap: float = 0.045, log: bool = True) -> None:
    """Label line ends at the right edge, nudged apart so labels never overlap.

    ``ends`` maps label -> last y value; ``min_gap`` is in axes-height fraction.
    """
    if not ends:
        return
    lo, hi = ax.get_ylim()
    tf = (np.log10 if log else (lambda v: v))
    to_frac = lambda v: (tf(v) - tf(lo)) / (tf(hi) - tf(lo))  # noqa: E731
    items = sorted(((to_frac(v), k) for k, v in ends.items()))
    pos = []
    for f, k in items:
        f = max(f, pos[-1][0] + min_gap) if pos else f
        pos.append((f, k))
    overflow = pos[-1][0] - 0.98
    if overflow > 0:
        pos = [(f - overflow, k) for f, k in pos]
    for f, k in pos:
        ax.text(1.01, f, k, transform=ax.transAxes, fontsize=7, color=INK_2, va="center")
