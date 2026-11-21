"""Qualification filters for LPPLS fits (Sornette et al. 2015, Financial Crisis Observatory).

A fit is *qualified* when its parameters fall in the empirically "stylised" ranges
seen before historical bubbles. Only qualified fits count towards the confidence
indicator.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from lppl.model.lppls import DAYS_PER_YEAR, LPPLSFit


@dataclass(frozen=True)
class FilterConfig:
    m_min: float = 0.1
    m_max: float = 0.9
    omega_min: float = 2.0
    omega_max: float = 25.0
    # tc must lie in [t2 - tc_before*dt, t2 + tc_after*dt], dt = t2 - t1
    tc_before: float = 0.05
    tc_after: float = 0.2
    # Reject tc within this many calendar days after t2. The GA searches tc from t2 + 1 day,
    # so such fits sit on the search bound: a degenerate "already critical" solution
    # (usually together with m -> 0), not an interior optimum.
    tc_margin_days: float = 3.0
    damping_min: float = 0.5
    oscillations_min: float = 2.5
    # Optional cap on the RMSE of the log-price fit (None disables it)
    max_rmse: float | None = None


def check(fit: LPPLSFit, cfg: FilterConfig | None = None) -> dict[str, bool]:
    """Return every individual condition; ``qualified`` is their conjunction."""
    cfg = cfg or FilterConfig()
    dt = fit.t2 - fit.t1
    n_osc = fit.n_oscillations
    conds = {
        "m": cfg.m_min <= fit.m <= cfg.m_max,
        "omega": cfg.omega_min <= fit.omega <= cfg.omega_max,
        "tc": fit.t2 - cfg.tc_before * dt <= fit.tc <= fit.t2 + cfg.tc_after * dt,
        "tc_margin": fit.tc - fit.t2 > cfg.tc_margin_days / DAYS_PER_YEAR,
        "damping": fit.damping >= cfg.damping_min,
        "oscillations": bool(np.isfinite(n_osc) and n_osc >= cfg.oscillations_min),
        "finite": bool(np.isfinite(fit.sse)),
        "rmse": cfg.max_rmse is None or fit.rmse <= cfg.max_rmse,
    }
    conds["qualified"] = all(conds.values())
    return conds


def is_qualified(fit: LPPLSFit, cfg: FilterConfig | None = None) -> bool:
    return check(fit, cfg)["qualified"]


def derived_arrays(*, t1, t2, tc, m, omega, B, C1, C2, sse, n_obs) -> dict[str, np.ndarray]:
    """Damping, number of oscillations and RMSE for many fits, with the same edge-case
    semantics as the :class:`LPPLSFit` properties."""
    t1, t2, tc, m, omega, B, C1, C2, sse, n_obs = (
        np.asarray(a, dtype=float) for a in (t1, t2, tc, m, omega, B, C1, C2, sse, n_obs))
    C = np.hypot(C1, C2)
    denom = omega * C
    with np.errstate(divide="ignore", invalid="ignore"):
        damping = np.where(denom == 0, np.inf, m * np.abs(B) / np.where(denom == 0, 1.0, denom))
        after = tc > t2
        n_osc = np.where(after, omega / np.pi * np.log(np.where(after, (tc - t1) / (tc - t2), 1.0)), np.nan)
        rmse = np.sqrt(sse / np.maximum(n_obs, 1))
    return {"damping": damping, "n_osc": n_osc, "rmse": rmse}


def qualify_arrays(*, t1, t2, tc, m, omega, sse, B=None, C1=None, C2=None, n_obs=None,
                   cfg: FilterConfig | None = None, derived: dict | None = None) -> np.ndarray:
    """Vectorised :func:`check` over many stored fits (1-D arrays, time in decimal years).

    Row for row identical to ``check(fit, cfg)["qualified"]`` (tested). Used by
    ``requalify`` and by the tuning studies, where it has to be fast. ``derived`` (the
    output of :func:`derived_arrays`) can be passed to skip recomputing it."""
    cfg = cfg or FilterConfig()
    t1, t2, tc, m, omega, sse = (np.asarray(a, dtype=float) for a in (t1, t2, tc, m, omega, sse))
    d = derived if derived is not None else derived_arrays(
        t1=t1, t2=t2, tc=tc, m=m, omega=omega, B=B, C1=C1, C2=C2, sse=sse, n_obs=n_obs)
    dt = t2 - t1
    with np.errstate(invalid="ignore"):
        ok = (m >= cfg.m_min) & (m <= cfg.m_max)
        ok &= (omega >= cfg.omega_min) & (omega <= cfg.omega_max)
        ok &= (tc >= t2 - cfg.tc_before * dt) & (tc <= t2 + cfg.tc_after * dt)
        ok &= (tc - t2) > cfg.tc_margin_days / DAYS_PER_YEAR
        ok &= d["damping"] >= cfg.damping_min
        ok &= np.isfinite(d["n_osc"]) & (d["n_osc"] >= cfg.oscillations_min)
        ok &= np.isfinite(sse)
        if cfg.max_rmse is not None:
            ok &= d["rmse"] <= cfg.max_rmse
    return ok
