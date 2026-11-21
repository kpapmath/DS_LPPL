"""Core LPPLS model (Filimonov & Sornette 2013 linearised form).

    ln p(t) = A + B f + C1 g + C2 h
    f = (tc - t)^m,  g = f cos(omega ln(tc - t)),  h = f sin(omega ln(tc - t))

For fixed nonlinear parameters (tc, m, omega) the linear parameters (A, B, C1, C2)
are the ordinary-least-squares solution of a 4-column regression. Every function
here is vectorised over a population of nonlinear parameter vectors so that a
genetic algorithm can evaluate a whole generation in one call.

Time is measured in decimal years (see :func:`dates_to_years`).
"""

from __future__ import annotations

from dataclasses import asdict, dataclass

import numpy as np
import pandas as pd

DAYS_PER_YEAR = 365.25
_EPOCH = pd.Timestamp("1970-01-01")


# --------------------------------------------------------------------------- time
def dates_to_years(dates) -> np.ndarray:
    """Convert dates to decimal years (1970.0 == 1970-01-01, 365.25-day years)."""
    idx = pd.DatetimeIndex(pd.to_datetime(dates))
    if idx.tz is not None:
        idx = idx.tz_localize(None)
    days = (idx - _EPOCH) / pd.Timedelta(days=1)
    return 1970.0 + np.asarray(days, dtype=float) / DAYS_PER_YEAR


def years_to_date(years: float) -> pd.Timestamp:
    """Inverse of :func:`dates_to_years` (rounded to the nearest calendar day)."""
    return _EPOCH + pd.Timedelta(days=round((float(years) - 1970.0) * DAYS_PER_YEAR))


# --------------------------------------------------------------------------- model
def lppls(t, tc, m, omega, A, B, C1, C2) -> np.ndarray:
    """Evaluate the LPPLS log-price. Returns NaN where t >= tc."""
    t = np.asarray(t, dtype=float)
    tau = tc - t
    valid = tau > 0
    tau = np.where(valid, tau, 1.0)
    f = tau**m
    lt = np.log(tau)
    out = A + B * f + C1 * f * np.cos(omega * lt) + C2 * f * np.sin(omega * lt)
    return np.where(valid, out, np.nan)


def _basis(t: np.ndarray, tc, m, omega):
    """Return the regressors f, g, h (each (P, N)) and a (P,) validity mask."""
    tc = np.atleast_1d(np.asarray(tc, dtype=float))[:, None]
    m = np.atleast_1d(np.asarray(m, dtype=float))[:, None]
    omega = np.atleast_1d(np.asarray(omega, dtype=float))[:, None]
    tau = tc - t[None, :]
    pos = tau > 0
    valid = np.all(pos, axis=1)
    lt = np.log(np.where(pos, tau, 1.0))
    f = np.exp(m * lt)  # == tau**m, faster
    wl = omega * lt
    return f, f * np.cos(wl), f * np.sin(wl), valid


def solve_linear(t, y, tc, m, omega) -> tuple[np.ndarray, np.ndarray]:
    """Slave the linear parameters to (tc, m, omega).

    Parameters may be scalars or 1-D arrays of length P.

    Returns
    -------
    coef : (P, 4) array of [A, B, C1, C2]
    sse  : (P,) sum of squared residuals; ``inf`` for invalid / degenerate candidates.
    """
    t = np.asarray(t, dtype=float)
    y = np.asarray(y, dtype=float)
    f, g, h, valid = _basis(t, tc, m, omega)
    P, N = f.shape
    cols = (f, g, h)

    # Normal equations X'X b = X'y with X = [1, f, g, h], built from row sums
    # (much faster than a (P, N, 4) tensor product).
    XtX = np.empty((P, 4, 4))
    Xty = np.empty((P, 4))
    XtX[:, 0, 0] = N
    Xty[:, 0] = y.sum()
    for i, a in enumerate(cols, start=1):
        XtX[:, 0, i] = XtX[:, i, 0] = a.sum(axis=1)
        Xty[:, i] = a @ y
        for j in range(i, 4):
            XtX[:, i, j] = XtX[:, j, i] = np.einsum("pn,pn->p", a, cols[j - 1])

    # Symmetric diagonal scaling greatly improves conditioning.
    d = np.sqrt(np.clip(np.diagonal(XtX, axis1=1, axis2=2), 1e-300, None))
    S = XtX / (d[:, :, None] * d[:, None, :]) + 1e-12 * np.eye(4)[None]
    r = Xty / d
    try:
        beta = np.linalg.solve(S, r[..., None])[..., 0]
    except np.linalg.LinAlgError:
        beta = np.einsum("pij,pj->pi", np.linalg.pinv(S), r)
    coef = beta / d

    resid = y[None, :] - coef[:, :1] - coef[:, 1:2] * f - coef[:, 2:3] * g - coef[:, 3:4] * h
    sse = np.einsum("pn,pn->p", resid, resid)
    ok = valid & np.isfinite(sse) & np.all(np.isfinite(coef), axis=1)
    return coef, np.where(ok, sse, np.inf)


# --------------------------------------------------------------------------- result
@dataclass(frozen=True)
class LPPLSFit:
    """A calibrated LPPLS window."""

    t1: float
    t2: float
    tc: float
    m: float
    omega: float
    A: float
    B: float
    C1: float
    C2: float
    sse: float
    n_obs: int

    @property
    def C(self) -> float:
        return float(np.hypot(self.C1, self.C2))

    @property
    def phi(self) -> float:
        """Phase in the alternative parameterisation C cos(omega ln(tc-t) - phi)."""
        return float(np.arctan2(self.C2, self.C1))

    @property
    def rmse(self) -> float:
        return float(np.sqrt(self.sse / max(self.n_obs, 1)))

    @property
    def damping(self) -> float:
        """D = m|B| / (omega|C|). Large D means the crash hazard stays non-negative."""
        denom = self.omega * self.C
        return float(np.inf if denom == 0 else self.m * abs(self.B) / denom)

    @property
    def n_oscillations(self) -> float:
        """O = (omega/pi) ln((tc - t1)/(tc - t2)): number of half-periods in the window."""
        if self.tc <= self.t2:
            return float("nan")
        return float(self.omega / np.pi * np.log((self.tc - self.t1) / (self.tc - self.t2)))

    @property
    def sign(self) -> int:
        """+1 positive (upward, B<0) bubble, -1 negative (anti-)bubble, 0 undefined."""
        return int(-np.sign(self.B))

    def predict(self, t) -> np.ndarray:
        return lppls(t, self.tc, self.m, self.omega, self.A, self.B, self.C1, self.C2)

    def to_dict(self) -> dict:
        d = asdict(self)
        d.update(
            C=self.C,
            rmse=self.rmse,
            damping=self.damping,
            n_oscillations=self.n_oscillations,
            sign=self.sign,
        )
        return d


def make_fit(t, y, tc: float, m: float, omega: float) -> LPPLSFit:
    """Build an :class:`LPPLSFit` from nonlinear parameters by solving the linear part."""
    t = np.asarray(t, dtype=float)
    coef, sse = solve_linear(t, y, tc, m, omega)
    A, B, C1, C2 = (float(v) for v in coef[0])
    return LPPLSFit(
        t1=float(t[0]), t2=float(t[-1]), tc=float(tc), m=float(m), omega=float(omega),
        A=A, B=B, C1=C1, C2=C2, sse=float(sse[0]), n_obs=int(t.size),
    )
