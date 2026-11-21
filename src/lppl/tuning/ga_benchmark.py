"""Study 1: tune the genetic algorithm on a fixed benchmark of LPPLS windows.

* The benchmark is a deterministic sample of real windows (ticker, end date, length) from
  the database plus synthetic LPPLS series with known parameters. It is built once,
  cached in ``data/tuning/ga_benchmark.pkl`` and identified by a content hash.
* For each window an expensive **oracle** (large GA with restarts, differential
  evolution, multistart Nelder-Mead) gives the reference SSE: the best value known.
* A trial runs the candidate GA ``seeds_per_window`` times on every window. The metrics are
  the excess SSE over the oracle (%) and the cost in objective *evaluations*. Evaluations
  are used instead of wall-clock time because they don't depend on hardware or on other
  trials running in parallel.
* ``loss = mean_excess_pct + cost_weight * mean_evals / 1000`` is reported after every
  chunk of windows, so ASHA (Ray) or a median pruner (Optuna) can stop bad trials early.
"""

from __future__ import annotations

import hashlib
import pickle
from collections.abc import Callable
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from joblib import Parallel, delayed

from lppl.config import PROJECT_ROOT
from lppl.model.calibrate import SearchBounds, fit_window, make_objective
from lppl.model.lppls import dates_to_years, lppls
from lppl.optim import baseline
from lppl.optim.ga import GAConfig
from lppl.optim.ga import minimize as ga_minimize
from lppl.tuning.settings import BenchmarkSettings, GAStudySettings, ga_config

BENCHMARK_PATH = PROJECT_ROOT / "data" / "tuning" / "ga_benchmark.pkl"


@dataclass
class BenchWindow:
    name: str
    t: np.ndarray
    y: np.ndarray
    ref_sse: float = float("nan")
    ref_x: np.ndarray | None = None
    truth: dict | None = None  # known parameters for synthetic windows


@dataclass
class GABenchmark:
    windows: list[BenchWindow]
    bounds: SearchBounds
    meta: dict = field(default_factory=dict)

    @property
    def digest(self) -> str:
        h = hashlib.sha256()
        for w in self.windows:
            h.update(w.name.encode())
            h.update(np.ascontiguousarray(w.y).tobytes())
        return h.hexdigest()[:16]


# --------------------------------------------------------------------------- build
def sample_real_windows(prices: pd.DataFrame, s: BenchmarkSettings) -> list[BenchWindow]:
    rng = np.random.default_rng(s.seed)
    tickers = sorted(prices.columns)
    out = []
    while len(out) < s.n_real:
        ticker = tickers[int(rng.integers(len(tickers)))]
        px = prices[ticker].dropna()
        n = int(rng.choice(s.lengths))
        ends = np.flatnonzero((px.index >= pd.Timestamp(s.start)) & (np.arange(len(px)) >= n - 1))
        if ends.size == 0:
            continue
        end = int(rng.choice(ends))
        seg = px.iloc[end - n + 1 : end + 1]
        out.append(BenchWindow(name=f"{ticker}:{seg.index[-1].date()}:{n}",
                               t=dates_to_years(seg.index), y=np.log(seg.to_numpy(float))))
    return out


def synthetic_windows(s: BenchmarkSettings) -> list[BenchWindow]:
    rng = np.random.default_rng(s.seed + 1)
    out = []
    for i in range(s.n_synthetic):
        n = int(rng.choice(s.lengths))
        t = 2020.0 + np.arange(n) / 252.0
        dt = t[-1] - t[0]
        truth = dict(tc=t[-1] + rng.uniform(0.02, 0.3) * dt, m=rng.uniform(0.2, 0.8),
                     omega=rng.uniform(4, 15), A=5.0, B=-rng.uniform(0.3, 1.0),
                     C1=rng.uniform(-0.05, 0.05), C2=rng.uniform(-0.05, 0.05))
        y = lppls(t, **truth) + 0.01 * rng.normal(size=n)
        out.append(BenchWindow(name=f"synthetic:{i}:{n}", t=t, y=y, truth=truth))
    return out


def oracle(w: BenchWindow, bounds: SearchBounds, restarts: int, seed: int) -> tuple[float, np.ndarray]:
    """Best SSE found by several strong optimisers (the reference for 'excess SSE')."""
    b = bounds.for_window(float(w.t[0]), float(w.t[-1]))
    obj = make_objective(w.t, w.y)
    cands = []
    big = GAConfig(population_size=150, generations=150, patience=40, polish_maxiter=600)
    for r in range(restarts):
        res = ga_minimize(obj, b, big, np.random.default_rng([seed, r]))
        cands.append((res.fun, res.x))
    x, f, _ = baseline.differential_evolution(obj, b, np.random.default_rng([seed, 99]),
                                              maxiter=300, popsize=30)
    cands.append((f, x))
    x, f, _ = baseline.multistart_nelder_mead(obj, b, np.random.default_rng([seed, 98]),
                                              n_starts=40, maxiter=600)
    cands.append((f, x))
    f, x = min(cands, key=lambda c: c[0])
    return float(f), np.asarray(x)


def build_benchmark(prices: pd.DataFrame, s: BenchmarkSettings, bounds: SearchBounds,
                    n_jobs: int = -1) -> GABenchmark:
    windows = sample_real_windows(prices, s) + synthetic_windows(s)
    refs = Parallel(n_jobs=n_jobs)(delayed(oracle)(w, bounds, s.oracle_restarts, s.seed + i)
                                   for i, w in enumerate(windows))
    for w, (f, x) in zip(windows, refs, strict=True):
        w.ref_sse, w.ref_x = f, x
    bench = GABenchmark(windows=windows, bounds=bounds, meta=s.model_dump(mode="json"))
    bench.meta["digest"] = bench.digest
    return bench


def save_benchmark(bench: GABenchmark, path: Path = BENCHMARK_PATH) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "wb") as fh:
        pickle.dump(bench, fh)
    return path


def load_benchmark(path: Path = BENCHMARK_PATH) -> GABenchmark:
    if not Path(path).exists():
        raise FileNotFoundError(f"{path} not found: run `lppl tune build-benchmark` first")
    with open(path, "rb") as fh:
        return pickle.load(fh)  # noqa: S301 - file produced by this project


# --------------------------------------------------------------------------- evaluate
ReportFn = Callable[[int, dict[str, float]], None]


def evaluate_ga(params: dict, bench: GABenchmark, settings: GAStudySettings, seed: int = 0,
                report: ReportFn | None = None) -> dict[str, float]:
    """Run the candidate GA on every benchmark window. ``report(step, metrics)`` is called
    after each chunk with running metrics; it may raise to stop the trial (pruning)."""
    cfg = ga_config(params)
    windows = bench.windows
    order = np.random.default_rng(seed).permutation(len(windows))  # chunks mix real + synthetic
    chunks = np.array_split(order, max(1, settings.chunks))
    excess, evals, ms, tc_err = [], [], [], []
    metrics: dict[str, float] = {}
    for step, chunk in enumerate(chunks, start=1):
        for i in chunk:
            w = windows[i]
            for r in range(settings.seeds_per_window):
                res = fit_window(w.t, w.y, bounds=bench.bounds, ga_config=cfg,
                                 rng=np.random.default_rng([seed, int(i), r]))
                excess.append(max(res.fit.sse / w.ref_sse - 1.0, 0.0) * 100.0)
                evals.append(res.n_evals)
                ms.append(res.elapsed_ms)
                if w.ref_x is not None:
                    tc_err.append(abs(res.fit.tc - w.ref_x[0]) * 365.25)
        ex = np.asarray(excess)
        metrics = {
            "mean_excess_pct": float(ex.mean()),
            "p90_excess_pct": float(np.quantile(ex, 0.9)),
            "hit_rate": float((ex < 0.01).mean()),
            "mean_evals": float(np.mean(evals)),
            "mean_ms": float(np.mean(ms)),
            "median_tc_err_days": float(np.median(tc_err)) if tc_err else float("nan"),
            "windows_done": float(sum(len(c) for c in chunks[:step])),
        }
        metrics["loss"] = metrics["mean_excess_pct"] + settings.cost_weight * metrics["mean_evals"] / 1000
        if report is not None:
            report(step, metrics)
    return metrics
