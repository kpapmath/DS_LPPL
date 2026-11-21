"""Run a tuning study with Optuna or Ray Tune and track it in MLflow.

Both backends share the same search space (``settings.suggest``, define-by-run), the
same evaluation functions and an Optuna study persisted in PostgreSQL.

* ``optuna``: ``study.optimize`` with a TPE sampler, a median pruner for the GA study
  (intermediate reports per chunk), threads via ``n_jobs``. Exactly reproducible with
  ``n_jobs=1``.
* ``ray``: ``ray.tune.Tuner`` with ``OptunaSearch`` (TPE), ``ConcurrencyLimiter``, an ASHA
  scheduler for the GA study, and ``MLflowLoggerCallback`` for per-trial tracking.
  Parallel trials make the sampling order (and so the trial sequence)
  non-deterministic.

The current configuration is always evaluated first (enqueued trial 0), so every study
reports its improvement over the status quo.
"""

from __future__ import annotations

import json
import logging
from dataclasses import dataclass, field
from typing import Any

import numpy as np
import pandas as pd

from lppl.config import PROJECT_ROOT, AppConfig
from lppl.tuning import tracking
from lppl.tuning.settings import (
    TuningConfig,
    apply_params,
    coerce,
    config_to_yaml,
    default_params,
    suggest,
)

log = logging.getLogger(__name__)
TUNING_DIR = PROJECT_ROOT / "data" / "tuning"


@dataclass
class StudySpec:
    kind: str                 # "ga" | "indicator"
    name: str
    metric: str               # metric optimised
    mode: str                 # "min" | "max"
    space: dict[str, Any]
    defaults: dict[str, Any]
    n_steps: int              # intermediate reports per trial
    data_tags: dict[str, Any] = field(default_factory=dict)

    @property
    def direction(self) -> str:
        return "minimize" if self.mode == "min" else "maximize"


# --------------------------------------------------------------------------- payloads
def prepare(kind: str, tcfg: TuningConfig, app_cfg: AppConfig, engine, name: str):
    """Load the data a study needs. Returns (spec, payload)."""
    from lppl.data import db

    if kind == "ga":
        from lppl.tuning.ga_benchmark import load_benchmark
        bench = load_benchmark()
        spec = StudySpec("ga", name, "loss", "min", tcfg.ga.space,
                         default_params(app_cfg, "ga", tcfg.ga.space), tcfg.ga.chunks,
                         {"benchmark_digest": bench.digest, "benchmark_windows": len(bench.windows)})
        return spec, bench
    if kind == "indicator":
        from lppl.tuning.detection import build_detection_data
        s = tcfg.indicator
        run_id = db.latest_run_for_label(engine, s.source_label)
        if run_id is None:
            raise RuntimeError(f"no finished scan labelled {s.source_label!r}; run `lppl scan` first")
        data = build_detection_data(db.load_fits(engine, run_id), db.load_all_prices(engine), s, run_id)
        source_cfg = source_run_config(engine, run_id)
        meta = dict(data.meta, window_grid_step=data.grid_step,
                    source_windows=f"{source_cfg.windows.min_len}-{source_cfg.windows.max_len}"
                                   f"/{source_cfg.windows.step}")
        spec = StudySpec("indicator", name, "objective", "max", s.space,
                         default_params(source_cfg, "indicator", s.space, data.grid_step), 1, meta)
        return spec, data
    raise ValueError(f"unknown study {kind!r}")


def source_run_config(engine, run_id: str) -> AppConfig:
    """The effective config stored with a scan run (filters reflect any `requalify`)."""
    from lppl.data import db

    with engine.connect() as conn:
        raw = conn.execute(db.fit_runs.select().where(db.fit_runs.c.run_id == run_id)).one().config
    return AppConfig.model_validate({k: v for k, v in raw.items() if k != "tickers"})


def evaluate(kind: str, params: dict, payload, tcfg: TuningConfig, report=None) -> dict[str, float]:
    if kind == "ga":
        from lppl.tuning.ga_benchmark import evaluate_ga
        return evaluate_ga(params, payload, tcfg.ga, seed=tcfg.seed, report=report)
    from lppl.tuning.detection import evaluate_detector
    out = evaluate_detector(params, payload, tcfg.indicator)
    if report is not None:
        report(1, out)
    return out


# --------------------------------------------------------------------------- optuna
def run_optuna(spec: StudySpec, payload, tcfg: TuningConfig, n_trials: int, n_jobs: int,
               storage: str, experiment: str, parent_id: str, cl) -> pd.DataFrame:
    import optuna

    sampler = optuna.samplers.TPESampler(seed=tcfg.seed, multivariate=True, n_startup_trials=10)
    pruner = (optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1)
              if spec.n_steps > 1 else optuna.pruners.NopPruner())
    study = optuna.create_study(study_name=spec.name, storage=storage, direction=spec.direction,
                                sampler=sampler, pruner=pruner, load_if_exists=True)
    if not study.trials:
        study.enqueue_trial(spec.defaults)
    tags = {"mlflow.source.name": "lppl tune"}

    def objective(trial: optuna.Trial) -> float:
        params = suggest(trial, spec.space)
        history: list[tuple[int, dict]] = []

        def report(step: int, metrics: dict) -> None:
            history.append((step, dict(metrics)))
            trial.report(metrics[spec.metric], step)
            if step < spec.n_steps and trial.should_prune():
                raise optuna.TrialPruned()

        try:
            final = evaluate(spec.kind, params, payload, tcfg, report)
        except optuna.TrialPruned:
            last = history[-1][1] if history else {}
            tracking.log_trial(cl, experiment, parent_id, trial.number, params, last,
                               history[:-1], state="PRUNED", tags=tags)
            raise
        for k, v in final.items():
            trial.set_user_attr(k, v)
        tracking.log_trial(cl, experiment, parent_id, trial.number, params, final,
                           history[:-1], tags=tags)
        return final[spec.metric]

    optuna.logging.set_verbosity(optuna.logging.WARNING)
    study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs, gc_after_trial=False,
                   show_progress_bar=False)
    rows = []
    for t in study.trials:
        rows.append({"number": t.number, "state": t.state.name, "value": t.value,
                     **{f"param.{k}": v for k, v in t.params.items()},
                     **{f"metric.{k}": v for k, v in t.user_attrs.items()}})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- ray
def _ray_trainable(config: dict, kind: str, payload, tcfg: TuningConfig):
    from ray import tune

    def report(step: int, metrics: dict) -> None:
        tune.report(dict(metrics, step=step))

    evaluate(kind, dict(config), payload, tcfg, report)


class DefineByRun:
    """Picklable define-by-run space for Ray's OptunaSearch (Ray pickles the searcher when
    it checkpoints experiment state, so a closure cannot be used)."""

    def __init__(self, space: dict[str, Any]):
        self.space = space

    def __call__(self, trial) -> None:  # returns no constants
        suggest(trial, self.space)
        return None


def run_ray(spec: StudySpec, payload, tcfg: TuningConfig, n_trials: int, max_concurrent: int,
            storage: str, experiment: str, parent_id: str, num_cpus: int | None = None,
            asha: bool = True) -> pd.DataFrame:
    import optuna
    import ray
    from ray import tune
    from ray.air.integrations.mlflow import MLflowLoggerCallback
    from ray.tune.schedulers import ASHAScheduler
    from ray.tune.search import ConcurrencyLimiter
    from ray.tune.search.optuna import OptunaSearch

    # payloads are a few MB: 1 GB of object store fits in the container's 2 GB /dev/shm
    # (Ray's default of 30 % of RAM would spill to disk)
    ray.init(num_cpus=num_cpus, include_dashboard=False, ignore_reinit_error=True,
             log_to_driver=False, configure_logging=False, object_store_memory=10**9)

    searcher = OptunaSearch(
        space=DefineByRun(spec.space), metric=spec.metric, mode=spec.mode, points_to_evaluate=[spec.defaults],
        sampler=optuna.samplers.TPESampler(seed=tcfg.seed, multivariate=True, n_startup_trials=10),
        study_name=spec.name, storage=optuna.storages.RDBStorage(storage))
    searcher = ConcurrencyLimiter(searcher, max_concurrent=max_concurrent)
    scheduler = (ASHAScheduler(time_attr="training_iteration", max_t=spec.n_steps,
                               grace_period=1, reduction_factor=2)
                 if asha and spec.n_steps > 1 else None)
    callback = MLflowLoggerCallback(
        tracking_uri=tracking.mlflow_tracking_uri(), experiment_name=experiment,
        tags={"mlflow.parentRunId": parent_id, "mlflow.source.name": "lppl tune (ray)"})
    trainable = tune.with_resources(
        tune.with_parameters(_ray_trainable, kind=spec.kind, payload=payload, tcfg=tcfg),
        {"cpu": 1})
    tuner = tune.Tuner(
        trainable,
        tune_config=tune.TuneConfig(metric=spec.metric, mode=spec.mode, search_alg=searcher,
                                    scheduler=scheduler, num_samples=n_trials,
                                    reuse_actors=True),
        run_config=tune.RunConfig(name=spec.name, storage_path=str(TUNING_DIR / "ray_results"),
                                  callbacks=[callback], verbose=1),
    )
    results = tuner.fit()
    df = results.get_dataframe()
    ray.shutdown()
    rows = []
    for _, r in df.iterrows():
        complete = int(r.get("training_iteration", 0)) >= spec.n_steps
        rows.append({
            "number": r.get("trial_id"), "state": "COMPLETE" if complete else "STOPPED",
            "value": r.get(spec.metric),
            **{f"param.{c[7:]}": r[c] for c in df.columns if c.startswith("config/")},
            **{f"metric.{c}": r[c] for c in df.columns
               if not c.startswith("config/") and isinstance(r[c], (int, float, np.floating))}})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- summary
def best_row(trials: pd.DataFrame, spec: StudySpec) -> pd.Series:
    done = trials[(trials["state"] == "COMPLETE") & trials["value"].notna()]
    if done.empty:
        raise RuntimeError("no completed trials")
    return done.loc[done["value"].idxmin() if spec.mode == "min" else done["value"].idxmax()]


def baseline_row(trials: pd.DataFrame, spec: StudySpec) -> pd.Series | None:
    """The completed trial whose parameters equal the enqueued status-quo configuration."""
    for _, row in trials[trials["state"] == "COMPLETE"].iterrows():
        p = coerce(params_of(row), spec.space)
        if set(p) == set(spec.defaults) and all(
                np.isclose(float(p[k]), float(v)) if isinstance(v, (int, float)) and not isinstance(v, bool)
                else p[k] == v for k, v in spec.defaults.items()):
            return row
    return None


def params_of(row: pd.Series) -> dict[str, Any]:
    out = {}
    for k, v in row.items():
        if k.startswith("param.") and not (isinstance(v, float) and np.isnan(v)):
            out[k[6:]] = v.item() if hasattr(v, "item") else v
    return out


def metrics_of(row: pd.Series) -> dict[str, float]:
    return {k[7:]: v for k, v in row.items() if k.startswith("metric.")}


def summarize(spec: StudySpec, trials: pd.DataFrame, tcfg: TuningConfig, app_cfg: AppConfig,
              storage: str, cl, parent_id: str, payload=None) -> dict[str, Any]:
    """Log best params/metrics, importances, plots and the tuned config to the parent run,
    and save ``data/tuning/<study>/{best.json,tuned_config.yaml,trials.csv}``."""
    best = best_row(trials, spec)
    best_params = coerce(params_of(best), spec.space)
    base = baseline_row(trials, spec)
    baseline_metrics = metrics_of(base) if base is not None else {}
    best_metrics = metrics_of(best)
    if spec.kind == "indicator" and payload is not None:
        # always re-evaluate both on the full data so train/valid figures are comparable
        best_metrics = evaluate("indicator", best_params, payload, tcfg)
        baseline_metrics = evaluate("indicator", spec.defaults, payload, tcfg)

    out_dir = TUNING_DIR / spec.name
    out_dir.mkdir(parents=True, exist_ok=True)
    tuned = apply_params(app_cfg, spec.kind, best_params, spec.data_tags.get("window_grid_step"))
    header = (f"Tuned by `lppl tune run {spec.kind}` - study {spec.name}\n"
              f"MLflow parent run {parent_id}; optimised {spec.metric} ({spec.mode})\n"
              f"data: {json.dumps({k: v for k, v in spec.data_tags.items() if not isinstance(v, float)})}")
    yaml_text = config_to_yaml(tuned, header)
    (out_dir / "tuned_config.yaml").write_text(yaml_text)
    summary = {"study": spec.name, "kind": spec.kind, "metric": spec.metric, "mode": spec.mode,
               "best_params": best_params, "best_metrics": best_metrics,
               "baseline_params": spec.defaults, "baseline_metrics": baseline_metrics,
               "n_trials": int(len(trials)), "n_complete": int((trials["state"] == "COMPLETE").sum()),
               "data": spec.data_tags, "mlflow_parent_run_id": parent_id}
    trials.to_csv(out_dir / "trials.csv", index=False)

    tracking.log_params(cl, parent_id, {f"best.{k}": v for k, v in best_params.items()})
    tracking.log_metrics(cl, parent_id, {f"best_{k}": v for k, v in best_metrics.items()})
    tracking.log_metrics(cl, parent_id, {f"baseline_{k}": v for k, v in baseline_metrics.items()})
    tracking.log_metrics(cl, parent_id, {"n_trials": len(trials),
                                         "n_complete": summary["n_complete"]})
    tracking.log_text(cl, parent_id, yaml_text, "tuned_config.yaml")
    tracking.log_dataframe(cl, parent_id, trials, "trials.csv")

    imp = _importances(spec, storage)
    if imp:
        tracking.log_dict(cl, parent_id, imp, "param_importances.json")
        tracking.log_metrics(cl, parent_id, {f"importance.{k}": v for k, v in imp.items()})
        summary["importances"] = imp
    (out_dir / "best.json").write_text(json.dumps(summary, indent=2, default=str))
    tracking.log_dict(cl, parent_id, summary, "best.json")
    for fname, fig in _figures(spec, trials, summary, payload):
        tracking.log_figure(cl, parent_id, fig, fname)
        fig.savefig(out_dir / fname, bbox_inches="tight")
    return summary


def _importances(spec: StudySpec, storage: str) -> dict[str, float]:
    try:
        import optuna

        study = optuna.load_study(study_name=spec.name, storage=storage)
        return {k: float(v) for k, v in optuna.importance.get_param_importances(study).items()}
    except Exception as exc:  # too few trials, conditional params, ...
        log.warning("parameter importances unavailable: %s", exc)
        return {}


def _figures(spec: StudySpec, trials: pd.DataFrame, summary: dict, payload):
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    from lppl import viz

    viz.apply_style()
    figs = []
    done = trials[trials["state"] == "COMPLETE"].reset_index(drop=True)
    # optimisation history
    fig, ax = plt.subplots(figsize=(8, 3.4))
    ax.scatter(range(len(done)), done["value"], s=22, color=viz.SERIES[0], label="trial",
               edgecolor=viz.SURFACE, linewidth=1)
    run_best = done["value"].cummin() if spec.mode == "min" else done["value"].cummax()
    ax.step(range(len(done)), run_best, where="post", color=viz.POS, lw=1.6, label="best so far")
    ax.set_xlabel("completed trial")
    ax.set_ylabel(spec.metric)
    ax.set_title(f"{spec.name}: optimisation history")
    ax.legend()
    figs.append(("history.png", fig))

    if spec.kind == "ga" and {"metric.mean_excess_pct", "metric.mean_evals"} <= set(done.columns):
        fig, ax = plt.subplots(figsize=(7, 4))
        ax.scatter(done["metric.mean_evals"], done["metric.mean_excess_pct"] + 1e-3, s=24,
                   color=viz.MUTED, label="trials", edgecolor=viz.SURFACE, linewidth=1)
        b, d = summary["best_metrics"], summary["baseline_metrics"]
        if d:
            ax.scatter([d["mean_evals"]], [d["mean_excess_pct"] + 1e-3], s=70, color=viz.SERIES[0],
                       label="current config", zorder=3, edgecolor=viz.SURFACE, linewidth=1.5)
        ax.scatter([b["mean_evals"]], [b["mean_excess_pct"] + 1e-3], s=70, color=viz.POS,
                   label="best (loss)", zorder=3, edgecolor=viz.SURFACE, linewidth=1.5)
        ax.set_yscale("log")
        ax.set_xscale("log")
        ax.set_xlabel("mean objective evaluations per fit (cost)")
        ax.set_ylabel("mean excess SSE over oracle (%) + 0.001")
        ax.set_title("Quality vs. cost")
        ax.legend()
        figs.append(("quality_vs_cost.png", fig))

    if spec.kind == "indicator":
        b, d, meta = summary["best_metrics"], summary["baseline_metrics"], spec.data_tags
        rows = [("base rate", meta.get("base_rate_train"), meta.get("base_rate_valid")),
                ("momentum baseline", meta.get("baseline_ap_train"), meta.get("baseline_ap_valid")),
                ("current detector", d.get("ap_train"), d.get("ap_valid")),
                ("tuned detector", b.get("ap_train"), b.get("ap_valid"))]
        fig, ax = plt.subplots(figsize=(8, 3.6))
        y = np.arange(len(rows))
        ax.barh(y + 0.2, [r[1] for r in rows], height=0.36, color=viz.SERIES[0], label="train (2019-2023)")
        ax.barh(y - 0.2, [r[2] for r in rows], height=0.36, color=viz.SERIES[1], label="validation (2024-)")
        ax.set_yticks(y, [r[0] for r in rows])
        ax.invert_yaxis()
        ax.set_xlabel("average precision for forward-drawdown events")
        ax.set_title("Does the detector beat the base rate out of sample?")
        ax.legend(loc="lower right")
        figs.append(("ap_train_vs_valid.png", fig))
    return figs
