"""MLflow tracking for tuning studies.

Layout in MLflow: one experiment per study kind (``lppl-ga``, ``lppl-indicator``). Each study
gets one *parent* run, and each trial is a *child* run (``mlflow.parentRunId``) with its
parameters, final metrics and the intermediate metric history (``step`` = chunk). The
parent run collects the best parameters and metrics, parameter importances, the trials
table, plots and the ready-to-use tuned config YAML.

Only ``MlflowClient`` is used (no fluent global state), so logging is safe from Optuna's
worker threads.
"""

from __future__ import annotations

import json
import math
import tempfile
from pathlib import Path
from typing import Any

from lppl import __version__
from lppl.pipeline import git_sha
from lppl.tuning.stores import mlflow_tracking_uri


def client():
    import mlflow
    from mlflow.tracking import MlflowClient

    mlflow.set_tracking_uri(mlflow_tracking_uri())
    return MlflowClient(tracking_uri=mlflow_tracking_uri())


def experiment_id(cl, name: str) -> str:
    exp = cl.get_experiment_by_name(name)
    return exp.experiment_id if exp else cl.create_experiment(name)


def base_tags(study_kind: str, study_name: str, backend: str, extra: dict | None = None) -> dict[str, str]:
    tags = {"study": study_name, "study_kind": study_kind, "backend": backend,
            "lppl_version": __version__, "git_sha": git_sha() or "unknown"}
    tags.update({k: str(v) for k, v in (extra or {}).items()})
    return tags


def _clean_metrics(metrics: dict[str, Any]) -> dict[str, float]:
    out = {}
    for k, v in metrics.items():
        try:
            f = float(v)
        except (TypeError, ValueError):
            continue
        if math.isfinite(f):
            out[k] = f
    return out


def start_parent(cl, experiment: str, study_name: str, tags: dict, params: dict) -> str:
    run = cl.create_run(experiment_id(cl, experiment), run_name=study_name, tags=tags)
    log_params(cl, run.info.run_id, params)
    return run.info.run_id


def log_params(cl, run_id: str, params: dict) -> None:
    from mlflow.entities import Param

    items = [Param(k, str(v)[:6000]) for k, v in params.items()]
    for i in range(0, len(items), 100):
        cl.log_batch(run_id, params=items[i : i + 100])


def log_metrics(cl, run_id: str, metrics: dict, step: int = 0) -> None:
    from mlflow.entities import Metric

    ts = _now_ms()
    items = [Metric(k, v, ts, step) for k, v in _clean_metrics(metrics).items()]
    if items:
        cl.log_batch(run_id, metrics=items)


def log_trial(cl, experiment: str, parent_id: str, number: int, params: dict,
              final: dict, history: list[tuple[int, dict]] | None = None,
              state: str = "COMPLETE", tags: dict | None = None) -> str:
    t = {"mlflow.parentRunId": parent_id, "trial_number": str(number), "trial_state": state}
    t.update(tags or {})
    run = cl.create_run(experiment_id(cl, experiment), run_name=f"trial-{number:04d}", tags=t)
    rid = run.info.run_id
    log_params(cl, rid, params)
    for step, m in history or []:
        log_metrics(cl, rid, m, step=step)
    log_metrics(cl, rid, final, step=len(history or []))
    cl.set_terminated(rid, "FINISHED" if state == "COMPLETE" else "KILLED")
    return rid


def log_text(cl, run_id: str, text: str, artifact_file: str) -> None:
    cl.log_text(run_id, text, artifact_file)


def log_figure(cl, run_id: str, fig, artifact_file: str) -> None:
    cl.log_figure(run_id, fig, artifact_file)


def log_dict(cl, run_id: str, d: dict, artifact_file: str) -> None:
    cl.log_text(run_id, json.dumps(d, indent=2, default=str), artifact_file)


def log_file(cl, run_id: str, path: Path, artifact_path: str | None = None) -> None:
    cl.log_artifact(run_id, str(path), artifact_path)


def log_dataframe(cl, run_id: str, df, artifact_file: str) -> None:
    with tempfile.TemporaryDirectory() as d:
        p = Path(d) / artifact_file
        df.to_csv(p, index=False)
        cl.log_artifact(run_id, str(p))


def finish(cl, run_id: str, status: str = "FINISHED") -> None:
    cl.set_terminated(run_id, status)


def _now_ms() -> int:
    import time

    return int(time.time() * 1000)
