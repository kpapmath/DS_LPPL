"""``lppl tune ...``: hyperparameter tuning commands (docs/tuning.md)."""

from __future__ import annotations

import datetime as dt
import json
import logging
import shutil
from pathlib import Path
from typing import Annotated

import typer

app = typer.Typer(no_args_is_help=True, pretty_exceptions_enable=False,
                  help="Hyperparameter tuning: Optuna search, Ray Tune execution, MLflow tracking.")

TuneCfgOpt = Annotated[Path | None, typer.Option("--tuning-config", help="default: config/tuning.yaml")]
CfgOpt = Annotated[Path | None, typer.Option("--config", "-c", help="experiment config (default.yaml)")]


def _setup():
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    for noisy in ("yfinance", "urllib3", "alembic", "mlflow", "optuna", "ray"):
        logging.getLogger(noisy).setLevel(logging.WARNING)


@app.command("init-stores")
def init_stores():
    """Create the `mlflow` and `optuna` databases in PostgreSQL (idempotent)."""
    from lppl.tuning.stores import ensure_databases, init_optuna_schema
    typer.echo(f"created: {ensure_databases() or 'nothing (already present)'}")
    init_optuna_schema()


@app.command("build-benchmark")
def build_benchmark(tuning_config: TuneCfgOpt = None, config: CfgOpt = None,
                    n_jobs: Annotated[int, typer.Option()] = -1,
                    force: Annotated[bool, typer.Option(help="rebuild even if cached")] = False):
    """Sample the GA benchmark windows and compute oracle reference solutions (cached)."""
    _setup()
    from lppl.config import load_config
    from lppl.data import db
    from lppl.tuning.ga_benchmark import (
        BENCHMARK_PATH,
        load_benchmark,
        save_benchmark,
    )
    from lppl.tuning.ga_benchmark import (
        build_benchmark as _build,
    )
    from lppl.tuning.settings import load_tuning_config
    if BENCHMARK_PATH.exists() and not force:
        b = load_benchmark()
        typer.echo(f"cached benchmark {BENCHMARK_PATH} ({len(b.windows)} windows, digest {b.digest}); "
                   "use --force to rebuild")
        return
    tcfg, cfg = load_tuning_config(tuning_config), load_config(config)
    bench = _build(db.load_all_prices(db.get_engine()), tcfg.ga.benchmark, cfg.bounds, n_jobs=n_jobs)
    path = save_benchmark(bench)
    typer.echo(f"wrote {path}: {len(bench.windows)} windows, digest {bench.digest}")


@app.command()
def run(study: Annotated[str, typer.Argument(help="ga | indicator")],
        backend: Annotated[str, typer.Option(help="optuna | ray")] = "optuna",
        trials: Annotated[int, typer.Option("--trials", "-n")] = 40,
        jobs: Annotated[int, typer.Option("--jobs", "-j", help="parallel trials")] = 1,
        name: Annotated[str | None, typer.Option(help="study name (resumes if it exists)")] = None,
        asha: Annotated[bool, typer.Option(help="ray: ASHA early stopping (GA study)")] = True,
        tuning_config: TuneCfgOpt = None, config: CfgOpt = None):
    """Run (or resume) a tuning study and log it to MLflow."""
    _setup()
    from lppl.config import load_config
    from lppl.data import db
    from lppl.tuning import runners, tracking
    from lppl.tuning.settings import STUDIES, load_tuning_config, validate_space
    from lppl.tuning.stores import ensure_databases, optuna_storage_url

    if study not in STUDIES:
        raise typer.BadParameter(f"study must be one of {STUDIES}")
    if backend not in ("optuna", "ray"):
        raise typer.BadParameter("backend must be optuna or ray")
    tcfg, cfg = load_tuning_config(tuning_config), load_config(config)
    validate_space(getattr(tcfg, study).space)
    ensure_databases()
    name = name or f"{study}-{backend}-{dt.datetime.now():%Y%m%d-%H%M%S}"
    spec, payload = runners.prepare(study, tcfg, cfg, db.get_engine(), name)
    storage = optuna_storage_url()
    experiment = tcfg.experiment(study)
    cl = tracking.client()
    study_settings = getattr(tcfg, study).model_dump(mode="json")
    parent = tracking.start_parent(
        cl, experiment, name, tracking.base_tags(study, name, backend, spec.data_tags),
        {"backend": backend, "n_trials": trials, "jobs": jobs, "seed": tcfg.seed,
         "metric": spec.metric, "mode": spec.mode,
         **{f"study.{k}": json.dumps(v) if isinstance(v, (dict, list)) else v
            for k, v in study_settings.items()}})
    tracking.log_dict(cl, parent, {"space": spec.space, "defaults": spec.defaults,
                                   "data": spec.data_tags}, "study_definition.json")
    typer.echo(f"study {name} | backend {backend} | MLflow parent run {parent}")
    try:
        if backend == "optuna":
            table = runners.run_optuna(spec, payload, tcfg, trials, jobs, storage, experiment, parent, cl)
        else:
            table = runners.run_ray(spec, payload, tcfg, trials, max(1, jobs), storage, experiment,
                                    parent, num_cpus=max(1, jobs), asha=asha)
        summary = runners.summarize(spec, table, tcfg, cfg, storage, cl, parent, payload)
    except BaseException:
        tracking.finish(cl, parent, "FAILED")
        raise
    tracking.finish(cl, parent)
    _print_summary(summary)


def _print_summary(s: dict):
    typer.echo(f"\n== {s['study']} ({s['n_complete']}/{s['n_trials']} complete) ==")
    typer.echo("best params: " + json.dumps(s["best_params"], default=str))
    keys = (["loss", "mean_excess_pct", "hit_rate", "mean_evals", "median_tc_err_days"] if s["kind"] == "ga"
            else ["ap_train", "ap_valid", "lift_train", "lift_valid", "precision_at_0.3_valid",
                  "recall_at_0.3_valid"])
    def fmt(v) -> str:
        return f"{v:12.4f}" if isinstance(v, (int, float)) else f"{'-':>12}"

    typer.echo(f"{'metric':<24}{'current':>12}{'tuned':>12}")
    for k in keys:
        typer.echo(f"{k:<24}{fmt(s['baseline_metrics'].get(k))}{fmt(s['best_metrics'].get(k))}")
    if s["kind"] == "indicator":
        d = s["data"]
        typer.echo(f"{'base rate':<24}{d['base_rate_train']:12.4f} (train) {d['base_rate_valid']:.4f} (valid)")
        typer.echo(f"{'momentum baseline AP':<24}{d['baseline_ap_train']:12.4f} (train) "
                   f"{d['baseline_ap_valid']:.4f} (valid)")
    typer.echo(f"tuned config: data/tuning/{s['study']}/tuned_config.yaml")


@app.command()
def best(name: str):
    """Print the summary of a finished study (data/tuning/<name>/best.json)."""
    from lppl.tuning.runners import TUNING_DIR
    _print_summary(json.loads((TUNING_DIR / name / "best.json").read_text()))


@app.command("export-best")
def export_best(name: str, out: Annotated[Path, typer.Option()] = Path("config/tuned.yaml")):
    """Copy a study's tuned experiment config to OUT (then: lppl scan -c OUT --label tuned)."""
    from lppl.tuning.runners import TUNING_DIR
    src = TUNING_DIR / name / "tuned_config.yaml"
    out.parent.mkdir(parents=True, exist_ok=True)
    shutil.copyfile(src, out)
    typer.echo(f"wrote {out}")


@app.command()
def adopt(ga_study: Annotated[str | None, typer.Option(help="GA study name (data/tuning/<name>)")] = None,
          indicator_study: Annotated[str | None, typer.Option(help="detector study name")] = None,
          config: CfgOpt = None,
          out: Annotated[Path, typer.Option(help="tuned experiment config to write")] = Path("config/tuned.yaml"),
          policy: Annotated[str, typer.Option(help="validated | all")] = "validated",
          tuning_config: TuneCfgOpt = None):
    """Build a tuned experiment config from finished studies.

    policy=validated (default): GA settings are adopted only if they beat the current GA on the
    benchmark loss; detector settings only if they also improve AP on the held-out validation
    period. policy=all adopts the best settings of each study regardless."""
    import yaml

    from lppl.config import load_config
    from lppl.tuning.runners import TUNING_DIR
    from lppl.tuning.settings import apply_params, coerce, config_to_yaml, load_tuning_config

    if policy not in ("validated", "all"):
        raise typer.BadParameter("policy must be 'validated' or 'all'")
    cfg = load_config(config)
    space = load_tuning_config(tuning_config)
    decisions = []
    for kind, name in (("ga", ga_study), ("indicator", indicator_study)):
        if not name:
            continue
        s = json.loads((TUNING_DIR / name / "best.json").read_text())
        best, base = s["best_metrics"], s["baseline_metrics"]
        if kind == "ga":
            key, better = "loss", best.get("loss", float("inf")) < base.get("loss", float("inf"))
        else:
            key, better = "ap_valid", best.get("ap_valid", 0.0) > base.get("ap_valid", 0.0)
        take = policy == "all" or better
        decisions.append({"study": name, "kind": kind, "criterion": key, "current": base.get(key),
                          "tuned": best.get(key), "adopted": bool(take),
                          "params": s["best_params"] if take else {}})
        if take:
            params = coerce(s["best_params"], getattr(space, kind).space)
            cfg = apply_params(cfg, kind, params, s.get("data", {}).get("window_grid_step"))
        typer.echo(f"{kind:9} {name}: {key} current {base.get(key)} -> tuned {best.get(key)} "
                   f"=> {'ADOPT' if take else 'keep current'}")
    header = "Tuned config written by `lppl tune adopt` (policy: " + policy + ")\n" + "\n".join(
        f"{d['kind']} study {d['study']}: {d['criterion']} {d['current']} -> {d['tuned']}: "
        f"{'adopted' if d['adopted'] else 'NOT adopted, current values kept'}" for d in decisions)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(config_to_yaml(cfg, header))
    out.with_suffix(".decisions.yaml").write_text(yaml.safe_dump(decisions, sort_keys=False))
    typer.echo(f"wrote {out} (+ {out.with_suffix('.decisions.yaml').name})")


@app.command("evaluate-config")
def evaluate_config(config: CfgOpt = None, tuning_config: TuneCfgOpt = None):
    """Score an experiment config's detector (filters + windows) on the stored fits."""
    from lppl.config import load_config
    from lppl.data import db
    from lppl.tuning.detection import build_detection_data, evaluate_detector
    from lppl.tuning.settings import default_params, load_tuning_config
    tcfg, cfg = load_tuning_config(tuning_config), load_config(config)
    s = tcfg.indicator
    eng = db.get_engine()
    run_id = db.latest_run_for_label(eng, s.source_label)
    data = build_detection_data(db.load_fits(eng, run_id), db.load_all_prices(eng), s, run_id)
    m = evaluate_detector(default_params(cfg, "indicator", s.space, data.grid_step), data, s)
    typer.echo(json.dumps({"config": str(config or "config/default.yaml"), "source_run": run_id,
                           **{k: round(v, 4) for k, v in m.items()},
                           **{k: (round(v, 4) if isinstance(v, float) else v) for k, v in data.meta.items()}},
                          indent=2))
