"""Command-line interface: ``lppl --help``."""

from __future__ import annotations

import datetime as dt
import json
import logging
from pathlib import Path
from typing import Annotated

import numpy as np
import pandas as pd
import typer

from lppl.config import load_config

app = typer.Typer(add_completion=False, no_args_is_help=True, pretty_exceptions_enable=False,
                  help="LPPLS bubble detection with a genetic algorithm.")

ConfigOpt = Annotated[Path | None, typer.Option("--config", "-c", help="YAML config file")]
TickersOpt = Annotated[str | None, typer.Option("--tickers", "-t", help="comma-separated subset")]


def _setup(verbose: bool = False):
    logging.basicConfig(level=logging.DEBUG if verbose else logging.INFO,
                        format="%(asctime)s %(levelname)s %(name)s: %(message)s")
    logging.getLogger("yfinance").setLevel(logging.WARNING)


def _tickers(s: str | None) -> list[str] | None:
    return [x.strip() for x in s.split(",") if x.strip()] if s else None


def _engine():
    from lppl.data.db import get_engine
    return get_engine()


@app.command("init-db")
def init_db_cmd():
    """Create all tables (idempotent)."""
    from lppl.data import db
    eng = _engine()
    db.init_db(eng)
    typer.echo(json.dumps(db.table_counts(eng), indent=2))


@app.command()
def ingest(config: ConfigOpt = None, tickers: TickersOpt = None,
           from_snapshot: Annotated[bool, typer.Option(help="load data/snapshots/*.csv instead of Yahoo")] = False):
    """Download prices from Yahoo Finance into PostgreSQL (and save CSV snapshots)."""
    _setup()
    from lppl import pipeline
    summary = pipeline.ingest(load_config(config), _engine(), from_snapshot=from_snapshot,
                              tickers=_tickers(tickers))
    typer.echo(summary.to_string(index=False))


@app.command()
def scan(config: ConfigOpt = None, tickers: TickersOpt = None,
         start: Annotated[dt.datetime | None, typer.Option(formats=["%Y-%m-%d"])] = None,
         end: Annotated[dt.datetime | None, typer.Option(formats=["%Y-%m-%d"])] = None,
         step: Annotated[int | None, typer.Option(help="trading days between analysis dates")] = None,
         n_jobs: Annotated[int | None, typer.Option()] = None,
         label: Annotated[str | None, typer.Option()] = None):
    """Run the multi-window LPPLS/GA scan and store fits + confidence indicator."""
    _setup()
    from lppl import pipeline
    cfg = load_config(config)
    upd = {k: v for k, v in {"start": start and start.date(), "end": end and end.date(),
                             "step": step, "n_jobs": n_jobs}.items() if v is not None}
    if upd:
        cfg = cfg.model_copy(update={"scan": cfg.scan.model_copy(update=upd)})
    run_id = pipeline.scan(cfg, _engine(), tickers=_tickers(tickers), label=label)
    typer.echo(f"run_id={run_id}")
    typer.echo(pipeline.report(_engine(), run_id).to_string())


@app.command()
def requalify(config: ConfigOpt = None,
              run_id: Annotated[str | None, typer.Option(help="default: all finished runs")] = None):
    """Re-apply the config's qualification filters to stored fits (no refitting)."""
    from lppl import pipeline
    from lppl.data import db
    cfg, eng = load_config(config), _engine()
    runs_df = db.list_runs(eng)
    ids = [run_id] if run_id else runs_df.loc[runs_df.status == "finished", "run_id"].tolist()
    for rid in ids:
        n = pipeline.requalify(cfg, eng, rid)
        typer.echo(f"{rid}: {n} fits changed qualification")


@app.command()
def report(run_id: Annotated[str | None, typer.Option(help="default: latest run labelled LABEL")] = None,
           label: Annotated[str, typer.Option()] = "default",
           csv: Annotated[Path | None, typer.Option(help="also write the table to CSV")] = None):
    """Print the latest bubble-indicator reading for each ticker."""
    from lppl import pipeline
    table = pipeline.report(_engine(), run_id, label)
    if table.empty:
        typer.echo("no indicator rows yet; run `lppl scan`")
        raise typer.Exit(1)
    pd.set_option("display.width", 200)
    typer.echo(table.to_string())
    if csv:
        csv.parent.mkdir(parents=True, exist_ok=True)
        table.to_csv(csv)


@app.command()
def runs():
    """List fit runs."""
    from lppl.data import db
    typer.echo(db.list_runs(_engine()).to_string(index=False))


@app.command("delete-run")
def delete_run(run_id: str):
    """Delete one run with its fits and indicator rows."""
    from lppl.data import db
    db.delete_run(_engine(), run_id)
    typer.echo(f"deleted {run_id}")


@app.command()
def export(out: Annotated[Path, typer.Option(help="CSV path")] = Path("reports/reference.csv"),
           run_id: Annotated[str | None, typer.Option()] = None):
    """Export every indicator row (all finished runs, or one run) to CSV."""
    from lppl.data import db
    ind = db.load_indicator(_engine(), run_id).drop(columns="run_id")
    ind = ind[["label"] + [c for c in ind.columns if c != "label"]]
    out.parent.mkdir(parents=True, exist_ok=True)
    ind.to_csv(out, index=False, float_format="%.10g")
    typer.echo(f"wrote {len(ind)} rows to {out}")


@app.command()
def verify(reference: Annotated[Path, typer.Option()] = Path("reports/reference.csv"),
           atol: Annotated[float, typer.Option()] = 1e-9):
    """Compare the indicator in the DB with a reference CSV (reproducibility check)."""
    from lppl.data import db
    ref = pd.read_csv(reference, parse_dates=["date"], keep_default_na=False,
                      na_values={"pos_conf": [""], "neg_conf": [""]})
    cur = db.load_indicator(_engine())
    m = ref.merge(cur, on=["label", "ticker", "date"], how="left", suffixes=("_ref", ""),
                  indicator=True)
    missing = int((m["_merge"] != "both").sum())
    diff = np.maximum((m["pos_conf"] - m["pos_conf_ref"]).abs(),
                      (m["neg_conf"] - m["neg_conf_ref"]).abs())
    bad = int((diff > atol).sum())
    typer.echo(f"reference rows: {len(ref)}, missing in DB: {missing}, "
               f"differing (> {atol}): {bad}, max abs diff: {float(diff.max()):.3g}")
    if missing or bad:
        typer.echo(m.loc[diff > atol, ["ticker", "date", "pos_conf_ref", "pos_conf",
                                       "neg_conf_ref", "neg_conf"]].head(20).to_string())
        raise typer.Exit(1)
    typer.echo("OK: results reproduced")


@app.command()
def validate(
    label: Annotated[str, typer.Option(help="validate the latest finished run with this label")] = "default",
    run_id: Annotated[str | None, typer.Option(help="validate this run instead")] = None,
    vconfig: Annotated[Path | None, typer.Option("--validation-config",
                                                 help="default: config/validation.yaml")] = None,
    null_paths: Annotated[int | None, typer.Option(help="override null_model.paths (0 = skip)")] = None,
    walk_forward: Annotated[bool, typer.Option(help="run the walk-forward folds")] = True,
    out: Annotated[Path | None, typer.Option(help="default: reports/validation/<label>_<run>")] = None,
    mlflow: Annotated[bool, typer.Option(help="log to MLflow if reachable")] = True,
    n_jobs: Annotated[int, typer.Option()] = -1,
    tuning_config: Annotated[Path | None, typer.Option(
        help="search space for walk-forward re-tuning (default: config/tuning.yaml)")] = None,
):
    """Validate a stored scan: skill, calibration, bootstrap CIs, placebo, robustness grid,
    timing, walk-forward, null-model false alarms and a de-risking backtest."""
    _setup()
    from lppl import pipeline, validation
    from lppl.data import db

    eng = _engine()
    rid = pipeline.resolve_run(eng, run_id, label)
    if rid is None:
        typer.echo("no finished run found; run `lppl scan` first")
        raise typer.Exit(1)
    s = validation.load_validation_config(vconfig)
    if null_paths is not None:
        s = s.model_copy(update={"null_model": s.null_model.model_copy(update={"paths": null_paths})})
    run_label = label if run_id is None else (db.list_runs(eng).set_index("run_id").loc[rid, "label"] or "")
    space = None
    if tuning_config is not None:
        from lppl.tuning.settings import load_tuning_config
        space = load_tuning_config(tuning_config).indicator.space
    rep = validation.run_validation(eng, rid, run_label, s, run_walk_forward=walk_forward,
                                    run_null=s.null_model.paths > 0, n_jobs=n_jobs, space=space)
    out_dir = out or Path("reports/validation") / f"{run_label.replace(':', '_')}_{rid[:8]}"
    validation.save_report(rep, out_dir)
    if mlflow:
        mlrun = validation.log_to_mlflow(rep, out_dir)
        if mlrun:
            typer.echo(f"MLflow run {mlrun} (experiment lppl-validation)")
    _print_validation(rep.summary)
    typer.echo(f"\nreport: {out_dir}  (notebook: notebooks/05_validation.ipynb)")


def _print_validation(s: dict):
    def f(v, pct=False):
        if v is None:
            return "-"
        return f"{v:.1%}" if pct else f"{v:.3f}"

    typer.echo(f"\n== validation of run {s['run_id'][:8]} ({s['label']}) | event: "
               f"{s['event']['horizon']}d drawdown >= {s['event']['k_sigma']} sigma*sqrt(h) ==")
    typer.echo(f"{'':28}{'all dates':>12}{'validation':>12}")
    for key, name in [("base_rate", "base rate"), ("ap_lppl", "AP LPPLS"), ("ap_momentum", "AP momentum"),
                      ("precision_at_alarm", "precision at alarm"), ("recall_at_alarm", "recall at alarm")]:
        typer.echo(f"{name:28}{f(s.get(f'all.{key}')):>12}{f(s.get(f'valid.{key}')):>12}")
    for key, name in [("lppl_minus_base", "LPPLS - base (95% CI)"),
                      ("lppl_minus_momentum", "LPPLS - momentum (95% CI)")]:
        ci = lambda sp, k=key: s.get(f"{sp}.ci.{k}") or [None, None]  # noqa: E731
        typer.echo(f"{name:28}{'[' + f(ci('all')[0]) + ', ' + f(ci('all')[1]) + ']':>12}"
                   f"  {'[' + f(ci('valid')[0]) + ', ' + f(ci('valid')[1]) + ']':>12}")
    typer.echo(f"placebo p-value: LPPLS {f(s.get('placebo.conf.p_value'))}, "
               f"momentum {f(s.get('placebo.momentum.p_value'))}")
    if "timing.episodes_complete" in s:
        typer.echo(f"timing: {s['timing.episodes_complete']} episodes, median lead to top "
                   f"{s['timing.median_lead_trading_days']:.0f} trading days, top in tc window "
                   f"{s['timing.top_in_tc_window']}/{s['timing.episodes_complete']}")
    if "walk_forward.mean_ap_stored_test" in s:
        typer.echo("walk-forward mean test AP: base " + f(s.get("walk_forward.mean_base_rate_test"))
                   + " | momentum " + f(s.get("walk_forward.mean_ap_momentum_test"))
                   + " | LPPLS " + f(s.get("walk_forward.mean_ap_stored_test"))
                   + " | LPPLS re-tuned " + f(s.get("walk_forward.mean_ap_tuned_test")))
    if "null.alarm_rate_null" in s:
        typer.echo(f"alarm rate: real {f(s['null.alarm_rate_real'], True)} vs bubble-free paths "
                   f"{f(s['null.alarm_rate_null'], True)} (one-sided p = {f(s.get('null.p_value_real_gt_null'))})")
    for strat in ("buy_and_hold", "lppl", "momentum", "lppl_placebo"):
        k = f"backtest.all.{strat}"
        typer.echo(f"backtest {strat:14} return {f(s.get(k + '.ann_return'), True):>7}  "
                   f"sharpe {f(s.get(k + '.sharpe')):>6}  max DD {f(s.get(k + '.max_drawdown'), True):>7}")
    typer.echo("verdict: " + ", ".join(f"{k}={v}" for k, v in s["verdict"].items()))


@app.command()
def combine(
    label: Annotated[str, typer.Option(help="long-scale scan label (latest finished run)")] = "bubbles",
    short_label: Annotated[str | None, typer.Option(help="short-scale scan label; omit to skip")] = None,
    run_id: Annotated[str | None, typer.Option(help="long-scale run instead of --label")] = None,
    cconfig: Annotated[Path | None, typer.Option("--combined-config",
                                                 help="default: config/combined.yaml")] = None,
    first_test_year: Annotated[int | None, typer.Option(help="override first_test_year")] = None,
    out: Annotated[Path | None, typer.Option(help="default: reports/combined/<label>_<run>")] = None,
):
    """Refined LPPLS signals + a walk-forward combined model: does LPPLS add information beyond
    market features? (docs/combined_model.md)"""
    _setup()
    from lppl import combined, pipeline

    eng = _engine()
    rid = pipeline.resolve_run(eng, run_id, label)
    if rid is None:
        typer.echo(f"no finished run for label {label!r}; run `lppl scan` first")
        raise typer.Exit(1)
    srid = pipeline.resolve_run(eng, None, short_label) if short_label else None
    if short_label and srid is None:
        typer.echo(f"no finished run for short label {short_label!r}")
        raise typer.Exit(1)
    s = combined.load_combined_config(cconfig)
    if first_test_year is not None:
        s = s.model_copy(update={"first_test_year": first_test_year})
    rep = combined.run_combined(eng, rid, label, srid, s)
    out_dir = out or Path("reports/combined") / f"{label}_{rid[:8]}"
    combined.save_report(rep, out_dir)
    pd.set_option("display.width", 200)
    met = rep.tables["metrics"].set_index(["label", "model"])
    typer.echo(met[["n", "base_rate", "log_loss", "ap", "precision_at_alarm"]].round(4).to_string())
    inc = rep.tables["increments"]
    typer.echo("\n" + inc[["label", "model", "versus", "log_loss_gain_pct", "ll_ci_low", "ll_ci_high",
                           "p_value", "p_holm", "ap_gain"]].round(4).to_string(index=False))
    typer.echo("\nverdict (Holm, 5 %): " + ", ".join(f"{k}={v}" for k, v in rep.summary["verdict"].items()))
    typer.echo(f"report: {out_dir}  (notebook: notebooks/06_combined_model.ipynb)")


@app.command("compare-reports")
def compare_reports(reports: Annotated[list[Path], typer.Argument(help="validation report folders")],
                    out: Annotated[Path | None, typer.Option(help="also write a Markdown table")] = None):
    """Side-by-side key numbers and verdicts of several validation reports."""
    from lppl import validation

    rows = {}
    for d in reports:
        s = validation.load_report(d).summary
        rows[f"{s['label']} ({s['run_id'][:8]})"] = s
    keys = [("all.base_rate", "base rate (all)"), ("all.ap_lppl", "AP LPPLS (all)"),
            ("valid.base_rate", "base rate (2024+)"), ("valid.ap_lppl", "AP LPPLS (2024+)"),
            ("valid.ap_momentum", "AP momentum (2024+)"), ("all.precision_at_alarm", "precision at alarm"),
            ("all.alarm_rate_at_alarm", "alarm rate"), ("placebo.conf.p_value", "placebo p-value"),
            ("timing.median_lead_trading_days", "median lead to top (days)"),
            ("timing.top_in_tc_window", "tops inside tc window"),
            ("walk_forward.mean_ap_stored_test", "walk-forward AP"),
            ("null.alarm_rate_real", "alarm rate real (null grid)"),
            ("null.alarm_rate_null", "alarm rate bubble-free"),
            ("backtest.all.lppl.ann_return", "backtest LPPLS return"),
            ("backtest.all.buy_and_hold.ann_return", "backtest buy & hold return"),
            ("backtest.all.lppl.sharpe", "backtest LPPLS Sharpe")]
    names = sorted({k.split(".")[1] for s in rows.values() for k in s if k.startswith("labels.")})
    for n in names:
        keys += [(f"labels.{n}.base_rate", f"{n}: base rate"), (f"labels.{n}.ap_lppl", f"{n}: AP LPPLS"),
                 (f"labels.{n}.ap_momentum", f"{n}: AP momentum"), (f"labels.{n}.p_holm", f"{n}: p (Holm)")]
    if any("tc_window.hit_rate" in s for s in rows.values()):
        keys += [("tc_window.episodes", "tc test: episodes"), ("tc_window.hit_rate", "tc test: hit rate"),
                 ("tc_window.placebo_mean", "tc test: placebo hit rate"),
                 ("tc_window.p_holm", "tc test: p (Holm)")]
    df = pd.DataFrame({name: {label: s.get(k) for k, label in keys} for name, s in rows.items()})
    verdicts = pd.DataFrame({name: s["verdict"] for name, s in rows.items()}).replace(
        {True: "yes", False: "no", None: "-"})
    pd.set_option("display.width", 200)
    typer.echo(df.round(4).to_string())
    typer.echo("\nverdict (5 % level):")
    typer.echo(verdicts.to_string())
    if out:
        out.parent.mkdir(parents=True, exist_ok=True)
        out.write_text("## Key numbers\n\n" + _md_table(df.round(4)) + "\n\n## Verdict (5 % level)\n\n"
                       + _md_table(verdicts) + "\n")
        typer.echo(f"wrote {out}")


def _md_table(df: pd.DataFrame) -> str:
    cols = [""] + [str(c) for c in df.columns]
    lines = ["| " + " | ".join(cols) + " |", "|" + "---|" * len(cols)]
    for idx, row in df.iterrows():
        lines.append("| " + " | ".join([str(idx)] + ["-" if pd.isna(v) else str(v) for v in row]) + " |")
    return "\n".join(lines)


@app.command("fit-one")
def fit_one(ticker: str, end: Annotated[dt.datetime, typer.Option(formats=["%Y-%m-%d"])],
            window: Annotated[int, typer.Option(help="window length in trading days")] = 500,
            config: ConfigOpt = None, seed: int = 0):
    """Fit one window ending at END and print parameters + filter checks (no DB write)."""
    from lppl.data import db
    from lppl.model.calibrate import fit_window
    from lppl.model.filters import check
    from lppl.model.lppls import dates_to_years, years_to_date
    cfg = load_config(config)
    px = db.load_prices(_engine(), ticker, end=end.date())[cfg.data.price_field].dropna()
    px = px.iloc[-window:]
    res = fit_window(dates_to_years(px.index), np.log(px.to_numpy()), bounds=cfg.bounds,
                     ga_config=cfg.ga, rng=np.random.default_rng(seed))
    out = res.fit.to_dict() | {"tc_date": str(years_to_date(res.fit.tc).date()),
                               "n_evals": res.n_evals, "elapsed_ms": round(res.elapsed_ms, 1)}
    out["checks"] = check(res.fit, cfg.filters)
    typer.echo(json.dumps(out, indent=2, default=float))


from lppl.tuning.cli import app as tune_app  # noqa: E402  (light: heavy imports are lazy)

app.add_typer(tune_app, name="tune")


if __name__ == "__main__":
    app()
