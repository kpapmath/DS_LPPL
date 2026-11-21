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
