"""Orchestration: ingest -> scan -> store -> report."""

from __future__ import annotations

import logging
import os
import subprocess
from pathlib import Path

import pandas as pd
from sqlalchemy.engine import Engine

from lppl import __version__
from lppl.config import PROJECT_ROOT, AppConfig
from lppl.data import db, yahoo
from lppl.model.indicator import scan_series

log = logging.getLogger(__name__)


def git_sha() -> str | None:
    """Commit of the code: ``LPPL_GIT_SHA`` (set by the Makefile for containers, where
    ``.git`` is not available), else ``git rev-parse HEAD``."""
    if os.environ.get("LPPL_GIT_SHA"):
        return os.environ["LPPL_GIT_SHA"]
    try:
        out = subprocess.run(["git", "rev-parse", "--verify", "-q", "HEAD"], cwd=PROJECT_ROOT, capture_output=True,
                             text=True, timeout=5, check=True)
        return out.stdout.strip() or None
    except Exception:
        return None


def _snapshot_dir(cfg: AppConfig) -> Path:
    p = cfg.data.snapshot_dir
    return p if p.is_absolute() else PROJECT_ROOT / p


def ingest(cfg: AppConfig, engine: Engine, from_snapshot: bool = False,
           tickers: list[str] | None = None) -> pd.DataFrame:
    """Download (or read snapshots), validate, upsert into Postgres, save snapshots."""
    import yfinance

    db.init_db(engine)
    selected = [i for i in cfg.data.instruments if not tickers or i.ticker in tickers]
    db.upsert_instruments(engine, selected)
    snap_dir = _snapshot_dir(cfg)
    summary = []
    for inst in selected:
        if from_snapshot:
            df, sha = yahoo.load_snapshot(snap_dir, inst.ticker)
            source = "snapshot"
        else:
            df = yahoo.download(inst.ticker, cfg.data.start, cfg.data.end)
            _, sha = yahoo.save_snapshot(df, snap_dir, inst.ticker)
            source = "yahoo"
        rep = yahoo.validate(df)
        n = db.upsert_prices(engine, inst.ticker, df)
        db.log_ingestion(engine, ticker=inst.ticker, source=source, n_rows=n,
                         first_date=rep.first_date.date(), last_date=rep.last_date.date(),
                         sha256=sha, yfinance_version=yfinance.__version__,
                         issues="; ".join(rep.issues) or None)
        log.info("%-6s %5d rows %s..%s %s", inst.ticker, n, rep.first_date.date(),
                 rep.last_date.date(), "; ".join(rep.issues))
        summary.append({"ticker": inst.ticker, "rows": n, "first": rep.first_date.date(),
                        "last": rep.last_date.date(), "sha256": sha[:12],
                        "issues": "; ".join(rep.issues)})
    return pd.DataFrame(summary)


def scan(cfg: AppConfig, engine: Engine, tickers: list[str] | None = None, label: str | None = None,
         progress: bool = True) -> str:
    """Run the multi-window LPPLS scan for each ticker and store fits + indicator."""
    db.init_db(engine)
    tickers = tickers or cfg.data.tickers
    run_id = db.create_run(engine, cfg.model_dump(mode="json") | {"tickers": tickers},
                           seed=cfg.seed, label=label, git_sha=git_sha(),
                           package_version=__version__)
    settings = cfg.scan_settings()
    try:
        for ticker in tickers:
            px = db.load_prices(engine, ticker, start=cfg.data.start, end=cfg.data.end)
            if px.empty:
                raise RuntimeError(f"no prices for {ticker}; run `lppl ingest` first")
            fits, ind = scan_series(px[cfg.data.price_field], ticker, settings,
                                    start=cfg.scan.start, end=cfg.scan.end, step=cfg.scan.step,
                                    master_seed=cfg.seed, n_jobs=cfg.scan.n_jobs,
                                    progress=progress)
            db.write_fits(engine, run_id, fits)
            db.write_indicator(engine, run_id, ind)
            log.info("%-6s %d fits, %d analysis dates", ticker, len(fits), len(ind))
    except BaseException:
        db.finish_run(engine, run_id, status="failed")
        raise
    db.finish_run(engine, run_id)
    return run_id


def requalify(cfg: AppConfig, engine: Engine, run_id: str) -> int:
    """Re-apply ``cfg.filters`` to the stored fits of ``run_id`` and rebuild its indicator.

    Fits (GA output) are independent of the filters, so the result equals a fresh scan with
    the new filters. The run's stored config is updated so it stays truthful."""
    from sqlalchemy import update

    from lppl.model.indicator import aggregate
    from lppl.model.indicator import requalify as _requalify

    fits = db.load_fits(engine, run_id)
    if fits.empty:
        raise RuntimeError(f"run {run_id} has no fits")
    fits["t1"] = fits["t1"].dt.date
    fits["t2"] = fits["t2"].dt.date
    fits["tc_date"] = fits["tc_date"].dt.date
    new = _requalify(fits, cfg.filters)
    changed = int((new["qualified"] != fits["qualified"]).sum())
    db.write_fits(engine, run_id, new.drop(columns=["id", "run_id"]))
    with engine.begin() as conn:
        conn.execute(db.bubble_indicator.delete().where(db.bubble_indicator.c.run_id == run_id))
        old_cfg = conn.execute(db.fit_runs.select().where(db.fit_runs.c.run_id == run_id)).one().config
        old_cfg["filters"] = cfg.filters.__dict__ | {}
        conn.execute(update(db.fit_runs).where(db.fit_runs.c.run_id == run_id)
                     .values(config=old_cfg))
    db.write_indicator(engine, run_id, aggregate(new))
    return changed


def resolve_run(engine: Engine, run_id: str | None = None, label: str | None = "default") -> str | None:
    """Explicit run id, else the latest finished run with ``label``, else the latest finished run."""
    if run_id:
        return run_id
    return (label and db.latest_run_for_label(engine, label)) or db.latest_run_id(engine)


def report(engine: Engine, run_id: str | None = None, label: str | None = "default") -> pd.DataFrame:
    """Latest indicator reading per ticker for one run (see :func:`resolve_run`)."""
    run_id = resolve_run(engine, run_id, label)
    if run_id is None:
        return pd.DataFrame()
    ind = db.load_indicator(engine, run_id)
    if ind.empty:
        return ind
    last = ind.sort_values("date").groupby("ticker").tail(1)
    # Also show the maximum confidence over the last 60 calendar days.
    recent = ind[ind["date"] >= ind.groupby("ticker")["date"].transform("max") - pd.Timedelta(days=60)]
    peak = recent.groupby("ticker")[["pos_conf", "neg_conf"]].max().add_suffix("_60d_max")
    out = last.set_index("ticker")[["date", "pos_conf", "neg_conf", "n_qualified", "n_windows",
                                    "tc_median_date", "tc_q25_date", "tc_q75_date"]]
    return out.join(peak).sort_values("pos_conf", ascending=False)
