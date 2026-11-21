"""Databases for the tuning services, inside the same PostgreSQL instance.

* ``mlflow``: MLflow tracking backend (experiments, runs, params, metrics)
* ``optuna``: Optuna RDB storage (studies are resumable and visible in optuna-dashboard)
"""

from __future__ import annotations

import os

from sqlalchemy import create_engine, text
from sqlalchemy.engine import make_url

from lppl.config import DBSettings

TUNING_DATABASES = ("mlflow", "optuna")


def database_url(dbname: str) -> str:
    return make_url(DBSettings().url).set(database=dbname).render_as_string(hide_password=False)


def ensure_databases(names=TUNING_DATABASES) -> list[str]:
    """CREATE DATABASE for each missing name (idempotent). Returns the ones created."""
    admin = create_engine(DBSettings().url, isolation_level="AUTOCOMMIT")
    created = []
    with admin.connect() as conn:
        for name in names:
            exists = conn.execute(text("select 1 from pg_database where datname = :n"), {"n": name}).scalar()
            if not exists:
                conn.execute(text(f'create database "{name}"'))
                created.append(name)
    admin.dispose()
    return created


def optuna_storage_url() -> str:
    return os.environ.get("LPPL_OPTUNA_STORAGE") or database_url("optuna")


def mlflow_tracking_uri() -> str:
    """``MLFLOW_TRACKING_URI`` (set in the containers), else the compose server on the host."""
    return os.environ.get("MLFLOW_TRACKING_URI") or "http://localhost:5050"


def init_optuna_schema() -> None:
    """Create Optuna's tables once, up front. optuna-dashboard's own first-time schema
    creation on an empty database fails (``version_info`` missing during its commit)."""
    import optuna

    optuna.storages.RDBStorage(optuna_storage_url())


if __name__ == "__main__":  # run by the mlflow container before starting the server
    print("created:", ensure_databases() or "nothing (already present)")
    init_optuna_schema()
    print("optuna schema ready")
