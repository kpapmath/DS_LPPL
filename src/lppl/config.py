"""Configuration: a YAML experiment file plus environment-based DB settings."""

from __future__ import annotations

import datetime as dt
from pathlib import Path
from typing import Literal

import yaml
from pydantic import BaseModel, Field
from pydantic_settings import BaseSettings, SettingsConfigDict

from lppl.model.calibrate import SearchBounds
from lppl.model.filters import FilterConfig
from lppl.model.indicator import RobustConfig, ScanSettings, WindowConfig
from lppl.optim.ga import GAConfig

PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "config" / "default.yaml"


class Instrument(BaseModel):
    ticker: str
    name: str
    group: Literal["index", "mag7", "extra", "bubble"]
    asset_type: Literal["index", "equity", "etf", "crypto", "commodity"] = "equity"


class DataConfig(BaseModel):
    start: dt.date = dt.date(1995, 1, 1)
    end: dt.date | None = None  # None = today
    price_field: Literal["adj_close", "close"] = "adj_close"
    snapshot_dir: Path = Path("data/snapshots")
    instruments: list[Instrument]

    @property
    def tickers(self) -> list[str]:
        return [i.ticker for i in self.instruments]


class ScanConfig(BaseModel):
    start: dt.date | None = None
    end: dt.date | None = None
    step: int = Field(10, ge=1, description="trading days between analysis dates")
    n_jobs: int = -1
    method: Literal["ga", "de", "nm"] = "ga"


class AppConfig(BaseModel):
    seed: int = 42
    data: DataConfig
    scan: ScanConfig = ScanConfig()
    windows: WindowConfig = WindowConfig()
    bounds: SearchBounds = SearchBounds()
    filters: FilterConfig = FilterConfig()
    ga: GAConfig = GAConfig()
    robust: RobustConfig = RobustConfig()

    def scan_settings(self) -> ScanSettings:
        return ScanSettings(windows=self.windows, bounds=self.bounds, filters=self.filters,
                            ga=self.ga, method=self.scan.method, robust=self.robust)


def load_config(path: str | Path | None = None) -> AppConfig:
    path = Path(path) if path else DEFAULT_CONFIG
    with open(path) as fh:
        raw = yaml.safe_load(fh) or {}
    return AppConfig.model_validate(raw)


class DBSettings(BaseSettings):
    """Database connection. ``LPPL_DATABASE_URL`` wins over the POSTGRES_* parts."""

    model_config = SettingsConfigDict(env_file=".env", extra="ignore")

    lppl_database_url: str | None = None
    postgres_user: str = "lppl"
    postgres_password: str = "lppl"
    postgres_db: str = "lppl"
    postgres_host: str = "localhost"
    postgres_port: int = 5442

    @property
    def url(self) -> str:
        if self.lppl_database_url:
            return self.lppl_database_url
        return (f"postgresql+psycopg://{self.postgres_user}:{self.postgres_password}"
                f"@{self.postgres_host}:{self.postgres_port}/{self.postgres_db}")
