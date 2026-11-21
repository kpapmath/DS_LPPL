"""Tuning configuration (config/tuning.yaml), search spaces and parameter mapping."""

from __future__ import annotations

import dataclasses
import datetime as dt
from pathlib import Path
from typing import Any

import yaml
from pydantic import BaseModel

from lppl.config import PROJECT_ROOT, AppConfig
from lppl.model.filters import FilterConfig
from lppl.model.indicator import WindowConfig
from lppl.optim.ga import GAConfig

DEFAULT_TUNING_CONFIG = PROJECT_ROOT / "config" / "tuning.yaml"
STUDIES = ("ga", "indicator")


class BenchmarkSettings(BaseModel):
    n_real: int = 40
    n_synthetic: int = 8
    lengths: list[int] = [125, 250, 500, 750]
    start: dt.date = dt.date(2019, 1, 1)
    seed: int = 7
    oracle_restarts: int = 3


class GAStudySettings(BaseModel):
    benchmark: BenchmarkSettings = BenchmarkSettings()
    seeds_per_window: int = 2
    chunks: int = 4
    cost_weight: float = 0.5
    space: dict[str, Any]


class IndicatorStudySettings(BaseModel):
    source_label: str = "default"
    horizon: int = 63
    k_sigma: float = 1.5
    vol_window: int = 750
    split_date: dt.date = dt.date(2024, 1, 1)
    embargo_days: int = 63
    min_windows: int = 5
    space: dict[str, Any]


class TuningConfig(BaseModel):
    seed: int = 0
    mlflow_experiment_prefix: str = "lppl"
    ga: GAStudySettings
    indicator: IndicatorStudySettings

    def experiment(self, study: str) -> str:
        return f"{self.mlflow_experiment_prefix}-{study}"


def load_tuning_config(path: str | Path | None = None) -> TuningConfig:
    with open(Path(path) if path else DEFAULT_TUNING_CONFIG) as fh:
        return TuningConfig.model_validate(yaml.safe_load(fh))


# --------------------------------------------------------------------------- spaces
def suggest(trial, space: dict[str, Any]) -> dict[str, Any]:
    """Sample one configuration from ``space`` with an Optuna trial (define-by-run).

    The same function drives native Optuna studies and Ray Tune's ``OptunaSearch``.
    """
    params: dict[str, Any] = {}
    for name, spec in space.items():
        if isinstance(spec, dict) and "when" in spec and not params.get(spec["when"], False):
            continue
        params[name] = _suggest_one(trial, name, spec)
    return params


def _suggest_one(trial, name: str, spec):
    if isinstance(spec, dict):
        if "choices" in spec:
            return trial.suggest_categorical(name, list(spec["choices"]))
        low, high = spec["low"], spec["high"]
        step, log = spec.get("step"), bool(spec.get("log", False))
    else:
        low, high = spec[0], spec[1]
        log = len(spec) > 2 and str(spec[2]).lower() == "log"
        step = None
    if isinstance(low, int) and isinstance(high, int) and not isinstance(low, bool):
        return trial.suggest_int(name, low, high, step=step or 1, log=log)
    return trial.suggest_float(name, float(low), float(high), step=step, log=log)


def coerce(params: dict[str, Any], space: dict[str, Any]) -> dict[str, Any]:
    """Restore the declared types of sampled values. Ray's result table stores parameters
    in pandas columns, which turns ints into floats when some trials lack a conditional
    parameter."""
    out = dict(params)
    for name, spec in space.items():
        if name not in out:
            continue
        v = out[name]
        if isinstance(spec, dict) and "choices" in spec:
            match = [c for c in spec["choices"] if c == v]
            out[name] = match[0] if match else v
            continue
        low, high = (spec["low"], spec["high"]) if isinstance(spec, dict) else (spec[0], spec[1])
        if isinstance(low, int) and isinstance(high, int) and not isinstance(low, bool):
            out[name] = int(round(float(v)))
        else:
            out[name] = float(v)
    return out


def validate_space(space: dict[str, Any]) -> None:
    """Fail early on malformed specs (runs every spec through a dummy trial)."""

    class _Probe:
        def suggest_categorical(self, name, choices):
            assert choices, name
            return choices[0]

        def suggest_int(self, name, low, high, step=1, log=False):
            assert low <= high, name
            return low

        def suggest_float(self, name, low, high, step=None, log=False):
            assert low <= high and (not log or low > 0), name
            return low

    suggest(_Probe(), space)


# --------------------------------------------------------------------------- mapping
_GA_FIELDS = {f.name for f in dataclasses.fields(GAConfig)}
_FILTER_FIELDS = {f.name for f in dataclasses.fields(FilterConfig)}


def ga_config(params: dict[str, Any], base: GAConfig | None = None) -> GAConfig:
    base = base or GAConfig()
    upd = {k: v for k, v in params.items() if k in _GA_FIELDS}
    if not upd.get("polish", base.polish):
        upd.setdefault("polish_maxiter", base.polish_maxiter)
    return dataclasses.replace(base, **upd)


def filter_config(params: dict[str, Any], base: FilterConfig | None = None) -> FilterConfig:
    base = base or FilterConfig()
    return dataclasses.replace(base, **{k: v for k, v in params.items() if k in _FILTER_FIELDS})


def window_config(params: dict[str, Any], base: WindowConfig | None = None,
                  grid_step: int | None = None) -> WindowConfig:
    """``window_stride`` k selects every k-th length of the stored window grid, whose
    spacing ``grid_step`` is the ``windows.step`` of the scan the fits come from."""
    base = base or WindowConfig()
    grid_step = grid_step or base.step
    return WindowConfig(
        min_len=int(params.get("window_min_len", base.min_len)),
        max_len=int(params.get("window_max_len", base.max_len)),
        step=int(params["window_stride"]) * grid_step if "window_stride" in params else base.step,
    )


def apply_params(cfg: AppConfig, study: str, params: dict[str, Any],
                 grid_step: int | None = None) -> AppConfig:
    """Return a copy of the experiment config with the tuned parameters applied."""
    if study == "ga":
        return cfg.model_copy(update={"ga": ga_config(params, cfg.ga)})
    if study == "indicator":
        return cfg.model_copy(update={"filters": filter_config(params, cfg.filters),
                                      "windows": window_config(params, cfg.windows, grid_step)})
    raise ValueError(f"unknown study {study!r}")


def default_params(cfg: AppConfig, study: str, space: dict[str, Any],
                   grid_step: int | None = None) -> dict[str, Any]:
    """A configuration expressed in the study's parameter names. It is enqueued as the
    first trial, so every study measures its improvement over the status quo. For the
    indicator study ``cfg`` is the source scan's own config and ``grid_step`` its window
    spacing."""
    if study == "ga":
        src = dataclasses.asdict(cfg.ga)
    else:
        grid_step = grid_step or cfg.windows.step
        src = dataclasses.asdict(cfg.filters) | {
            "window_min_len": cfg.windows.min_len, "window_max_len": cfg.windows.max_len,
            "window_stride": max(1, cfg.windows.step // grid_step)}
    out = {k: src[k] for k in space if k in src}
    if study == "ga" and not out.get("polish", True):
        out.pop("polish_maxiter", None)
    return out


def config_to_yaml(cfg: AppConfig, header: str = "") -> str:
    body = yaml.safe_dump(cfg.model_dump(mode="json"), sort_keys=False, allow_unicode=True)
    lines = "".join(f"# {h}\n" for h in header.splitlines()) if header else ""
    return lines + body
