"""Tuning layer: search spaces, mapping, labels, metrics, objectives and the Optuna runner.

Optuna is needed for the sampler/runner tests (``pip install optuna``); they are skipped
otherwise. Ray and MLflow are never needed here: tracking is stubbed.
"""

import dataclasses

import numpy as np
import pandas as pd
import pytest
import yaml

from lppl.config import AppConfig, load_config
from lppl.model.filters import FilterConfig
from lppl.model.indicator import aggregate, requalify
from lppl.tuning.detection import (
    DetectionData,
    average_precision,
    build_detection_data,
    confidence,
    drawdown_labels,
    evaluate_detector,
)
from lppl.tuning.settings import (
    IndicatorStudySettings,
    apply_params,
    config_to_yaml,
    default_params,
    filter_config,
    ga_config,
    load_tuning_config,
    suggest,
    validate_space,
    window_config,
)

TCFG = load_tuning_config()


# --------------------------------------------------------------------------- spaces
def test_tuning_yaml_spaces_are_valid():
    validate_space(TCFG.ga.space)
    validate_space(TCFG.indicator.space)


@pytest.mark.parametrize("bad", [{"x": [2, 1]}, {"x": [0.0, 1.0, "log"]}, {"x": {"choices": []}}])
def test_validate_space_rejects_bad_specs(bad):
    with pytest.raises(AssertionError):
        validate_space(bad)


def test_suggest_respects_ranges_types_and_conditions():
    optuna = pytest.importorskip("optuna")
    study = optuna.create_study(sampler=optuna.samplers.RandomSampler(seed=0))
    seen_polish = set()
    for _ in range(60):
        trial = study.ask()
        p = suggest(trial, TCFG.ga.space)
        assert 10 <= p["population_size"] <= 120 and isinstance(p["population_size"], int)
        assert 0.01 <= p["mutation_sigma"] <= 0.4
        assert ("polish_maxiter" in p) == p["polish"]  # conditional parameter
        if p["polish"]:
            assert p["polish_maxiter"] % 25 == 0
        seen_polish.add(p["polish"])
        ga_config(p)  # always a valid GAConfig
        study.tell(trial, 0.0)
    assert seen_polish == {True, False}


# --------------------------------------------------------------------------- mapping
def test_param_mapping_round_trip():
    cfg = load_config()
    for study, space in (("ga", TCFG.ga.space), ("indicator", TCFG.indicator.space)):
        d = default_params(cfg, study, space)
        assert set(d) <= set(space)
        assert apply_params(cfg, study, d) == cfg  # status quo maps back onto itself


def test_window_and_filter_mapping():
    w = window_config({"window_min_len": 150, "window_max_len": 700, "window_stride": 2})
    assert (w.min_len, w.max_len, w.step) == (150, 700, 50)
    f = filter_config({"m_min": 0.2, "not_a_filter": 1})
    assert f.m_min == 0.2 and f.m_max == FilterConfig().m_max
    g = ga_config({"population_size": 77, "polish": False})
    assert g.population_size == 77 and g.polish is False


def test_tuned_yaml_loads_back(tmp_path):
    cfg = apply_params(load_config(), "indicator", {"m_min": 0.25, "window_stride": 2}, grid_step=25)
    path = tmp_path / "tuned.yaml"
    path.write_text(config_to_yaml(cfg, "header line"))
    back = AppConfig.model_validate(yaml.safe_load(path.read_text()))
    assert back == cfg and back.filters.m_min == 0.25 and back.windows.step == 50


# --------------------------------------------------------------------------- metrics
def _ap_bruteforce(y, s):
    """Average precision by definition: mean precision at each positive's rank (ties
    resolved pessimistically = sklearn's step-wise convention)."""
    y, s = np.asarray(y, float), np.asarray(s, float)
    thresholds = np.unique(s)[::-1]
    ap, prev_r = 0.0, 0.0
    for t in thresholds:
        pred = s >= t
        tp = (pred & (y == 1)).sum()
        p, r = tp / pred.sum(), tp / y.sum()
        ap += (r - prev_r) * p
        prev_r = r
    return ap


def test_average_precision_known_cases_and_bruteforce():
    assert average_precision([1, 1, 0, 0], [0.9, 0.8, 0.2, 0.1]) == pytest.approx(1.0)
    assert average_precision([0, 0, 1, 1], [0.9, 0.8, 0.2, 0.1]) == pytest.approx((1 / 3 + 2 / 4) / 2)
    assert average_precision([1, 0, 1, 0], [0.5, 0.5, 0.5, 0.5]) == pytest.approx(0.5)  # all tied
    assert np.isnan(average_precision([0, 0], [0.1, 0.2]))
    rng = np.random.default_rng(0)
    for _ in range(20):
        y = (rng.random(200) < 0.15).astype(float)
        s = np.round(rng.random(200), 1)  # many ties
        assert average_precision(y, s) == pytest.approx(_ap_bruteforce(y, s))


# --------------------------------------------------------------------------- labels
def _prices(values, start="2020-01-01"):
    return pd.Series(values, index=pd.bdate_range(start, periods=len(values)))


def test_drawdown_labels_detect_crash_and_ignore_calm():
    rng = np.random.default_rng(1)
    calm = 100 * np.exp(np.cumsum(0.01 * rng.normal(size=900)))
    crash = calm.copy()
    crash[820:] *= 0.6  # -51% log drawdown at day 820
    dates = _prices(calm).index[[800, 700]]
    lab_crash = drawdown_labels(_prices(crash), dates, horizon=63, k_sigma=1.5, vol_window=500)
    lab_calm = drawdown_labels(_prices(calm), dates, horizon=63, k_sigma=1.5, vol_window=500)
    assert lab_crash["label"].tolist() == [1.0, 0.0]
    assert lab_calm.loc[dates[0], "label"] == 0.0


def test_drawdown_labels_have_no_lookahead_in_threshold():
    rng = np.random.default_rng(2)
    a = 100 * np.exp(np.cumsum(0.01 * rng.normal(size=900)))
    b = a.copy()
    b[801:] = b[801:] * np.exp(np.cumsum(0.05 * rng.normal(size=99)))  # wild future only
    d = _prices(a).index[[800]]
    la = drawdown_labels(_prices(a), d, 63, 1.5, 500)
    lb = drawdown_labels(_prices(b), d, 63, 1.5, 500)
    assert la["threshold"].iloc[0] == pytest.approx(lb["threshold"].iloc[0])
    assert la["momentum"].iloc[0] == pytest.approx(lb["momentum"].iloc[0])


def test_drawdown_labels_nan_when_horizon_incomplete():
    p = _prices(100 + np.arange(400.0))
    lab = drawdown_labels(p, p.index[[-10]], horizon=63, k_sigma=1.5, vol_window=250)
    assert lab["label"].isna().all()


# --------------------------------------------------------------------------- detector
@pytest.fixture
def fits_and_prices():
    """Synthetic stored fits for 2 tickers x 30 analysis dates x 4 windows."""
    rng = np.random.default_rng(3)
    dates = pd.bdate_range("2019-01-01", periods=1600)
    prices = pd.DataFrame({t: 100 * np.exp(np.cumsum(0.012 * rng.normal(size=dates.size)))
                           for t in ("AAA", "BBB")}, index=dates)
    rows = []
    for t in prices:
        for p in range(800, 1500, 25):
            for n in (125, 150, 175, 200):
                t2, t1 = dates[p], dates[p - n + 1]
                rows.append({"ticker": t, "t1": t1, "t2": t2, "window_len": n,
                             "tc": 2019 + p / 252 + rng.uniform(0.003, 0.3), "m": rng.uniform(0, 1),
                             "omega": rng.uniform(1, 30), "a": 1.0, "b": rng.normal(-0.3, 0.5),
                             "c1": rng.normal(0, 0.03), "c2": rng.normal(0, 0.03),
                             "sse": rng.uniform(0.01, 1), "qualified": False, "sign": 1})
    fits = pd.DataFrame(rows)
    # use the real decimal-year time of the dates for tc so filters are meaningful
    from lppl.model.lppls import dates_to_years
    fits["tc"] = dates_to_years(fits["t2"]) + rng.uniform(0.003, 0.3, len(fits))
    fits["sign"] = np.sign(-fits["b"]).astype(int)
    settings = IndicatorStudySettings(space=TCFG.indicator.space, split_date="2023-06-01",
                                      horizon=40, vol_window=300, min_windows=2)
    return fits, prices, settings


def test_tuning_confidence_matches_production_path(fits_and_prices):
    fits, prices, settings = fits_and_prices
    data = build_detection_data(fits, prices, settings)
    filt = dataclasses.replace(FilterConfig(), m_min=0.05, oscillations_min=1.0, tc_margin_days=1)
    conf, n_win = confidence(data, filt, 125, 200, 25)
    prod = aggregate(requalify(fits.assign(t1=fits.t1.dt.date, t2=fits.t2.dt.date), filt))
    prod["date"] = pd.to_datetime(prod["date"])
    merged = data.groups.assign(conf=conf).merge(prod, on=["ticker", "date"])
    assert len(merged) == len(data.groups)
    np.testing.assert_allclose(merged["conf"], merged["pos_conf"])
    assert (n_win == 4).all()
    conf2, n2 = confidence(data, filt, 150, 200, 50)  # windows 200, 150 only
    assert (n2 == 2).all()


def test_evaluate_detector_split_and_objective(fits_and_prices):
    fits, prices, settings = fits_and_prices
    data = build_detection_data(fits, prices, settings)
    g = data.groups
    assert set(g["split"]) == {"train", "embargo", "valid"}
    assert g.loc[g.split == "train", "date"].max() < g.loc[g.split == "valid", "date"].min()
    m = evaluate_detector({"window_min_len": 125, "window_max_len": 200}, data, settings)
    for k in ("ap_train", "ap_valid", "lift_train", "objective", "n_windows_min"):
        assert k in m
    assert m["objective"] == m["ap_train"] or m["objective"] == 0.0
    # too few windows -> invalid trial scores 0
    strict = settings.model_copy(update={"min_windows": 10})
    assert evaluate_detector({}, data, strict)["objective"] == 0.0


# --------------------------------------------------------------------------- GA benchmark
def test_evaluate_ga_on_tiny_benchmark():
    from lppl.model.calibrate import SearchBounds
    from lppl.tuning.ga_benchmark import GABenchmark, evaluate_ga, oracle, synthetic_windows
    from lppl.tuning.settings import BenchmarkSettings

    s = BenchmarkSettings(n_real=0, n_synthetic=2, lengths=[150], oracle_restarts=1)
    wins = synthetic_windows(s)
    for w in wins:
        w.ref_sse, w.ref_x = oracle(w, SearchBounds(), 1, 0)
        assert w.ref_sse > 0
    bench = GABenchmark(windows=wins, bounds=SearchBounds())
    calls = []
    settings = TCFG.ga.model_copy(update={"chunks": 2, "seeds_per_window": 1})
    m = evaluate_ga({"population_size": 20, "generations": 10}, bench, settings,
                    report=lambda step, met: calls.append((step, met["loss"])))
    assert [c[0] for c in calls] == [1, 2]
    assert m["mean_excess_pct"] >= 0 and m["mean_evals"] > 0
    assert m["loss"] == pytest.approx(m["mean_excess_pct"] + settings.cost_weight * m["mean_evals"] / 1000)
    again = evaluate_ga({"population_size": 20, "generations": 10}, bench, settings)
    assert again["mean_excess_pct"] == m["mean_excess_pct"]  # common random numbers


# --------------------------------------------------------------------------- runner
class _NullClient:
    def __getattr__(self, name):
        return lambda *a, **k: None


def test_optuna_runner_end_to_end(fits_and_prices, tmp_path, monkeypatch):
    pytest.importorskip("optuna")
    from lppl.tuning import runners, tracking

    for fn in ("log_trial", "log_params", "log_metrics", "log_text", "log_dict",
               "log_dataframe", "log_figure"):
        monkeypatch.setattr(tracking, fn, lambda *a, **k: None)
    monkeypatch.setattr(runners, "TUNING_DIR", tmp_path)
    fits, prices, settings = fits_and_prices
    tcfg = TCFG.model_copy(update={"indicator": settings})
    data = build_detection_data(fits, prices, settings)
    cfg = load_config()
    spec = runners.StudySpec("indicator", "t-study", "objective", "max", settings.space,
                             default_params(cfg, "indicator", settings.space), 1, dict(data.meta))
    storage = f"sqlite:///{tmp_path / 'optuna.db'}"
    table = runners.run_optuna(spec, data, tcfg, n_trials=12, n_jobs=1, storage=storage,
                               experiment="x", parent_id="p", cl=_NullClient())
    assert len(table) == 12 and (table["state"] == "COMPLETE").all()
    assert runners.baseline_row(table, spec) is not None  # trial 0 = status quo
    summary = runners.summarize(spec, table, tcfg, cfg, storage, _NullClient(), "p", data)
    assert (tmp_path / "t-study" / "tuned_config.yaml").exists()
    assert summary["best_metrics"]["ap_train"] >= summary["baseline_metrics"]["ap_train"]
    # resuming the same study appends trials
    more = runners.run_optuna(spec, data, tcfg, n_trials=3, n_jobs=1, storage=storage,
                              experiment="x", parent_id="p", cl=_NullClient())
    assert len(more) == 15


def test_detection_data_is_picklable(fits_and_prices):
    import pickle

    fits, prices, settings = fits_and_prices
    data = build_detection_data(fits, prices, settings)
    assert isinstance(pickle.loads(pickle.dumps(data)), DetectionData)


def test_coerce_restores_declared_types():
    from lppl.tuning.settings import coerce

    p = coerce({"population_size": 91.0, "polish_maxiter": 375.0, "polish": True,
                "blx_alpha": 0.5}, TCFG.ga.space)
    assert p == {"population_size": 91, "polish_maxiter": 375, "polish": True, "blx_alpha": 0.5}
    assert isinstance(p["population_size"], int) and isinstance(p["polish_maxiter"], int)
    q = coerce({"window_stride": 2.0, "window_min_len": 150.0}, TCFG.indicator.space)
    assert q == {"window_stride": 2, "window_min_len": 150} and isinstance(q["window_stride"], int)
