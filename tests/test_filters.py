import math

import pytest

from lppl.model.filters import FilterConfig, check, is_qualified
from lppl.model.lppls import LPPLSFit


def make(**kw) -> LPPLSFit:
    base = dict(t1=2020.0, t2=2022.0, tc=2022.1, m=0.5, omega=10.0,
                A=5.0, B=-1.0, C1=0.01, C2=0.0, sse=0.1, n_obs=500)
    base.update(kw)
    return LPPLSFit(**base)


def test_typical_bubble_fit_is_qualified():
    fit = make()
    c = check(fit)
    assert c["qualified"], c
    assert fit.sign == 1


@pytest.mark.parametrize("m, ok", [(0.099, False), (0.1, True), (0.9, True), (0.901, False)])
def test_m_bounds(m, ok):
    assert check(make(m=m))["m"] is ok


@pytest.mark.parametrize("omega, ok", [(1.99, False), (2.0, True), (25.0, True), (25.01, False)])
def test_omega_bounds(omega, ok):
    assert check(make(omega=omega))["omega"] is ok


@pytest.mark.parametrize("tc, ok", [(2021.89, False), (2021.9, True), (2022.4, True), (2022.41, False)])
def test_tc_bounds(tc, ok):
    # dt = 2 years -> allowed [t2 - 0.1, t2 + 0.4]
    assert check(make(tc=tc))["tc"] is ok


def test_damping_boundary():
    # D = m|B|/(omega|C|) = 0.5*|B|/(10*0.01) -> D = 0.5 at |B| = 0.1
    assert check(make(B=-0.1))["damping"]
    assert not check(make(B=-0.0999))["damping"]


def test_oscillations_boundary():
    cfg = FilterConfig()
    fit = make()
    # choose omega so that O is exactly the threshold
    ratio = math.log((fit.tc - fit.t1) / (fit.tc - fit.t2))
    w = cfg.oscillations_min * math.pi / ratio
    assert check(make(omega=w * 1.0001), cfg)["oscillations"]
    assert not check(make(omega=w * 0.999), cfg)["oscillations"]


def test_oscillations_undefined_when_tc_before_t2():
    c = check(make(tc=2021.95))
    assert c["oscillations"] is False
    assert not c["qualified"]


def test_negative_bubble_sign():
    fit = make(B=1.0)
    assert fit.sign == -1
    assert is_qualified(fit)


def test_optional_rmse_filter():
    fit = make(sse=0.5, n_obs=500)  # rmse ~ 0.0316
    assert check(fit, FilterConfig(max_rmse=0.05))["rmse"]
    assert not check(fit, FilterConfig(max_rmse=0.01))["rmse"]


def test_tc_pinned_at_search_bound_is_rejected():
    # GA lower bound is t2 + 1 day; such "already critical" fits must not qualify
    c = check(make(tc=2022.0 + 1 / 365.25))
    assert c["tc"] and not c["tc_margin"] and not c["qualified"]
    assert check(make(tc=2022.0 + 4 / 365.25))["tc_margin"]


def test_vectorised_qualification_matches_scalar_check():
    import numpy as np

    from lppl.model.filters import qualify_arrays

    rng = np.random.default_rng(0)
    n = 3000
    t1 = 2020.0 + rng.random(n)
    t2 = t1 + rng.uniform(0.5, 3.0, n)
    tc = t2 + rng.uniform(-0.1, 1.0, n)
    tc[:20] = t2[:20] + 1 / 365.25  # pinned at the search bound
    m, omega = rng.uniform(0, 1, n), rng.uniform(1, 50, n)
    B, C1, C2 = rng.normal(0, 1, n), rng.normal(0, 0.05, n), rng.normal(0, 0.05, n)
    C1[:5] = C2[:5] = 0.0  # infinite damping
    sse = rng.uniform(0, 1, n)
    sse[5:8] = np.inf
    n_obs = rng.integers(125, 750, n)
    for cfg in (FilterConfig(), FilterConfig(m_min=0.01, tc_margin_days=0, max_rmse=0.03)):
        vec = qualify_arrays(t1=t1, t2=t2, tc=tc, m=m, omega=omega, B=B, C1=C1, C2=C2, sse=sse,
                             n_obs=n_obs, cfg=cfg)
        ref = [check(LPPLSFit(t1=t1[i], t2=t2[i], tc=tc[i], m=m[i], omega=omega[i], A=0.0, B=B[i],
                              C1=C1[i], C2=C2[i], sse=sse[i], n_obs=int(n_obs[i])), cfg)["qualified"]
               for i in range(n)]
        assert vec.tolist() == ref
        assert 0 < vec.sum() < n
