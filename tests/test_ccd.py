"""Every CCD closed form against brute-force enumeration, plus the paper's propositions."""

from __future__ import annotations

import itertools
import math

import numpy as np
import pytest

from nesy_tabfm.ccd import brute_force, fit_log_strength, joint_nll, soft_map_prefers_incoherent, soft_marginal
from nesy_tabfm.ccd import closed_forms as cf
from nesy_tabfm.ccd.core import strength_objective

SEEDS = range(25)


def dist(rng: np.random.Generator, k: int, zero_some: bool = False) -> np.ndarray:
    p = rng.dirichlet(np.full(k, 0.7))
    if zero_some and k > 2:
        p[rng.integers(k)] = 0.0
        p = p / p.sum()
    return p


def log(x: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore"):
        return np.log(x)


def log_prob(marginals, y) -> float:
    with np.errstate(divide="ignore"):
        return float(sum(np.log(m[v]) for m, v in zip(marginals, y)))


def assert_matches(closed, brute, marginals, is_coherent) -> None:
    assert closed.log_qC == pytest.approx(brute.log_qC, abs=1e-9, rel=1e-9)
    if brute.map is None:
        assert closed.map is None
        return
    for got, want in zip(closed.marginals, brute.marginals):
        np.testing.assert_allclose(got, want, atol=1e-9)
    # ties can make several assignments optimal, so compare the MAP's probability
    assert is_coherent(closed.map)
    assert log_prob(marginals, closed.map) == pytest.approx(log_prob(marginals, brute.map), abs=1e-9)


@pytest.mark.parametrize("seed", SEEDS)
def test_domain(seed):
    rng = np.random.default_rng(seed)
    q = dist(rng, 6, zero_some=True)
    mask = rng.random(6) < 0.5
    mask[rng.integers(6)] = True
    ok = lambda y: bool(mask[y[0]])  # noqa: E731
    assert_matches(cf.domain(q, mask), brute_force([q], ok), [q], ok)


@pytest.mark.parametrize("seed", SEEDS)
def test_fd_cross_task(seed):
    rng = np.random.default_rng(seed)
    q_a, q_b = dist(rng, 5, True), dist(rng, 4, True)
    if seed % 2:  # an exact FD b = g(a)
        R = np.zeros((5, 4), dtype=bool)
        R[np.arange(5), rng.integers(4, size=5)] = True
    else:  # a general admissible relation
        R = rng.random((5, 4)) < 0.4
        R[0, 0] = True
    ok = lambda y: bool(R[y[0], y[1]])  # noqa: E731
    assert_matches(cf.fd_cross_task(q_a, q_b, R), brute_force([q_a, q_b], ok), [q_a, q_b], ok)


@pytest.mark.parametrize("seed", SEEDS)
def test_group_agreement(seed):
    rng = np.random.default_rng(seed)
    qs = np.stack([dist(rng, 4, True) for _ in range(3)])
    ok = lambda y: len(set(y)) == 1  # noqa: E731
    assert_matches(cf.group_agreement(qs), brute_force(list(qs), ok), list(qs), ok)


@pytest.mark.parametrize("seed", SEEDS)
def test_agreement_fd(seed):
    rng = np.random.default_rng(seed)
    m, n_parent, n_child = 2, 3, 5
    q_parent = np.stack([dist(rng, n_parent, True) for _ in range(m)])
    q_child = np.stack([dist(rng, n_child, True) for _ in range(m)])
    g = rng.integers(-1, n_parent, size=n_child)
    g[0] = 0
    marginals = list(q_parent) + list(q_child)

    def ok(y):
        parents, children = y[:m], y[m:]
        return len(set(parents)) == 1 and all(g[s] == parents[0] for s in children)

    assert_matches(cf.agreement_fd(q_parent, q_child, g), brute_force(marginals, ok), marginals, ok)


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("bounds", [(2, 2), (0, 2), (1, 4), (3, 6)])
def test_cardinality(seed, bounds):
    rng = np.random.default_rng(seed)
    p = rng.random(6)
    if seed % 3 == 0:
        p[rng.integers(6)] = 1.0
    if seed % 4 == 0:
        p[rng.integers(6)] = 0.0
    lo, hi = bounds
    marginals = [np.array([1 - pi, pi]) for pi in p]
    ok = lambda y: lo <= sum(y) <= hi  # noqa: E731
    assert_matches(cf.cardinality(p, lo, hi), brute_force(marginals, ok), marginals, ok)


@pytest.mark.parametrize("seed", SEEDS)
def test_aggregate_sum(seed):
    rng = np.random.default_rng(seed)
    values = [np.array([0, 1, 2]), np.array([0, 2, 5]), np.array([1, 3])]
    qs = [dist(rng, len(v), True) for v in values]
    total = int(rng.choice([3, 4, 5, 6]))
    ok = lambda y: sum(int(v[k]) for v, k in zip(values, y)) == total  # noqa: E731
    assert_matches(cf.aggregate_sum(qs, values, total), brute_force(qs, ok), qs, ok)


@pytest.mark.parametrize("seed", SEEDS)
def test_temporal_order(seed):
    rng = np.random.default_rng(seed)
    q_i, q_j = dist(rng, 6, True), dist(rng, 6, True)
    ok = lambda y: y[0] <= y[1]  # noqa: E731
    assert_matches(cf.temporal_order(q_i, q_j), brute_force([q_i, q_j], ok), [q_i, q_j], ok)


@pytest.mark.parametrize("seed", SEEDS)
def test_isotonic_projection_matches_sklearn(seed):
    from sklearn.isotonic import IsotonicRegression

    rng = np.random.default_rng(seed)
    y = rng.normal(size=12)
    w = rng.random(12) + 0.1
    got = cf.isotonic_projection(y, w)
    want = IsotonicRegression().fit_transform(np.arange(12), y, sample_weight=w)
    np.testing.assert_allclose(got, want, atol=1e-10)
    assert np.all(np.diff(got) >= -1e-12)
    assert np.all(np.diff(cf.isotonic_projection(y, w, increasing=False)) <= 1e-12)


def brute_soft(marginals, is_coherent, delta):
    """The soft-conditioned joint, enumerated: q(y) delta^v(y) / Z."""
    ys = list(itertools.product(*(range(len(m)) for m in marginals)))
    w = np.array([math.prod(m[v] for m, v in zip(marginals, y)) * (1 if is_coherent(y) else delta) for y in ys])
    w = w / w.sum()
    cond = [np.zeros(len(m)) for m in marginals]
    for wi, y in zip(w, ys):
        for i, v in enumerate(y):
            cond[i][v] += wi
    return cond, ys[int(np.argmax(w))], w.max()


@pytest.mark.parametrize("seed", SEEDS)
@pytest.mark.parametrize("delta", [0.01, 0.3, 2.0])
def test_soft_marginals_and_map(seed, delta):
    rng = np.random.default_rng(seed)
    q_a, q_b = dist(rng, 4), dist(rng, 3)
    R = rng.random((4, 3)) < 0.5
    R[0, 0] = True
    ok = lambda y: bool(R[y[0], y[1]])  # noqa: E731
    hard = cf.fd_cross_task(q_a, q_b, R)
    want_cond, want_map, want_max = brute_soft([q_a, q_b], ok, delta)
    q_C = math.exp(hard.log_qC)
    for raw, h, want in zip([q_a, q_b], hard.marginals, want_cond):
        np.testing.assert_allclose(soft_marginal(raw, h, q_C, delta), want, atol=1e-12)

    joint = np.outer(q_a, q_b)
    best_incoherent = np.unravel_index(int(np.argmax(np.where(R, -1.0, joint))), joint.shape)
    prefer = soft_map_prefers_incoherent(
        log_prob([q_a, q_b], hard.map), np.log(joint[best_incoherent]), math.log(delta)
    )
    got_map = best_incoherent if prefer else hard.map
    got_weight = q_a[got_map[0]] * q_b[got_map[1]] * (1 if ok(got_map) else delta)
    assert got_weight == pytest.approx(want_max * (q_C + delta * (1 - q_C)), rel=1e-9)


@pytest.mark.parametrize("seed", SEEDS)
def test_prop1_hard_conditioning_lowers_loss_by_deficit(seed):
    rng = np.random.default_rng(seed)
    p = rng.random(5)
    res = cf.cardinality(p, 2, 2)
    for y in itertools.product([0, 1], repeat=5):
        raw = -log_prob([np.array([1 - pi, pi]) for pi in p], y)
        violated = sum(y) != 2
        got = joint_nll(raw, violated, res.log_qC, log_delta=-np.inf)
        assert got == (np.inf if violated else pytest.approx(raw - res.deficit))


@pytest.mark.parametrize("pi,r", [(0.01, 0.3), (0.2, 0.05), (0.5, 0.5), (0.0003, 0.6)])
def test_prop2_optimal_strength_is_odds_ratio(pi, r):
    log_qC = math.log(1 - r)
    # expected loss gain over q as a function of delta, from joint_nll on the two outcome kinds
    def gain(delta):
        t = math.log(delta)
        return -(pi * joint_nll(0.0, True, log_qC, t) + (1 - pi) * joint_nll(0.0, False, log_qC, t))

    delta_star = (pi / (1 - pi)) / (r / (1 - r))
    kl = pi * math.log(pi / r) + (1 - pi) * math.log((1 - pi) / (1 - r))
    assert gain(delta_star) == pytest.approx(kl, abs=1e-12)
    for d in (delta_star * 0.7, delta_star * 1.4, 1.0):
        assert gain(d) <= gain(delta_star) + 1e-12


def test_prop3_fitted_strength():
    rng = np.random.default_rng(0)
    r = rng.beta(0.5, 3, size=2000)
    log_qC = np.log1p(-r)
    v = rng.random(2000) < 0.05
    t = fit_log_strength(v, log_qC)
    delta = math.exp(t)
    tilted = np.mean(r * delta / (1 - r + r * delta))
    assert tilted == pytest.approx(v.mean(), abs=1e-6)
    assert strength_objective(t, v, log_qC) <= strength_objective(0.0, v, log_qC) + 1e-12
    for dt in (-0.5, 0.5):
        assert strength_objective(t, v, log_qC) <= strength_objective(t + dt, v, log_qC) + 1e-12
    assert fit_log_strength(np.zeros(10, dtype=bool), log_qC[:10]) == -np.inf


def test_prop3_strength_below_float_range():
    # scopes whose coherent mass is far below the smallest float: the fitted log-strength must
    # still be finite and interior, not clipped (exp(t) itself would underflow to 0)
    rng = np.random.default_rng(1)
    log_qC = -rng.uniform(800, 1200, size=500)
    v = np.zeros(500, dtype=bool)
    v[:5] = True
    t = fit_log_strength(v, log_qC)
    assert -1e7 < t < -700
    logit_r = np.log(-np.expm1(log_qC)) - log_qC
    assert np.mean(1 / (1 + np.exp(-(logit_r + t)))) == pytest.approx(v.mean(), abs=1e-6)
    soft = joint_nll(np.zeros(500), v, log_qC, t)
    assert np.isfinite(soft).all()


def test_corollary1_example():
    # 20 drivers, each top-3 with probability 3/20, exactly 3 qualify
    res = cf.cardinality(np.full(20, 3 / 20), 3, 3)
    assert math.exp(res.log_qC) == pytest.approx(math.comb(20, 3) * 0.15**3 * 0.85**17)
    assert math.exp(res.log_qC) == pytest.approx(0.243, abs=5e-4)
    assert res.deficit == pytest.approx(1.42, abs=5e-3)
    assert sum(res.map) == 3  # thresholding at 1/2 would predict 0 qualifiers


@pytest.mark.parametrize("seed", range(10))
def test_salt_item_fd_matches_closed_form(seed):
    from nesy_tabfm.ccd.salt import item_fd

    rng = np.random.default_rng(seed)
    n, n_ship, n_plant = 40, 7, 4
    q_ship = np.stack([dist(rng, n_ship, True) for _ in range(n)])
    q_plant = np.stack([dist(rng, n_plant, True) for _ in range(n)])
    g = rng.integers(-1, n_plant, size=n_ship)
    g[0] = 1
    R = np.zeros((n_ship, n_plant), dtype=bool)
    R[np.flatnonzero(g >= 0), g[g >= 0]] = True
    got = item_fd(log(q_ship), log(q_plant), g, chunk=7)
    for i in range(n):
        want = cf.fd_cross_task(q_ship[i], q_plant[i], R)
        assert got.log_qC[i] == pytest.approx(want.log_qC, abs=1e-9)
        joint = np.outer(q_ship[i], q_plant[i])
        assert got.log_q_map[i] == pytest.approx(np.log(joint[want.map]), abs=1e-9)
        assert R[got.map_ship[i], got.map_plant[i]]
        assert not R[got.incoherent_ship[i], got.incoherent_plant[i]]
        assert got.log_q_incoherent[i] == pytest.approx(np.log(np.where(R, -1.0, joint).max()), abs=1e-9)


@pytest.mark.parametrize("seed", range(10))
def test_salt_order_agreement_fd_matches_closed_form(seed):
    from nesy_tabfm.ccd.salt import order_agreement_fd

    rng = np.random.default_rng(seed)
    n_ship, n_plant = 6, 3
    sizes = rng.integers(1, 4, size=8)
    order_index = np.repeat(np.arange(len(sizes)), sizes)
    n = len(order_index)
    q_ship = np.stack([dist(rng, n_ship, True) for _ in range(n)])
    q_plant = np.stack([dist(rng, n_plant, True) for _ in range(n)])
    g = rng.integers(-1, n_plant, size=n_ship)
    g[:2] = [0, 1]
    got = order_agreement_fd(log(q_ship), log(q_plant), g, order_index, len(sizes))
    for o in range(len(sizes)):
        items = np.flatnonzero(order_index == o)
        want = cf.agreement_fd(q_plant[items], q_ship[items], g)
        assert got.log_qC[o] == pytest.approx(want.log_qC, abs=1e-9)
        if want.map is None:
            continue
        m = len(items)
        want_logp = log_prob(list(q_plant[items]) + list(q_ship[items]), want.map)
        got_map = (got.map_plant[o],) * m + tuple(got.item_map_ship[items])
        assert got.log_q_map[o] == pytest.approx(want_logp, abs=1e-9)
        assert log_prob(list(q_plant[items]) + list(q_ship[items]), got_map) == pytest.approx(want_logp, abs=1e-9)


def test_temper_keeps_argmax_and_normalizes():
    from nesy_tabfm.ccd.calibration import temper

    rng = np.random.default_rng(0)
    z = rng.normal(scale=4, size=(50, 7))
    log_q = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
    log_q[0, 3] = -np.inf  # a probability of exactly 0 stays 0
    for t in (0.3, 1.0, 5.0):
        out = temper(log_q, t)
        np.testing.assert_allclose(np.exp(out).sum(axis=1), 1.0, rtol=1e-12)
        np.testing.assert_array_equal(np.argmax(out, axis=1), np.argmax(log_q, axis=1))
        assert np.isneginf(out[0, 3])
    np.testing.assert_allclose(temper(log_q, 1.0)[1:], log_q[1:], atol=1e-12)


@pytest.mark.parametrize("true_t", [0.5, 2.5, 8.0])
def test_fit_temperature_recovers_miscalibration(true_t):
    from nesy_tabfm.ccd.calibration import fit_temperature

    rng = np.random.default_rng(1)
    z = rng.normal(scale=3, size=(40_000, 6))
    # labels drawn from softmax(z / T); the model reports softmax(z), i.e. it is miscalibrated by T
    p_true = np.exp(z / true_t)
    p_true /= p_true.sum(axis=1, keepdims=True)
    y = (rng.random((len(z), 1)) > np.cumsum(p_true, axis=1)).sum(axis=1)
    log_q = z - np.log(np.exp(z).sum(axis=1, keepdims=True))
    assert fit_temperature(log_q, y) == pytest.approx(true_t, rel=0.05)
    # labels outside the classes are ignored, not fitted
    y_with_unseen = y.copy()
    y_with_unseen[:500] = -1
    assert fit_temperature(log_q, y_with_unseen) == pytest.approx(fit_temperature(log_q[500:], y[500:]), rel=1e-6)


def test_strength_and_loss_tolerate_rounding_above_zero():
    # log q(C) can come out a hair above 0 when q(C) is 1; that must not turn into NaN
    log_qC = np.array([2e-16, -0.5, -3.0, 1e-17, -40.0])
    v = np.array([False, False, True, False, False])
    t = fit_log_strength(v, log_qC)
    assert np.isfinite(t)
    assert np.isfinite(joint_nll(np.ones(5), v, log_qC, t)).all()
    assert np.isfinite(strength_objective(t, v, log_qC))
