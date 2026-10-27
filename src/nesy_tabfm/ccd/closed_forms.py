"""Exact CCD for each integrity-constraint type (Table 3 of the CCD paper).

Each function takes the per-cell predictive distributions of one scope and returns a
``ScopeResult``: ``log q(C)``, the hard-conditioned marginals and the coherent MAP. Every one is
checked against ``core.brute_force`` in ``tests/test_ccd.py``.

Value indices, not raw labels, are used throughout: cell ``i``'s distribution is an array
over its own label set, and relations between cells are given as index arrays or boolean
matrices over those label sets.
"""

from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from scipy.special import logsumexp

from nesy_tabfm.ccd.core import ScopeResult


def _log(x: np.ndarray) -> np.ndarray:
    with np.errstate(divide="ignore"):
        return np.log(np.asarray(x, dtype=float))


def _infeasible(sizes: Sequence[int]) -> ScopeResult:
    return ScopeResult(-np.inf, [np.full(s, np.nan) for s in sizes], None)


def domain(q: np.ndarray, admissible: np.ndarray) -> ScopeResult:
    """One cell whose value must lie in an admissible set (range/domain, referential
    integrity, or an FD whose determinant is observed -- pass that determinant's row of the
    relation as ``admissible``)."""
    q = np.asarray(q, dtype=float)
    mask = np.asarray(admissible, dtype=bool)
    q_C = q[mask].sum()
    if q_C <= 0:
        return _infeasible([len(q)])
    return ScopeResult(float(np.log(q_C)), [np.where(mask, q, 0.0) / q_C], (int(np.argmax(np.where(mask, q, -1.0))),))


def fd_cross_task(q_a: np.ndarray, q_b: np.ndarray, relation: np.ndarray) -> ScopeResult:
    """Two cells predicted by two different tasks for the same entity, whose values must form
    an admissible pair: ``relation[a, b]`` is True for admissible ``(a, b)`` (for an exact FD
    ``b = g(a)``, row ``a`` has a single True at ``g(a)``)."""
    q_a = np.asarray(q_a, dtype=float)
    q_b = np.asarray(q_b, dtype=float)
    R = np.asarray(relation, dtype=float)
    support_a = R @ q_b  # mass of b-values admissible with each a
    support_b = R.T @ q_a
    q_C = float(q_a @ support_a)
    if q_C <= 0:
        return _infeasible([len(q_a), len(q_b)])
    joint = np.where(R > 0, np.outer(q_a, q_b), -1.0)
    a, b = np.unravel_index(int(np.argmax(joint)), joint.shape)
    return ScopeResult(float(np.log(q_C)), [q_a * support_a / q_C, q_b * support_b / q_C], (int(a), int(b)))


def group_agreement(qs: np.ndarray) -> ScopeResult:
    """``m`` cells over one shared label set that must all take the same value (a denial
    constraint such as "no two items of an order disagree on the plant"). ``qs``: ``(m, Y)``."""
    qs = np.asarray(qs, dtype=float)
    m, n_values = qs.shape
    log_scores = _log(qs).sum(axis=0)  # log prod_j q_j(c), per shared value c
    log_qC = float(logsumexp(log_scores))
    if np.isneginf(log_qC):
        return _infeasible([n_values] * m)
    cond = np.exp(log_scores - log_qC)
    c = int(np.argmax(log_scores))
    return ScopeResult(log_qC, [cond.copy() for _ in range(m)], (c,) * m)


def agreement_fd(q_parent: np.ndarray, q_child: np.ndarray, child_to_parent: np.ndarray) -> ScopeResult:
    """An order whose ``m`` items must all share one parent value (plant), where each item's
    child value (shipping point) must also belong to that parent: ``child_to_parent[s]`` is
    the parent index of child value ``s``, or -1 if ``s`` belongs to no parent in the label
    set. ``q_parent``: ``(m, P)``; ``q_child``: ``(m, A)``. Cells are ordered as the ``m``
    parent cells followed by the ``m`` child cells.

    Summing out each item's child value gives the order-level parent posterior
    ``q_C(c) ∝ prod_j q_parent_j(c) * sum_{s: g(s) = c} q_child_j(s)``.
    """
    q_parent = np.asarray(q_parent, dtype=float)
    q_child = np.asarray(q_child, dtype=float)
    g = np.asarray(child_to_parent, dtype=int)
    m, n_parent = q_parent.shape
    n_child = q_child.shape[1]
    member = np.zeros((n_child, n_parent))
    valid = g >= 0
    member[np.flatnonzero(valid), g[valid]] = 1.0

    child_sum = q_child @ member  # (m, P): mass of child values belonging to each parent
    log_item = _log(q_parent) + _log(child_sum)  # (m, P)
    log_order = log_item.sum(axis=0)  # (P,)
    log_qC = float(logsumexp(log_order))
    if np.isneginf(log_qC):
        return _infeasible([n_parent] * m + [n_child] * m)

    parent_cond = np.exp(log_order - log_qC)
    marginals = [parent_cond.copy() for _ in range(m)]
    for j in range(m):
        # child value s of item j, with every other item summed out under parent c = g(s)
        log_rest = np.delete(log_item, j, axis=0).sum(axis=0)  # (P,): every other item's factor
        logit = np.full(n_child, -np.inf)
        logit[valid] = _log(q_child[j, valid]) + _log(q_parent[j, g[valid]]) + log_rest[g[valid]]
        marginals.append(np.exp(logit - log_qC))

    # MAP maximizes prod_j q_parent_j(c) q_child_j(s_j) with g(s_j) = c for all j.
    child_max = np.full((m, n_parent), -np.inf)
    for c in range(n_parent):
        cols = np.flatnonzero(g == c)
        if len(cols):
            child_max[:, c] = _log(q_child[:, cols].max(axis=1))
    map_scores = (_log(q_parent) + child_max).sum(axis=0)
    c_star = int(np.argmax(map_scores))
    cols = np.flatnonzero(g == c_star)
    s_star = cols[np.argmax(q_child[:, cols], axis=1)]
    return ScopeResult(log_qC, marginals, (c_star,) * m + tuple(int(s) for s in s_star))


def _log_esp(log_w: np.ndarray, k_max: int) -> np.ndarray:
    """log of the elementary symmetric polynomials e_0..e_{k_max} of exp(log_w)."""
    e = np.full(k_max + 1, -np.inf)
    e[0] = 0.0
    for lw in log_w:
        e[1:] = np.logaddexp(e[1:], e[:-1] + lw)
    return e


def cardinality(p: np.ndarray, lo: int, hi: int) -> ScopeResult:
    """``n`` binary cells whose number of positives must lie in ``[lo, hi]`` (``lo == hi`` is
    an exactly-K constraint, ``lo == 0`` an at-most-K one). ``p``: each cell's probability of 1.

    ``q(C) = prod_i (1 - p_i) sum_{k=lo}^{hi} e_k(w)`` with odds ``w_i = p_i / (1 - p_i)``, and
    the conditioned marginal of each cell is the conditional Bernoulli
    ``w_i sum_k e_{k-1}(w_{-i}) / sum_k e_k(w)``, from prefix and suffix tables. The MAP keeps
    the cells with odds above 1, clipped to between ``lo`` and ``hi`` of the largest odds.
    Works in log space, so probabilities of exactly 0 or 1 are allowed.
    """
    p = np.asarray(p, dtype=float)
    n = len(p)
    hi = min(hi, n)
    if lo > hi:
        return _infeasible([2] * n)
    log_w = _log(p) - _log(1.0 - p)
    # Cells with p == 1 have infinite odds: they are always on, so condition them out.
    forced = np.isposinf(log_w)
    log_base = float(_log(1.0 - p)[~forced].sum())
    n_forced = int(forced.sum())
    free = np.flatnonzero(~forced)
    lo_free, hi_free = max(lo - n_forced, 0), hi - n_forced
    if hi_free < 0 or lo_free > len(free):
        return _infeasible([2] * n)
    hi_free = min(hi_free, len(free))

    e = _log_esp(log_w[free], hi_free)
    log_norm = float(logsumexp(e[lo_free : hi_free + 1]))
    if np.isneginf(log_norm):
        return _infeasible([2] * n)
    log_qC = log_base + log_norm

    on = np.ones(n)
    n_free = len(free)
    prefix = [np.r_[0.0, np.full(hi_free, -np.inf)]]
    for i in range(n_free):
        nxt = prefix[-1].copy()
        nxt[1:] = np.logaddexp(nxt[1:], prefix[-1][:-1] + log_w[free[i]])
        prefix.append(nxt)
    suffix = [np.r_[0.0, np.full(hi_free, -np.inf)]]
    for i in reversed(range(n_free)):
        nxt = suffix[-1].copy()
        nxt[1:] = np.logaddexp(nxt[1:], suffix[-1][:-1] + log_w[free[i]])
        suffix.append(nxt)
    suffix = suffix[::-1]  # suffix[i] covers free cells i..n_free-1
    for idx, i in enumerate(free):
        pre, suf = prefix[idx], suffix[idx + 1]
        # log e_k(w_{-i}) for k = 0..hi_free, by convolving prefix and suffix tables
        e_minus = np.array([logsumexp(pre[: k + 1] + suf[k::-1]) for k in range(hi_free + 1)])
        ks = np.arange(max(lo_free, 1), hi_free + 1)
        log_on = log_w[i] + logsumexp(e_minus[ks - 1]) if len(ks) else -np.inf
        on[i] = np.exp(log_on - log_norm)
    marginals = [np.array([1.0 - o, o]) for o in on]

    k_star = int(np.clip((log_w[free] > 0).sum(), lo_free, hi_free))
    chosen = set(free[np.argsort(-log_w[free], kind="stable")[:k_star]]) | set(np.flatnonzero(forced))
    return ScopeResult(log_qC, marginals, tuple(int(i in chosen) for i in range(n)))


def aggregate_sum(qs: Sequence[np.ndarray], values: Sequence[np.ndarray], total: int) -> ScopeResult:
    """Integer-valued cells whose values must sum to ``total``: cell ``i`` takes value
    ``values[i][k]`` with probability ``qs[i][k]``. Exact by dynamic programming over partial
    sums: forward/backward convolutions give ``q(C)`` and the marginals, max-product with
    backtracking gives the MAP."""
    qs = [np.asarray(q, dtype=float) for q in qs]
    values = [np.asarray(v, dtype=int) for v in values]
    n = len(qs)
    log_q = [_log(q) for q in qs]

    def forward(order: list[int], use_max: bool) -> list[dict[int, float]]:
        tables = [{0: 0.0}]
        for i in order:
            nxt: dict[int, float] = {}
            for s, ls in tables[-1].items():
                for v, lq in zip(values[i], log_q[i]):
                    if np.isneginf(lq):
                        continue
                    key, val = s + int(v), ls + lq
                    if key not in nxt:
                        nxt[key] = val
                    else:
                        nxt[key] = max(nxt[key], val) if use_max else float(np.logaddexp(nxt[key], val))
            tables.append(nxt)
        return tables

    fwd = forward(list(range(n)), use_max=False)
    if total not in fwd[n]:
        return _infeasible([len(q) for q in qs])
    log_qC = fwd[n][total]
    bwd = forward(list(reversed(range(n))), use_max=False)[::-1]  # bwd[i]: sums of cells i..n-1

    marginals = []
    for i in range(n):
        logits = np.full(len(qs[i]), -np.inf)
        for k, (v, lq) in enumerate(zip(values[i], log_q[i])):
            terms = [ls + bwd[i + 1].get(total - s - int(v), -np.inf) for s, ls in fwd[i].items()]
            logits[k] = lq + (logsumexp(terms) if terms else -np.inf)
        marginals.append(np.exp(logits - log_qC))

    best = forward(list(range(n)), use_max=True)
    assignment = [0] * n
    remaining = total
    for i in reversed(range(n)):
        cands = [
            (best[i].get(remaining - int(v), -np.inf) + lq, k)
            for k, (v, lq) in enumerate(zip(values[i], log_q[i]))
        ]
        score, k = max(cands)
        assignment[i] = k
        remaining -= int(values[i][k])
    return ScopeResult(float(log_qC), marginals, tuple(assignment))


def temporal_order(q_i: np.ndarray, q_j: np.ndarray) -> ScopeResult:
    """Two cells over the same ordered value grid, constrained by ``y_i <= y_j`` (an event that
    must not follow another). ``q_C(y_i = u) ∝ q_i(u) Pr_j[y_j >= u]`` and symmetrically for
    ``y_j``; the MAP pairs each ``v`` with the best ``u <= v`` via a running maximum."""
    q_i = np.asarray(q_i, dtype=float)
    q_j = np.asarray(q_j, dtype=float)
    tail_j = np.cumsum(q_j[::-1])[::-1]  # Pr_j[y_j >= u]
    head_i = np.cumsum(q_i)  # Pr_i[y_i <= v]
    q_C = float(q_i @ tail_j)
    if q_C <= 0:
        return _infeasible([len(q_i), len(q_j)])
    running_max = np.maximum.accumulate(q_i)
    running_arg = np.array([int(np.argmax(q_i[: v + 1])) for v in range(len(q_i))])
    v = int(np.argmax(q_j * running_max))
    return ScopeResult(float(np.log(q_C)), [q_i * tail_j / q_C, q_j * head_i / q_C], (int(running_arg[v]), v))


def isotonic_projection(y: np.ndarray, weights: np.ndarray | None = None, increasing: bool = True) -> np.ndarray:
    """Least-squares projection of a sequence onto monotone sequences, by pool-adjacent-
    violators. Used for monotonicity, which constrains the predictive function along a feature
    grid rather than outcomes within a scope, so CCD enforces it by projection."""
    y = np.asarray(y, dtype=float)
    w = np.ones_like(y) if weights is None else np.asarray(weights, dtype=float)
    if not increasing:
        return -isotonic_projection(-y, w, increasing=True)
    means, wts, counts = [], [], []
    for val, wt in zip(y, w):
        means.append(val)
        wts.append(wt)
        counts.append(1)
        while len(means) > 1 and means[-2] > means[-1]:
            total = wts[-2] + wts[-1]
            merged = (means[-2] * wts[-2] + means[-1] * wts[-1]) / total
            count = counts[-2] + counts[-1]
            means[-2:], wts[-2:], counts[-2:] = [merged], [total], [count]
    return np.repeat(means, counts)
