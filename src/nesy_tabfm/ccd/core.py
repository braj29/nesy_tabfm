"""Constraint-Conditioned Decoding (CCD): pieces shared by every constraint type.

A *scope* is a set of target cells coupled by one grounded constraint, each with a per-cell
predictive distribution ``q_i`` from a frozen model. The factorized joint is
``q(y_S) = prod_i q_i(y_i)`` and the constraint defines an event ``C`` of admissible joint
assignments. Hard conditioning restricts ``q`` to ``C`` and renormalizes; soft conditioning
down-weights incoherent assignments by a strength ``delta`` (``delta=1`` is no conditioning,
``delta -> 0`` is hard conditioning).

Everything here works from three per-scope quantities, which each closed form in
``closed_forms.py`` provides: ``log q(C)``, the hard-conditioned marginals, and the coherent
MAP assignment.
"""

from __future__ import annotations

import itertools
from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
from scipy.special import expit, logsumexp


@dataclass
class ScopeResult:
    """Exact CCD output for one scope.

    ``log_qC``: log of the probability mass the factorized predictive puts on coherent
    assignments (``-inf`` if none). ``marginals``: one hard-conditioned marginal per cell.
    ``map``: the coherent assignment with the highest factorized probability, as value
    indices per cell (``None`` if no coherent assignment has positive mass).
    """

    log_qC: float
    marginals: list[np.ndarray]
    map: tuple[int, ...] | None

    @property
    def deficit(self) -> float:
        """Coherence deficit ``Delta = -log q(C)``, in nats."""
        return -self.log_qC


def brute_force(
    marginals: Sequence[np.ndarray], is_coherent: Callable[[tuple[int, ...]], bool]
) -> ScopeResult:
    """Exact CCD by enumerating every joint assignment. Exponential in the scope size, so only
    for small scopes: the reference every closed form is tested against, and the fallback for
    constraints with no closed form."""
    sizes = [len(m) for m in marginals]
    with np.errstate(divide="ignore"):
        log_q = [np.log(np.asarray(m, dtype=float)) for m in marginals]
    coherent_logp, coherent_assignments = [], []
    for y in itertools.product(*(range(s) for s in sizes)):
        if is_coherent(y):
            coherent_logp.append(sum(lq[v] for lq, v in zip(log_q, y)))
            coherent_assignments.append(y)
    if not coherent_assignments or np.all(np.isneginf(coherent_logp)):
        return ScopeResult(-np.inf, [np.full(s, np.nan) for s in sizes], None)

    logp = np.array(coherent_logp)
    log_qC = float(logsumexp(logp))
    weights = np.exp(logp - log_qC)
    cond = [np.zeros(s) for s in sizes]
    for w, y in zip(weights, coherent_assignments):
        for i, v in enumerate(y):
            cond[i][v] += w
    best = coherent_assignments[int(np.argmax(logp))]
    return ScopeResult(log_qC, cond, tuple(best))


def soft_marginal(raw: np.ndarray, hard: np.ndarray, q_C: float, delta: float) -> np.ndarray:
    """Marginal of one cell under soft conditioning with strength ``delta``.

    The soft-conditioned joint is a mixture of the coherent and incoherent parts of ``q``, so
    its marginal needs only the raw marginal, the hard-conditioned marginal and ``q(C)``:
    ``(q(C) q_C(y) + delta (q(y) - q(C) q_C(y))) / (q(C) + delta (1 - q(C)))``.
    """
    coherent = q_C * np.asarray(hard, dtype=float)
    return (coherent + delta * (np.asarray(raw, dtype=float) - coherent)) / (q_C + delta * (1.0 - q_C))


def joint_nll(raw_nll: np.ndarray, violated: np.ndarray, log_qC: np.ndarray, log_delta: float) -> np.ndarray:
    """Joint negative log-likelihood of each scope's realized outcome under conditioning with
    strength ``delta = exp(log_delta)``.

    ``raw_nll`` is ``-log q(y_S)`` under the factorized predictive, ``violated`` whether the
    realized outcome breaks the constraint. ``log_delta = 0`` returns ``raw_nll`` unchanged;
    ``log_delta = -inf`` is hard conditioning (``raw_nll - Delta`` on coherent outcomes, ``inf``
    on violations); otherwise ``raw_nll - v log(delta) + log(q(C) + delta (1 - q(C)))``, computed
    in log space because a fitted strength can be far below the smallest positive float.
    """
    raw_nll = np.asarray(raw_nll, dtype=float)
    violated = np.asarray(violated, dtype=bool)
    log_qC = np.asarray(log_qC, dtype=float)
    if np.isneginf(log_delta):
        return np.where(violated, np.inf, raw_nll + log_qC)
    log_z = np.logaddexp(log_qC, log_delta + _log_not_coherent(log_qC))
    return raw_nll - violated * log_delta + log_z


def soft_map_prefers_incoherent(
    log_q_best_coherent: np.ndarray, log_q_best_incoherent: np.ndarray, log_delta: float
) -> np.ndarray:
    """Where the MAP of the soft-conditioned joint is the best *incoherent* assignment rather
    than the coherent MAP: iff ``delta * q(best incoherent) > q(best coherent)``.

    For ``delta <= 1`` the best incoherent assignment only matters when it is the unconstrained
    argmax (otherwise the argmax is coherent and wins), so callers can pass ``log q(argmax)``
    when the argmax is incoherent and ``-inf`` when it is coherent. For ``delta > 1``, which
    up-weights exceptions, the best incoherent assignment must be computed explicitly.
    """
    if np.isneginf(log_delta):
        return np.zeros(np.shape(log_q_best_coherent), dtype=bool)
    return log_delta + np.asarray(log_q_best_incoherent) > np.asarray(log_q_best_coherent)


def _log_not_coherent(log_qC: np.ndarray) -> np.ndarray:
    """log(1 - q(C)) from log q(C), without cancellation when q(C) is close to 0 or 1. Rounding
    can leave log q(C) a hair above 0 when q(C) is 1, so it is clamped there first."""
    with np.errstate(divide="ignore"):
        return np.log(-np.expm1(np.minimum(np.asarray(log_qC, dtype=float), 0.0)))


def strength_objective(t: float, violated: np.ndarray, log_qC: np.ndarray) -> float:
    """Validation joint log-loss of soft conditioning at ``t = log(delta)``, minus its value at
    ``t = 0`` (Proposition 3): ``mean(-v t + log(1 - r + r e^t))`` with ``r = 1 - q(C)``.
    Convex in ``t``."""
    v = np.asarray(violated, dtype=float)
    log_qC = np.asarray(log_qC, dtype=float)
    return float(np.mean(-v * t + np.logaddexp(log_qC, _log_not_coherent(log_qC) + t)))


def fit_log_strength(
    violated: np.ndarray, log_qC: np.ndarray, t_bounds: tuple[float, float] = (-1e7, 1e7)
) -> float:
    """Fit the soft-conditioning strength on validation scopes (Proposition 3); returns
    ``t = log(delta)``.

    ``violated``: whether each scope's realized outcome breaks the constraint. ``log_qC``: the
    model's ``log q(C)`` for each scope. The objective is convex in ``t`` and its stationarity
    condition matches the mean tilted violation probability to the observed violation rate, so
    ``t`` is found by bisection. Works on ``logit(1 - q(C))`` directly, since scopes whose
    ``q(C)`` is below the smallest float would otherwise all look identical. Returns ``-inf``
    (hard conditioning) when no validation scope violates the constraint.
    """
    v = np.asarray(violated, dtype=float)
    target = v.mean()
    if target == 0.0:
        return -np.inf
    logit_r = _log_not_coherent(log_qC) - np.minimum(np.asarray(log_qC, dtype=float), 0.0)

    def mean_tilted(t: float) -> float:
        with np.errstate(invalid="ignore"):
            return float(np.mean(expit(logit_r + t)))

    lo, hi = t_bounds
    if mean_tilted(hi) < target:
        return float(hi)
    if mean_tilted(lo) > target:
        return float(lo)
    for _ in range(200):
        mid = 0.5 * (lo + hi)
        if mean_tilted(mid) < target:
            lo = mid
        else:
            hi = mid
    return float(0.5 * (lo + hi))
