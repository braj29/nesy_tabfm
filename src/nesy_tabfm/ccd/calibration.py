"""Temperature scaling of per-cell predictive distributions, as a step before CCD.

CCD trusts the model's probabilities: on a violating scope it changes the cell the model is
least sure about. When a model is overconfident (LightGBM on rel-salt reports ~1.0 on
shipping points it gets wrong), the probabilities cannot tell which cell is wrong. Temperature
scaling (Guo et al., 2017) fits one scalar per task on held-out labels and rescales the
log-probabilities; it never changes a cell's argmax, only how confident the model is.
"""

from __future__ import annotations

import numpy as np
from scipy.optimize import minimize_scalar
from scipy.special import logsumexp


def temper(log_q: np.ndarray, temperature: float) -> np.ndarray:
    """Temperature-scaled distribution ``q_T(y) ∝ q(y)^(1/T)``, in log space. ``-inf``
    (probability exactly 0) stays ``-inf``."""
    z = np.asarray(log_q, dtype=float) / temperature
    with np.errstate(invalid="ignore"):
        return z - logsumexp(z, axis=1, keepdims=True)


T_BOUNDS = (1e-3, 1e7)


def fit_temperature(log_q: np.ndarray, y_idx: np.ndarray, t_bounds: tuple[float, float] = T_BOUNDS) -> float:
    """Temperature minimizing the mean negative log-likelihood of the labels ``y_idx``
    (column indices into ``log_q``). Labels outside the model's classes (``-1``) and labels the
    model gives probability exactly 0 have infinite loss at every temperature, so they are left
    out. The loss is convex in the inverse temperature, so a bounded scalar search finds it."""
    log_q = np.asarray(log_q, dtype=float)
    y_idx = np.asarray(y_idx, dtype=int)
    keep = y_idx >= 0
    rows = np.flatnonzero(keep)
    true = log_q[rows, y_idx[keep]]
    finite = np.isfinite(true)
    lq, true = log_q[rows[finite]], true[finite]
    if len(true) == 0:
        return 1.0

    def nll(log_beta: float) -> float:
        beta = np.exp(log_beta)
        with np.errstate(invalid="ignore"):
            return float(np.mean(logsumexp(beta * lq, axis=1) - beta * true))

    lo, hi = np.log(1.0 / t_bounds[1]), np.log(1.0 / t_bounds[0])
    res = minimize_scalar(nll, bounds=(lo, hi), method="bounded", options={"xatol": 1e-6})
    return float(1.0 / np.exp(res.x))
