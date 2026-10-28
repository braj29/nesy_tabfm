"""Paired significance tests for comparing decoders on the same scopes.

Items of one rel-salt order are not independent (an order-level decision changes all of them
at once, and they share a customer), so intervals come from a cluster bootstrap that resamples
whole orders. McNemar's test is also reported for accuracy, but it assumes independent items.
"""

from __future__ import annotations

import numpy as np
from scipy.stats import binomtest


def cluster_bootstrap_ci(
    diff: np.ndarray,
    cluster: np.ndarray,
    n_clusters: int,
    mask: np.ndarray | None = None,
    n_boot: int = 2000,
    seed: int = 0,
    level: float = 0.95,
) -> tuple[float, float, float]:
    """Mean of a paired per-scope difference and its percentile bootstrap interval, resampling
    clusters (orders) with replacement. ``diff[i]`` is method A minus method B on scope ``i``,
    ``cluster[i]`` its cluster index in ``0..n_clusters-1``; ``mask`` selects the scopes used."""
    diff = np.asarray(diff, dtype=float)
    cluster = np.asarray(cluster, dtype=int)
    keep = np.ones(len(diff), dtype=bool) if mask is None else np.asarray(mask, dtype=bool)
    sums = np.bincount(cluster[keep], weights=diff[keep], minlength=n_clusters)
    counts = np.bincount(cluster[keep], minlength=n_clusters).astype(float)
    estimate = float(sums.sum() / counts.sum())
    rng = np.random.default_rng(seed)
    reps = []
    for start in range(0, n_boot, 100):
        idx = rng.integers(0, n_clusters, size=(min(100, n_boot - start), n_clusters))
        reps.append(sums[idx].sum(axis=1) / counts[idx].sum(axis=1))
    reps = np.concatenate(reps)
    alpha = (1.0 - level) / 2.0
    return estimate, float(np.quantile(reps, alpha)), float(np.quantile(reps, 1.0 - alpha))


def mcnemar(a_correct: np.ndarray, b_correct: np.ndarray, mask: np.ndarray | None = None) -> dict:
    """Exact McNemar test: ``a_only`` scopes only method A gets right, ``b_only`` only B; the
    two-sided p-value is a binomial test of ``a_only`` against ``a_only + b_only`` at 1/2."""
    a = np.asarray(a_correct, dtype=bool)
    b = np.asarray(b_correct, dtype=bool)
    if mask is not None:
        a, b = a[mask], b[mask]
    a_only, b_only = int((a & ~b).sum()), int((~a & b).sum())
    p = binomtest(a_only, a_only + b_only, 0.5).pvalue if a_only + b_only else 1.0
    return {"a_only": a_only, "b_only": b_only, "p_value": float(p)}
