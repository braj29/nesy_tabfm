"""Turn a ViolationResult into per-instance flags and stratified violation-rate metrics.

A constraint's violations can be at the row level (ReferentialIntegrity, unary
DenialConstraint) or the group level (Cardinality, AggregateConsistency, binary
DenialConstraint, TemporalOrder). Either way, RQ1 wants to know: among a set of *model
prediction instances* (one row per (entity, timestamp) with a true label, a predicted
value, and an error), which ones sit inside a violation, and does the violation rate
concentrate in the temporal-shift tail or in rare classes?
"""

from __future__ import annotations

import pandas as pd

from nesy_tabfm.constraints.dsl import ViolationResult


def instance_violation_flags(
    result: ViolationResult, instances: pd.DataFrame, join_cols: list[str]
) -> pd.Series:
    """Bool Series aligned to ``instances.index``: True where that instance participates in
    a violating row/group of ``result``."""
    if result.n_violations == 0:
        return pd.Series(False, index=instances.index)
    keys = result.violations[join_cols].drop_duplicates()
    keys["_violated"] = True
    merged = instances[join_cols].merge(keys, on=join_cols, how="left")
    return merged["_violated"].fillna(False).to_numpy()


def pairwise_instance_violation_flags(
    result: ViolationResult, instances: pd.DataFrame, id_col: str, suffixes: tuple[str, str] = ("_l", "_r")
) -> pd.Series:
    """Like ``instance_violation_flags``, but for a *pairwise* ``DenialConstraint`` (a
    self-join): ``result.violations`` has one row per violating pair (columns
    ``id_col + suffixes[0]`` and ``id_col + suffixes[1]``), so an instance counts as violated
    if it appears on *either* side of any violating pair."""
    if result.n_violations == 0:
        return pd.Series(False, index=instances.index).to_numpy()
    left_col, right_col = id_col + suffixes[0], id_col + suffixes[1]
    violated_ids = set(result.violations[left_col]) | set(result.violations[right_col])
    return instances[id_col].isin(violated_ids).to_numpy()


def temporal_buckets(timestamps: pd.Series, n_bins: int = 4) -> pd.Series:
    """Quantile-based buckets over ``timestamps`` labeled ``bucket_1`` (earliest) ...
    ``bucket_k`` (latest, i.e. furthest from the training cutoff) -- a simple proxy for
    temporal distribution shift within a test window."""
    ranks = timestamps.rank(method="first")
    labels = [f"bucket_{i + 1}" for i in range(n_bins)]
    return pd.qcut(ranks, q=n_bins, labels=labels)


def stratified_rate(flags, group: pd.Series, group_name: str = "group") -> pd.DataFrame:
    """Violation rate and count, grouped by an arbitrary label (temporal bucket, rare-class
    indicator, ...)."""
    df = pd.DataFrame({group_name: group, "violated": pd.Series(flags).astype(bool).to_numpy()})
    out = (
        df.groupby(group_name, observed=True)["violated"]
        .agg(n="count", n_violations="sum", violation_rate="mean")
        .reset_index()
    )
    out["n_violations"] = out["n_violations"].astype(int)
    return out


def violation_error_correlation(flags, error: pd.Series) -> float:
    """Point-biserial correlation between the binary violation flag and prediction error
    (e.g. absolute error for regression, 1 - accuracy indicator for classification). NaN if
    either series has zero variance (e.g. no violations at all)."""
    flag_arr = pd.Series(flags).astype(float).reset_index(drop=True)
    err_arr = pd.Series(error).astype(float).reset_index(drop=True)
    if flag_arr.std() == 0 or err_arr.std() == 0:
        return float("nan")
    return flag_arr.corr(err_arr)
