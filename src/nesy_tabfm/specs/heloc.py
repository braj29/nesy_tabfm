"""Hand-specified monotonicity constraints for TabArena's ``heloc`` dataset (FICO's Home
Equity Line of Credit / Explainable ML Challenge dataset).

Single-table, no relational structure -- the other 5 constraint types have nothing to bind
to here (no FK graph, no natural row groups), which is exactly the gap ``MonotonicityConstraint``
was added for. Each feature below was checked against the real target correlation before
being written down (``tests/test_heloc_ground_truth.py``), not assumed from the name:

- ``ExternalRiskEstimate`` (FICO's own consolidated risk score): corr = +0.46 with the
  "Good" outcome -- the strongest of any feature checked.
- ``NetFractionRevolvingBurden`` (revolving credit utilization): corr = -0.36.

Both signs match the domain-defensible direction (higher risk-estimate score -> better;
higher utilization -> worse), so a model whose *predictions* ever move the wrong way when
one of these is nudged in the "should help" direction is violating a real, checkable
relationship, not an invented one.
"""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from nesy_tabfm.constraints.dsl import MonotonicityConstraint


@dataclass(frozen=True)
class MonotonicityFeature:
    name: str
    delta: float  # signed perturbation, in the direction that should help the outcome
    clip_min: float | None
    clip_max: float | None
    note: str


FEATURES: list[MonotonicityFeature] = [
    MonotonicityFeature(
        "ExternalRiskEstimate",
        delta=10,
        clip_min=None,
        clip_max=94,  # observed max; don't extrapolate outside the training distribution
        note="FICO's own consolidated risk score; verified corr=+0.46 with Good outcome.",
    ),
    MonotonicityFeature(
        "NetFractionRevolvingBurden",
        delta=-20,
        clip_min=0,
        clip_max=None,
        note="revolving credit utilization; verified corr=-0.36 with Good outcome.",
    ),
]


def perturb(x: pd.DataFrame, feature: MonotonicityFeature) -> tuple[pd.DataFrame, pd.Series]:
    """A copy of ``x`` with ``feature.name`` nudged by ``feature.delta`` (clipped to the
    observed valid range), plus the boolean mask of rows where the feature wasn't already
    missing (a missing value can't be meaningfully perturbed, so those rows are excluded
    from the check, not perturbed to a fabricated value)."""
    valid_mask = x[feature.name].notna()
    x_pert = x.copy()
    new_val = x_pert.loc[valid_mask, feature.name] + feature.delta
    if feature.clip_min is not None:
        new_val = new_val.clip(lower=feature.clip_min)
    if feature.clip_max is not None:
        new_val = new_val.clip(upper=feature.clip_max)
    x_pert.loc[valid_mask, feature.name] = new_val
    return x_pert, valid_mask


def monotonicity_constraint(feature: MonotonicityFeature) -> MonotonicityConstraint:
    """Headline constraint for ``feature``: predicted creditworthiness must never decrease
    when ``feature`` is nudged in its "should help" direction (see ``perturb``). Expects
    tables ``"original"``/``"perturbed"`` with ``_row_id`` and ``y_pred`` columns."""
    return MonotonicityConstraint(
        f"{feature.name}_monotonic", id_col="_row_id", direction="non_decreasing"
    )
