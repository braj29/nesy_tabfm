"""Hand-specified constraints for rel-f1's ``qualifying-position``/``results-position``
autocomplete tasks -- these predict the literal ``position`` column of an individual
``qualifying``/``results`` row (entity_col is the row's own id), unlike ``driver-top3``/
``driver-position`` which predict a 30/60-day forward-looking window aggregate. That makes a
genuine **permutation constraint** checkable directly: verified against real ground truth
(``tests/test_rel_f1_autocomplete_ground_truth.py``) -- 0 of 271 qualifying-position test
races have two drivers assigned the same position, out of ~18-24 drivers per race.

Continuous regression predictions are rounded to the nearest integer before checking for
duplicates (two predictions of 4.4 and 4.6 aren't "the same position" until rounded) -- this
is a real, if minor, design choice: it means a violation only registers when the model's
*rounded* output collides, not a raw floating-point tie, which would almost never happen and
would make the constraint vacuous.
"""

from __future__ import annotations

import pandas as pd

from nesy_tabfm.constraints.dsl import Cardinality


def with_race_id(predictions: pd.DataFrame, session_table: pd.DataFrame, entity_col: str) -> pd.DataFrame:
    """Join ``raceId`` onto a predictions table keyed by the session's own row id (``qualifyId``
    or ``resultId``) -- the official task table doesn't carry it, but the permutation check
    needs it to group by race."""
    return predictions.merge(session_table[[entity_col, "raceId"]], on=entity_col, how="left")


def position_no_duplicates_constraint(name: str) -> Cardinality:
    """Headline: no two predictions in the same race may round to the same position. Expects
    a ``"predictions"`` table with ``raceId`` and ``position_pred_rounded`` columns."""
    return Cardinality(
        name,
        table="predictions",
        group_cols=["raceId", "position_pred_rounded"],
        max=1,
    )
