"""Standardized interface every model adapter implements, so the audit runner (and the
constraint specs) don't need to know which model produced a prediction table.

Kept free of any relbench/torch import so the constraint engine and its tests (the ``dev``
extra) never need the heavier ``tabpfn``/``gnn`` extras installed.
"""

from __future__ import annotations

from typing import Any, Protocol

import pandas as pd


class ModelAdapter(Protocol):
    """``fit_predict`` takes a relbench ``Task`` (has ``.entity_col``/``.time_col``/
    ``.target_col``/``.get_table(split)``) and the model-visible ``Database`` (already
    truncated to what the model may see -- the caller is responsible for that, not the
    adapter), and returns a DataFrame with exactly these columns, one row per test instance:
    ``[entity_col, time_col, "y_true", "y_pred"]``. For classification, ``y_pred`` is the
    hard 0/1 prediction (the constraint specs threshold on it directly)."""

    def fit_predict(self, task: Any, db: Any) -> pd.DataFrame: ...
