"""Loads TabArena's ``heloc`` dataset (FICO's Home Equity Line of Credit / Explainable ML
Challenge dataset) directly from OpenML -- no relbench, no relational structure, single
table.

FICO encodes missing values as negative sentinel codes (-7, -8, -9) rather than NaN, which
LightGBM/sklearn would otherwise silently treat as real (very low) feature values -- a real
methodological trap for this specific dataset, not a general concern, so it's handled once
here rather than left for every caller to rediscover.
"""

from __future__ import annotations

import openml
import pandas as pd

HELOC_OPENML_DATASET_ID = 46932
TARGET_COL = "RiskPerformance"


def load_heloc() -> tuple[pd.DataFrame, pd.Series]:
    """Returns ``(X, y)``: X with FICO's sentinel missing-value codes replaced by NaN, y as
    ``{0, 1}`` (1 = "Good")."""
    dataset = openml.datasets.get_dataset(HELOC_OPENML_DATASET_ID)
    x, y, _, _ = dataset.get_data(target=TARGET_COL)
    x_clean = x.where(x >= 0)
    y_bin = (y == "Good").astype(int)
    return x_clean, y_bin
