"""LightGBM baseline for HELOC. A different, simpler interface than the RelBench task
adapters (``fit``/``predict_proba_positive`` on plain feature frames, not
``fit_predict(task, db)``) because the monotonicity constraint needs live predictions on
synthetic counterfactual rows, not just a static test-set prediction table.
"""

from __future__ import annotations

from typing import Any

import lightgbm as lgb
import numpy as np
import pandas as pd


class HelocLGBMAdapter:
    def __init__(self, **lgbm_kwargs: Any):
        self.lgbm_kwargs = {"n_estimators": 200, "num_leaves": 31, "verbose": -1, **lgbm_kwargs}
        self.model: lgb.LGBMClassifier | None = None

    def fit(self, x_train: pd.DataFrame, y_train: pd.Series) -> "HelocLGBMAdapter":
        self.model = lgb.LGBMClassifier(**self.lgbm_kwargs)
        self.model.fit(x_train, y_train.to_numpy())
        return self

    def predict_proba_positive(self, x: pd.DataFrame) -> np.ndarray:
        assert self.model is not None, "call fit() first"
        return self.model.predict_proba(x)[:, 1]
