"""TabPFN baseline for HELOC. Same interface as ``heloc_lgbm.HelocLGBMAdapter`` -- live
``predict_proba_positive`` on arbitrary rows, needed for the monotonicity constraint's
synthetic counterfactuals. Needs a one-time ``TABPFN_TOKEN`` license (see
``relarena_predict/README.md``).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from tabpfn import TabPFNClassifier


class HelocTabPFNAdapter:
    def __init__(self, **tabpfn_kwargs: Any):
        self.tabpfn_kwargs = tabpfn_kwargs
        self.model: TabPFNClassifier | None = None

    def fit(self, x_train: pd.DataFrame, y_train: pd.Series) -> "HelocTabPFNAdapter":
        self.model = TabPFNClassifier(**self.tabpfn_kwargs)
        self.model.fit(x_train, y_train.to_numpy())
        return self

    def predict_proba_positive(self, x: pd.DataFrame) -> np.ndarray:
        assert self.model is not None, "call fit() first"
        proba = self.model.predict_proba(x)
        positive_idx = list(self.model.classes_).index(1)
        return proba[:, positive_idx]
