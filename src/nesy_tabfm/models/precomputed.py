"""Adapter for predictions produced outside this repo (e.g. RelGNN trained on a cluster).

Expects a directory containing ``<task-name>_predictions.parquet`` files with columns
``[entity_col, time_col, "y_true", "y_pred"]`` -- the same schema ``fit_predict`` returns for
every other adapter, so the audit runner doesn't need to special-case where predictions came
from.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd


class PrecomputedAdapter:
    def __init__(self, predictions_dir: str | Path):
        self.predictions_dir = Path(predictions_dir)

    def fit_predict(self, task: Any, db: Any) -> pd.DataFrame:
        path = self.predictions_dir / f"{task.name}_predictions.parquet"
        if not path.exists():
            raise FileNotFoundError(
                f"expected precomputed predictions at {path} with columns "
                f"[{task.entity_col!r}, {task.time_col!r}, 'y_true', 'y_pred']"
            )
        return pd.read_parquet(path)
