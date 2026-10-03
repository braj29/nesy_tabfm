"""LightGBM baseline for rel-salt's item-level multiclass classification tasks
(``item-plant``, ``item-shippoint``, ...).

Deliberately simple -- no hyperparameter tuning, no DFS-style feature synthesis (see
``data/salt_features.py``): every feature here is a categorical SAP business code, so
they're cast to pandas ``category`` dtype and handed to LightGBM's native categorical
handling. This isn't meant to be a strong baseline, just a real one, so the constraint audit
has real predictions to check.
"""

from __future__ import annotations

from typing import Any

import lightgbm as lgb
import pandas as pd

from nesy_tabfm.data.salt_features import build_features_for_task
from nesy_tabfm.progress import get_logger, timed


def _cast_categorical(x_train: pd.DataFrame, x_test: pd.DataFrame) -> None:
    """In place: cast every feature column to `category`, aligning test's categories to
    train's (a category unseen in training becomes NaN in test, which LightGBM handles
    natively -- never a training-only category leaking a shortcut into test). Iterates
    x_train's own columns, not a fixed list -- the feature set varies per task (see
    build_features' remove_columns)."""
    for col in x_train.columns:
        train_cat = pd.Categorical(x_train[col])
        x_train[col] = train_cat
        x_test[col] = pd.Categorical(x_test[col], categories=train_cat.categories)


class SaltLGBMAdapter:
    def __init__(self, **lgbm_kwargs: Any):
        self.lgbm_kwargs = {"n_estimators": 200, "num_leaves": 63, "verbose": -1, **lgbm_kwargs}

    def fit_predict(self, task: Any, db: Any) -> pd.DataFrame:
        tables = {name: t.df for name, t in db.table_dict.items()}
        entity_col, time_col, target_col = task.entity_col, task.time_col, task.target_col

        train_df = task.get_table("train").df
        test_df = task.get_table("test", mask_input_cols=False).df

        remove_columns = task.remove_columns
        get_logger().info(
            "%s: train=%d test=%d rows, %d classes", task.target_col, len(train_df), len(test_df), train_df[target_col].nunique()
        )
        with timed("build features"):
            x_train = build_features_for_task(train_df, tables, entity_col=entity_col, remove_columns=remove_columns)
        x_test = build_features_for_task(test_df, tables, entity_col=entity_col, remove_columns=remove_columns)
        _cast_categorical(x_train, x_test)

        y_train = train_df[target_col].to_numpy()
        model = lgb.LGBMClassifier(**self.lgbm_kwargs)
        with timed(f"LightGBM fit ({len(x_train)} rows, {x_train.shape[1]} features)"):
            model.fit(x_train, y_train)
        with timed(f"LightGBM predict ({len(x_test)} rows)"):
            y_pred = model.predict(x_test)

        return pd.DataFrame(
            {
                entity_col: test_df[entity_col].to_numpy(),
                time_col: test_df[time_col].to_numpy(),
                "y_true": test_df[target_col].to_numpy(),
                "y_pred": y_pred,
            }
        )
