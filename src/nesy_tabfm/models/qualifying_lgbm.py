"""LightGBM baseline for rel-f1's ``qualifying-position`` task. The task table only carries
``[qualifyId, date, position]``, so ``driverId``/``constructorId``/``raceId`` are joined back
in from the ``qualifying`` table itself before featurizing (see ``fit_predict``).
"""

from __future__ import annotations

from typing import Any

import lightgbm as lgb
import pandas as pd

from nesy_tabfm.data.qualifying_features import build_features


class QualifyingLGBMAdapter:
    def __init__(self, **lgbm_kwargs: Any):
        self.lgbm_kwargs = {"n_estimators": 200, "num_leaves": 31, "verbose": -1, **lgbm_kwargs}

    def fit_predict(self, task: Any, db: Any, db_full: Any) -> pd.DataFrame:
        """``db`` (model-visible, truncated at test_timestamp) supplies the lag-feature
        history; ``db_full`` supplies entity *identity* only (which driver/constructor/race
        a given qualifyId belongs to -- public schedule/entry info, not the hidden position
        outcome). Test-split qualifyId rows don't exist in ``db`` at all (they're dated after
        the cutoff), so the identity lookup has to come from ``db_full``.
        """
        qualifying = db.table_dict["qualifying"].df
        results = db.table_dict["results"].df
        keys = db_full.table_dict["qualifying"].df[["qualifyId", "driverId", "constructorId", "raceId"]]
        entity_col, time_col, target_col = task.entity_col, task.time_col, task.target_col

        train_df = task.get_table("train").df.merge(keys, on=entity_col)
        test_df = task.get_table("test", mask_input_cols=False).df.merge(keys, on=entity_col)

        x_train = build_features(train_df, qualifying, results)
        x_test = build_features(test_df, qualifying, results)

        model = lgb.LGBMRegressor(**self.lgbm_kwargs)
        model.fit(x_train, train_df[target_col].to_numpy())
        y_pred = model.predict(x_test)

        return pd.DataFrame(
            {
                entity_col: test_df[entity_col].to_numpy(),
                time_col: test_df[time_col].to_numpy(),
                "raceId": test_df["raceId"].to_numpy(),
                "y_true": test_df[target_col].to_numpy(),
                "y_pred": y_pred,
            }
        )
