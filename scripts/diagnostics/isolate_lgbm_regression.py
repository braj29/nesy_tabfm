#!/usr/bin/env python
"""Isolate which LightGBM kwarg (random_state / deterministic / force_row_wise) is
responsible for sales-group's accuracy collapsing to ~1-2% after the determinism fix, when
it was the *most stable* task (~86% both times) across the two original unseeded runs.

Runs sales-group only, four ways, and reports accuracy for each:
    (a) no special kwargs (original unseeded behaviour)
    (b) random_state=0 only
    (c) random_state=0 + deterministic=True
    (d) random_state=0 + deterministic=True + force_row_wise=True (current fix)

    uv run python scripts/diagnostics/isolate_lgbm_regression.py
"""

from __future__ import annotations

from nesy_tabfm.data.relbench_loader import get_full_db, load_salt
from nesy_tabfm.data.salt_features import build_features_for_task
from nesy_tabfm.models.salt_lgbm import _cast_categorical
import lightgbm as lgb

VARIANTS = {
    "(a) unseeded (original)": {},
    "(b) random_state only": {"random_state": 0},
    "(c) random_state + deterministic": {"random_state": 0, "deterministic": True},
    "(d) random_state + deterministic + force_row_wise (current fix)": {
        "random_state": 0,
        "deterministic": True,
        "force_row_wise": True,
    },
}


def main() -> None:
    print("loading rel-salt...")
    dataset = load_salt()
    db = get_full_db(dataset)
    tables = {name: t.df for name, t in db.table_dict.items()}
    task = dataset.load_task("sales-group")

    train_df = task.get_table("train").df
    test_df = task.get_table("test", mask_input_cols=False).df
    entity_col, target_col = task.entity_col, task.target_col

    print("building features...")
    x_train = build_features_for_task(train_df, tables, entity_col=entity_col, remove_columns=task.remove_columns)
    x_test = build_features_for_task(test_df, tables, entity_col=entity_col, remove_columns=task.remove_columns)
    _cast_categorical(x_train, x_test)
    y_train = train_df[target_col].to_numpy()
    y_true = test_df[target_col].to_numpy()

    for name, kwargs in VARIANTS.items():
        full_kwargs = {"n_estimators": 200, "num_leaves": 63, "verbose": -1, **kwargs}
        model = lgb.LGBMClassifier(**full_kwargs)
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        acc = (y_pred == y_true).mean()
        print(f"{name}: accuracy={acc:.4%}  kwargs={kwargs}")


if __name__ == "__main__":
    main()
