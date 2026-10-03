"""TabPFN v3 baseline for rel-salt's autocomplete tasks -- same flat feature join as
``SaltLGBMAdapter`` (``data/salt_features.py``), swapping LightGBM for ``TabPFNClassifier``.

TabPFN's officially supported limits (``tabpfn.inference_config``) cap classification at 10
classes and ~10,000 training rows on CPU; rel-salt's 8 targets have 13-502 classes and up to
~1.6M rows. So this adapter (a) subsamples train/test down to ``train_sample_size``/
``test_sample_size`` rows (plain random, not stratified -- infeasible to stratify a 10k
budget across up to 502 classes) and (b) passes ``ignore_pretraining_limits=True`` to run
past the 10-class cap. Both are explicitly *unsupported* regimes for TabPFN, not documented
ones -- this is a feasibility trial, not a tuned or fully-powered baseline, and should be
reported as such.

A first trial using LightGBM's exact feature set (``salt_features.build_features``) collapsed
to predicting a single constant class on ``item-plant``/``item-shippoint`` (4.6% accuracy) --
not a real finding, just model failure, traced to ``PRODUCT``/``SOLDTOPARTY``/``SHIPTOPARTY``/
``BILLTOPARTY``/``PAYERPARTY``: LightGBM-friendly at the full 1.6M-row training set (repeat
enough for tree splits to use), but near-unique per-row IDs once subsampled to TabPFN's
~3,000-10,000-row budget (e.g. 2,060 distinct ``PRODUCT`` values in 3,000 sampled rows) --
pure noise at that scale, and enough of it to starve the model of any usable signal. So
``ID_LIKE_COLS`` are dropped for TabPFN specifically, leaving only the business-classification
codes (item category, sales org/office/group, division, company code, currency, country/
region, incoterms, payment terms, shipping condition) -- a smaller, TabPFN-specific feature
set than LightGBM's, not the same inputs re-run through a different model. Comparisons
between the two adapters' violation rates should account for that difference, not just the
model swap.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
from tabpfn import TabPFNClassifier

from nesy_tabfm.data.salt_features import build_features_for_task
from nesy_tabfm.progress import get_logger, timed

#: Near-unique-per-row identifier columns that are informative for LightGBM (trained on the
#: full ~1.6M-row table, where each value repeats often enough for tree splits to exploit) but
#: are noise for TabPFN's much smaller required training sample -- see module docstring.
ID_LIKE_COLS = ["PRODUCT", "SOLDTOPARTY", "SHIPTOPARTY", "BILLTOPARTY", "PAYERPARTY"]


class SaltTabPFNAdapter:
    def __init__(
        self,
        train_sample_size: int | None = 8000,
        test_sample_size: int | None = 1000,
        seed: int = 0,
        **tabpfn_kwargs: Any,
    ):
        self.train_sample_size = train_sample_size
        self.test_sample_size = test_sample_size
        self.seed = seed
        self.tabpfn_kwargs = {"ignore_pretraining_limits": True, "device": "cpu", **tabpfn_kwargs}

    def fit_predict(self, task: Any, db: Any) -> pd.DataFrame:
        tables = {name: t.df for name, t in db.table_dict.items()}
        entity_col, time_col, target_col = task.entity_col, task.time_col, task.target_col

        train_df = task.get_table("train").df
        test_df = task.get_table("test", mask_input_cols=False).df

        if self.train_sample_size is not None and len(train_df) > self.train_sample_size:
            train_df = train_df.sample(n=self.train_sample_size, random_state=self.seed).reset_index(drop=True)
        if self.test_sample_size is not None and len(test_df) > self.test_sample_size:
            test_df = test_df.sample(n=self.test_sample_size, random_state=self.seed).reset_index(drop=True)

        remove_columns = task.remove_columns
        get_logger().info(
            "%s: TabPFN context=%d rows, test=%d rows, %d classes in context",
            target_col, len(train_df), len(test_df), train_df[target_col].nunique(),
        )
        with timed("build features"):
            x_train = build_features_for_task(train_df, tables, entity_col=entity_col, remove_columns=remove_columns)
            x_test = build_features_for_task(test_df, tables, entity_col=entity_col, remove_columns=remove_columns)

        drop_cols = [c for c in ID_LIKE_COLS if c in x_train.columns]
        x_train = x_train.drop(columns=drop_cols)
        x_test = x_test.drop(columns=drop_cols)

        # Every feature here is a categorical SAP business code (see salt_features.py) -- cast
        # to `category`, aligning test's categories to train's, same convention as
        # SaltLGBMAdapter (a category unseen in training becomes NaN in test rather than
        # silently leaking a train-only category as a shortcut).
        for col in x_train.columns:
            train_cat = pd.Categorical(x_train[col])
            x_train[col] = train_cat
            x_test[col] = pd.Categorical(x_test[col], categories=train_cat.categories)

        y_train = train_df[target_col].to_numpy()
        model = TabPFNClassifier(**self.tabpfn_kwargs)
        with timed("TabPFN fit"):
            model.fit(x_train, y_train)
        with timed(f"TabPFN predict ({len(x_test)} rows)"):
            y_pred = model.predict(x_test)

        return pd.DataFrame(
            {
                entity_col: test_df[entity_col].to_numpy(),
                time_col: test_df[time_col].to_numpy(),
                "y_true": test_df[target_col].to_numpy(),
                "y_pred": y_pred,
            }
        )
