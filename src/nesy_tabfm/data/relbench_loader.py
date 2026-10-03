"""Thin wrapper around relbench's hub-backed loader.

relbench 3.x loads datasets from the Hugging Face Hub as ``org/repo[/subdir]``. RelBench's
own ``Dataset.get_db`` defaults to truncating rows past the dataset's ``test_timestamp`` --
correct for anything a *model* is allowed to see, but wrong for verifying constraints that
reference the future (e.g. "a season's final standings equal the sum of that season's race
points" needs rows past test_timestamp for late-season races in the test period). This module
exposes both explicitly so callers don't get this backwards by accident: ``get_full_db`` is
only ever for constraint ground-truth checks, never for model input.
"""

from __future__ import annotations

import pandas as pd
import relbench
from relbench.base import Database, Dataset

REL_F1_HUB_PATH = "stanford-star/relbench-v1/rel-f1"
SALT_HUB_PATH = "stanford-star/relbench-v2-extra/rel-salt"


def load_dataset(hub_path: str) -> Dataset:
    return relbench.load_dataset(hub_path)


def load_rel_f1() -> Dataset:
    return load_dataset(REL_F1_HUB_PATH)


def load_salt() -> Dataset:
    return load_dataset(SALT_HUB_PATH)


def get_full_db(dataset: Dataset) -> Database:
    """The complete database, including rows after test_timestamp. For checking constraints
    against ground truth only -- never as model input (it leaks the future)."""
    return dataset.get_db(upto_test_timestamp=False)


def get_model_visible_db(dataset: Dataset) -> Database:
    """The database truncated at test_timestamp: everything a model may see at inference
    time."""
    return dataset.get_db()


def as_frames(db: Database) -> dict[str, pd.DataFrame]:
    return {name: table.df for name, table in db.table_dict.items()}
