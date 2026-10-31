#!/usr/bin/env python
"""How often the data itself breaks each prediction-level constraint, on exactly the test items
the models are audited on: every audit check, run on the true labels instead of predictions.

    uv run python scripts/diagnostics/ground_truth_test_items.py

Only the ``y_true`` columns of saved prediction files are used, so which model wrote them does
not matter. Writes ``results/ground_truth_on_test_items.csv``.
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from nesy_tabfm.constraints.check import summarize
from nesy_tabfm.data.relbench_loader import as_frames, get_full_db, load_rel_f1
from nesy_tabfm.specs.rel_f1 import (
    position_domain_bounds,
    position_domain_constraint,
    top3_cardinality_constraint,
    top3_window_bounds,
)
from nesy_tabfm.specs.salt import (
    ALL_TASKS,
    CONSISTENCY_TARGETS,
    CROSS_TASK_CHECKS,
    consistency_constraint,
    cross_task_fd_constraint,
    item_header_incoterms_cross_constraint,
)

REPO = Path(__file__).resolve().parent.parent.parent
SALT_PREDS = REPO / "results_lgbm_seed42" / "salt" / "lightgbm"  # full test split, all 8 tasks
TEST_ITEMS = REPO / "results_ccd" / "lightgbm" / "salt" / "test_items.parquet"  # item ID -> order
RELF1_PREDS = REPO / "relarena_predict" / "results" / "tabpfn-rel-local"
TASK_BY_TARGET = {t.target_col: t for t in ALL_TASKS}


def truth_as_predictions(task_name: str, target_col: str) -> pd.DataFrame:
    df = pd.read_parquet(SALT_PREDS / f"{task_name}_predictions.parquet")
    return df.drop(columns=["y_pred"]).rename(columns={"y_true": f"{target_col.lower()}_pred"})


def salt_checks() -> list:
    items = pd.read_parquet(TEST_ITEMS)
    results = []
    for target in CONSISTENCY_TARGETS:
        preds = truth_as_predictions(TASK_BY_TARGET[target].task_name, target).merge(items, on="ID", how="left")
        if preds["SALESDOCUMENT"].isna().any():
            raise SystemExit(f"{target}: test items without an order id")
        results.append(consistency_constraint(target).check({"predictions": preds}))
    for name, det_task, det_col, dep_task, dep_col, join_col in CROSS_TASK_CHECKS:
        det = truth_as_predictions(det_task, det_col)[[join_col, f"{det_col.lower()}_pred"]]
        dep = truth_as_predictions(dep_task, dep_col)[[join_col, f"{dep_col.lower()}_pred"]]
        combined = det.merge(dep, on=join_col, how="inner")
        results.append(cross_task_fd_constraint(name, det_col, dep_col).check({"combined": combined}))
    item_inc = truth_as_predictions("item-incoterms", "ITEMINCOTERMSCLASSIFICATION").merge(items, on="ID", how="left")
    header_inc = truth_as_predictions("sales-incoterms", "HEADERINCOTERMSCLASSIFICATION")
    combined = item_inc.merge(header_inc[["SALESDOCUMENT", "headerincotermsclassification_pred"]], on="SALESDOCUMENT", how="inner")
    results.append(item_header_incoterms_cross_constraint().check({"combined": combined}))
    return results


def rel_f1_checks() -> list:
    dataset = load_rel_f1()
    gt = as_frames(get_full_db(dataset))
    top3 = pd.read_parquet(RELF1_PREDS / "driver-top3_predictions.parquet")
    top3 = top3.drop(columns=["y_pred"]).rename(columns={"y_true": "top3_pred"})
    top3["date"] = pd.to_datetime(top3["date"])
    bounds = top3_window_bounds(gt["qualifying"], top3["date"])
    top3_result = top3_cardinality_constraint().check(
        {"predictions": top3, "test_windows": bounds[["date"]], "top3_bounds": bounds}
    )
    pos = pd.read_parquet(RELF1_PREDS / "driver-position_predictions.parquet")
    pos = pos.drop(columns=["y_pred"]).rename(columns={"y_true": "position_pred"})
    pos["date"] = pd.to_datetime(pos["date"])
    days = dataset.load_task("driver-position").timedelta.days
    pos = pos.merge(position_domain_bounds(gt["results"], pos["date"], days), on="date", how="left")
    return [top3_result, position_domain_constraint().check({"predictions": pos})]


def main() -> None:
    table = pd.concat(
        [summarize(salt_checks()).assign(database="rel-salt"), summarize(rel_f1_checks()).assign(database="rel-f1")]
    )
    out = REPO / "results" / "ground_truth_on_test_items.csv"
    table.to_csv(out, index=False)
    print(table.to_string(index=False))
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
