"""Orchestrates one audit run against rel-f1: run a model on its 3 official tasks, check the
hand-specified constraints (DB-level ground truth + the two headline prediction-level
checks), compute violation-rate metrics stratified by temporal shift and rare class, and
write everything to ``results/rel_f1/<model_name>/``.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from nesy_tabfm.audit.report import write_findings
from nesy_tabfm.constraints.check import run_constraints, summarize
from nesy_tabfm.constraints.metrics import (
    instance_violation_flags,
    stratified_rate,
    temporal_buckets,
    violation_error_correlation,
)
from nesy_tabfm.data.relbench_loader import (
    as_frames,
    get_full_db,
    get_model_visible_db,
    load_rel_f1,
)
from nesy_tabfm.progress import get_logger, setup_run_logging, timed, track
from nesy_tabfm.specs.rel_f1 import (
    db_level_constraints,
    derive_db_level_tables,
    position_domain_bounds,
    position_domain_constraint,
    top3_cardinality_constraint,
    top3_window_bounds,
)

TASK_NAMES = ["driver-position", "driver-dnf", "driver-top3"]


def run_audit(model_adapter: Any, model_name: str, results_root: Path | str) -> Path:
    out_dir = Path(results_root) / "rel_f1" / model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    setup_run_logging(out_dir)
    logger = get_logger()
    logger.info("rel-f1 audit, model=%s, output=%s", model_name, out_dir)

    with timed("load rel-f1"):
        dataset = load_rel_f1()
        db_full = get_full_db(dataset)  # ground-truth only, never fed to a model
        db_visible = get_model_visible_db(dataset)  # what the model is allowed to see

    gt_tables = as_frames(db_full)
    gt_tables.update(derive_db_level_tables(db_full))
    gt_results = run_constraints(db_level_constraints(), gt_tables, progress=True, desc="rel-f1 DB-level")
    db_level_summary = summarize(gt_results)
    db_level_summary.to_csv(out_dir / "db_level_violation_rates.csv", index=False)

    predictions: dict[str, pd.DataFrame] = {}
    tasks: dict[str, Any] = {}
    for task_name in track(TASK_NAMES, desc="rel-f1 tasks"):
        task = dataset.load_task(task_name)
        tasks[task_name] = task
        with timed(f"predictions for {task_name}"):
            preds = model_adapter.fit_predict(task, db_visible)
        preds["_row_id"] = range(len(preds))
        predictions[task_name] = preds
        preds.to_parquet(out_dir / f"{task_name}_predictions.parquet")

    top3_pred = predictions["driver-top3"].rename(columns={"y_pred": "top3_pred"})
    top3_pred["date"] = pd.to_datetime(top3_pred["date"])
    bounds = top3_window_bounds(gt_tables["qualifying"], top3_pred["date"])
    top3_result = top3_cardinality_constraint().check(
        {
            "predictions": top3_pred,
            "test_windows": bounds[["date"]],
            "top3_bounds": bounds,
        }
    )

    position_pred = predictions["driver-position"].rename(columns={"y_pred": "position_pred"})
    position_pred["date"] = pd.to_datetime(position_pred["date"])
    position_timedelta_days = tasks["driver-position"].timedelta.days
    position_bounds = position_domain_bounds(
        gt_tables["results"], position_pred["date"], position_timedelta_days
    )
    position_pred = position_pred.merge(position_bounds, on="date", how="left")
    position_result = position_domain_constraint().check({"predictions": position_pred})

    prediction_level_summary = summarize([top3_result, position_result])
    prediction_level_summary.to_csv(out_dir / "prediction_violation_rates.csv", index=False)

    # --- stratified metrics: does the violation rate concentrate in the temporal-shift
    # tail, or on the rare (minority) class? and does it correlate with prediction error? ---
    top3_pred["error"] = (top3_pred["y_true"] != top3_pred["top3_pred"]).astype(float)
    top3_pred["rare_class"] = top3_pred["y_true"].astype(bool)
    top3_flags = instance_violation_flags(top3_result, top3_pred, join_cols=["date"])
    top3_bucket = temporal_buckets(top3_pred["date"])

    position_pred["error"] = (position_pred["y_true"] - position_pred["position_pred"]).abs()
    position_flags = instance_violation_flags(position_result, position_pred, join_cols=["_row_id"])
    position_bucket = temporal_buckets(pd.to_datetime(position_pred["date"]))

    stratified = {
        "driver-top3 violation rate by temporal bucket": stratified_rate(
            top3_flags, top3_bucket, group_name="temporal_bucket"
        ),
        "driver-top3 violation rate by rare class (true top3=1)": stratified_rate(
            top3_flags, top3_pred["rare_class"], group_name="rare_class"
        ),
        "driver-position violation rate by temporal bucket": stratified_rate(
            position_flags, position_bucket, group_name="temporal_bucket"
        ),
    }
    for name, df in stratified.items():
        df.to_csv(out_dir / f"{name.replace(' ', '_')}.csv", index=False)

    correlations = {
        "driver-top3: violation vs. (1 - accuracy)": violation_error_correlation(
            top3_flags, top3_pred["error"]
        ),
        "driver-position: violation vs. |error|": violation_error_correlation(
            position_flags, position_pred["error"]
        ),
    }

    logger.info("prediction-level results:\n%s", prediction_level_summary.to_string(index=False))
    findings_path = write_findings(
        out_dir, "rel-f1", model_name, db_level_summary, prediction_level_summary, stratified, correlations
    )
    logger.info("wrote %s", findings_path)
    return findings_path
