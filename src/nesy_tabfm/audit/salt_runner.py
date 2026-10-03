"""Orchestrates one audit run against rel-salt: run a model on all 8 official autocomplete
tasks, check the DB-level ground-truth constraints plus the prediction-level consistency
checks (within-order for the 3 item-level targets, cross-task for the OBKG-derived FDs), and
write everything to ``results/salt/<model_name>/``.

Parallels ``audit/runner.py`` (the rel-f1 version) rather than sharing code with it: the two
datasets' constraint bindings differ enough (grouped-by-window vs. pairwise-within-order/
cross-task) that a forced shared abstraction would be premature after just two datasets.
"""

from __future__ import annotations

from pathlib import Path
from typing import Any

import pandas as pd

from nesy_tabfm.audit.report import write_findings
from nesy_tabfm.constraints.check import run_constraints, summarize
from nesy_tabfm.constraints.metrics import (
    pairwise_instance_violation_flags,
    stratified_rate,
    temporal_buckets,
    violation_error_correlation,
)
from nesy_tabfm.data.relbench_loader import as_frames, get_full_db, load_salt
from nesy_tabfm.progress import get_logger, setup_run_logging, timed, track
from nesy_tabfm.specs.salt import (
    ALL_TASKS,
    CONSISTENCY_TARGETS,
    CROSS_TASK_CHECKS,
    consistency_constraint,
    cross_task_fd_constraint,
    db_level_constraints,
    item_header_incoterms_cross_constraint,
    with_order_id,
)

TASK_BY_TARGET = {t.target_col: t for t in ALL_TASKS}
TASK_BY_NAME = {t.task_name: t for t in ALL_TASKS}


def run_audit(model_adapter: Any, model_name: str, results_root: Path | str) -> Path:
    out_dir = Path(results_root) / "salt" / model_name
    out_dir.mkdir(parents=True, exist_ok=True)
    setup_run_logging(out_dir)
    logger = get_logger()
    logger.info("rel-salt audit, model=%s, output=%s", model_name, out_dir)

    with timed("load rel-salt"):
        dataset = load_salt()
    # NOT get_model_visible_db(): that censors every table to CREATIONTIMESTAMP <=
    # test_timestamp, which is right for a *forecast* task (driver-position) but wrong here --
    # these are row-masking AutoCompleteTasks, so a test-split row's own CREATIONTIMESTAMP is
    # necessarily > test_timestamp, and that censoring would silently drop the row itself out
    # of its own entity table before build_features' FK joins ever run, turning every feature
    # into NaN and models into constant-class predictors (caught by checking accuracy after a
    # suspiciously-perfect 0% consistency result). RelBench's own AutoCompleteTask._get_table
    # uses the full db for its test split for exactly this reason. Using the full db here is
    # safe against target leakage because build_features only does row-scoped FK joins (this
    # row's own order/customer/address), never an aggregate over other rows -- the actual
    # target-column exclusion is build_features' remove_columns argument, not db censoring.
    with timed("build full db"):
        db_full = get_full_db(dataset)
    gt_tables = as_frames(db_full)

    gt_results = run_constraints(db_level_constraints(), gt_tables, progress=True, desc="rel-salt DB-level")
    db_level_summary = summarize(gt_results)
    db_level_summary.to_csv(out_dir / "db_level_violation_rates.csv", index=False)

    # -- run every task, keep predictions (renamed to f"{target_col.lower()}_pred") for the
    # within-order and cross-task checks below. One task's model failure (e.g. TabPFN has a
    # hard, unbypassable 160-class ceiling -- sales-group's 502 real classes exceed it) must
    # not lose every other task's already-computed results, so this mirrors
    # relarena_predict/predict.py's per-(model, task) try/except rather than letting one
    # exception abort the whole run silently mid-loop (write_findings would never be reached).
    predictions: dict[str, pd.DataFrame] = {}
    accuracy_rows = []
    failed_tasks: dict[str, str] = {}
    task_bar = track(ALL_TASKS, desc=f"rel-salt tasks [{model_name}]")
    for spec in task_bar:
        task_bar.set_postfix_str(spec.task_name)
        task = dataset.load_task(spec.task_name)
        try:
            with timed(f"{model_name} on {spec.task_name}"):
                preds = model_adapter.fit_predict(task, db_full)
        except Exception as e:  # noqa: BLE001 -- keep going across tasks; see comment above
            logger.error("%s FAILED: %s", spec.task_name, e)
            failed_tasks[spec.task_name] = str(e)
            continue
        preds.to_parquet(out_dir / f"{spec.task_name}_predictions.parquet")

        pred_col = f"{spec.target_col.lower()}_pred"
        preds = preds.rename(columns={"y_pred": pred_col})
        predictions[spec.task_name] = preds
        logger.info(
            "%s accuracy=%.4f on n=%d", spec.task_name, (preds["y_true"] == preds[pred_col]).mean(), len(preds)
        )
        accuracy_rows.append(
            {
                "task": spec.task_name,
                "target_col": spec.target_col,
                "n": len(preds),
                "accuracy": (preds["y_true"] == preds[pred_col]).mean(),
            }
        )
    if failed_tasks:
        pd.Series(failed_tasks, name="error").to_csv(out_dir / "failed_tasks.csv", header=True)
    accuracy_summary = pd.DataFrame(accuracy_rows)
    accuracy_summary.to_csv(out_dir / "accuracy_by_task.csv", index=False)

    prediction_results = []
    stratified: dict[str, pd.DataFrame] = {}
    correlations: dict[str, float] = {}

    # -- within-order pairwise consistency: the 3 item-level targets --------------------------
    for target_col in CONSISTENCY_TARGETS:
        spec = TASK_BY_TARGET[target_col]
        if spec.task_name not in predictions:
            continue  # that task's model run failed -- see failed_tasks.csv
        pred_col = f"{target_col.lower()}_pred"
        preds = predictions[spec.task_name]
        preds_with_order = with_order_id(preds, gt_tables["salesdocumentitem"])

        result = consistency_constraint(target_col).check({"predictions": preds_with_order})
        prediction_results.append(result)

        task = dataset.load_task(spec.task_name)
        flags = pairwise_instance_violation_flags(result, preds, id_col=task.entity_col)
        bucket = temporal_buckets(pd.to_datetime(preds[task.time_col]))
        stratified[f"{spec.task_name} violation rate by temporal bucket"] = stratified_rate(
            flags, bucket, group_name="temporal_bucket"
        )

        error = (preds["y_true"] != preds[pred_col]).astype(float)
        correlations[f"{spec.task_name}: violation vs. (1 - accuracy)"] = violation_error_correlation(
            flags, error
        )

    # -- cross-task OBKG-derived FD checks: two different tasks' predictions, same entity -----
    for name, det_task_name, det_col, dep_task_name, dep_col, join_col in CROSS_TASK_CHECKS:
        if det_task_name not in predictions or dep_task_name not in predictions:
            continue  # one side's model run failed -- see failed_tasks.csv
        det_preds = predictions[det_task_name][[join_col, f"{det_col.lower()}_pred"]]
        dep_preds = predictions[dep_task_name][[join_col, f"{dep_col.lower()}_pred"]]
        combined = det_preds.merge(dep_preds, on=join_col, how="inner")
        result = cross_task_fd_constraint(name, det_col, dep_col).check({"combined": combined})
        prediction_results.append(result)

    # -- cross-task, cross-entity-level check: item-incoterms vs. sales-incoterms -------------
    if "item-incoterms" in predictions and "sales-incoterms" in predictions:
        item_preds = with_order_id(predictions["item-incoterms"], gt_tables["salesdocumentitem"])
        header_preds = predictions["sales-incoterms"][["SALESDOCUMENT", "headerincotermsclassification_pred"]]
        combined_incoterms = item_preds.merge(header_preds, on="SALESDOCUMENT", how="inner")
        prediction_results.append(
            item_header_incoterms_cross_constraint().check({"combined": combined_incoterms})
        )

    prediction_level_summary = summarize(prediction_results)
    prediction_level_summary.to_csv(out_dir / "prediction_violation_rates.csv", index=False)
    for name, df in stratified.items():
        df.to_csv(out_dir / f"{name.replace(' ', '_')}.csv", index=False)

    logger.info("prediction-level results:\n%s", prediction_level_summary.to_string(index=False))
    findings_path = write_findings(
        out_dir,
        "rel-salt",
        model_name,
        db_level_summary,
        prediction_level_summary,
        stratified,
        correlations,
        accuracy=accuracy_summary,
    )
    logger.info("wrote %s", findings_path)
    return findings_path
