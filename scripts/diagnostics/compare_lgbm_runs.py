#!/usr/bin/env python
"""Confirm whether two LightGBM rel-salt runs (results/ vs results_cluster/) trained the
same model, by comparing their per-row predictions directly rather than just their
aggregate violation-rate CSVs.

    uv run python scripts/diagnostics/compare_lgbm_runs.py

Prints, per item-level task: how many test rows both runs share, how many of those rows got
a *different* predicted label between the two runs, each run's accuracy, and each run's
predicted-label distribution (to show whether one run collapsed to a dominant class and the
other didn't).
"""

from __future__ import annotations

from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent.parent
RUN_A = REPO_ROOT / "results" / "salt" / "lightgbm"
RUN_B = REPO_ROOT / "results_cluster" / "salt" / "lightgbm"

TASKS = ["item-plant", "item-shippoint", "item-incoterms"]


def load(run_dir: Path, task: str) -> pd.DataFrame:
    return pd.read_parquet(run_dir / f"{task}_predictions.parquet")


def compare_task(task: str) -> None:
    a = load(RUN_A, task)
    b = load(RUN_B, task)

    entity_col = [c for c in a.columns if c not in ("y_true", "y_pred") and "time" not in c.lower()]
    time_col = [c for c in a.columns if "time" in c.lower()]
    key_cols = entity_col + time_col
    if not key_cols:
        key_cols = [c for c in a.columns if c not in ("y_true", "y_pred")]

    merged = a.merge(b, on=key_cols, suffixes=("_a", "_b"), how="inner")

    print(f"\n=== {task} ===")
    print(f"run A (results/)         rows={len(a)}  accuracy={(a['y_true'] == a['y_pred']).mean():.4%}")
    print(f"run B (results_cluster/) rows={len(b)}  accuracy={(b['y_true'] == b['y_pred']).mean():.4%}")
    print(f"matched on key columns {key_cols}: {len(merged)} rows")

    if len(merged) == 0:
        print("  no overlapping rows on these key columns -- cannot compare predictions directly.")
        return

    # y_true should be identical between runs on the same row (same ground truth); if not,
    # the two runs are not even looking at the same test rows.
    mismatched_truth = (merged["y_true_a"] != merged["y_true_b"]).sum()
    print(f"  rows where y_true differs between runs: {mismatched_truth} (should be 0)")

    differ = (merged["y_pred_a"] != merged["y_pred_b"]).sum()
    print(f"  rows where y_pred DIFFERS between the two runs: {differ} ({differ / len(merged):.4%})")

    print("  run A predicted-label distribution (top 5):")
    print(a["y_pred"].value_counts().head(5).to_string())
    print("  run B predicted-label distribution (top 5):")
    print(b["y_pred"].value_counts().head(5).to_string())


def main() -> None:
    for task in TASKS:
        a_path = RUN_A / f"{task}_predictions.parquet"
        b_path = RUN_B / f"{task}_predictions.parquet"
        if not a_path.exists() or not b_path.exists():
            print(f"skipping {task}: missing {a_path if not a_path.exists() else b_path}")
            continue
        compare_task(task)


if __name__ == "__main__":
    main()
